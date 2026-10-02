import CoreGraphics
import Foundation
import MLX
import MLXNN

enum RIFEEngineError: Error, LocalizedError {
    case missingFile(String)
    case emptyWeights(String)
    case imagePrepFailed
    case sizeMismatch
    case decodeOutputFailed

    var errorDescription: String? {
        switch self {
        case .missingFile(let f):  return "RIFE: required file not found: \(f)"
        case .emptyWeights(let f): return "RIFE: no arrays in '\(f)'"
        case .imagePrepFailed:     return "RIFE: failed to prepare an input image"
        case .sizeMismatch:        return "RIFE: the two frames must be the same size"
        case .decodeOutputFailed:  return "RIFE: interpolation produced no image"
        }
    }
}

/// Orchestrates RIFE 4.25 frame interpolation: load the tiny (~23 MB) checkpoint fresh
/// per call (no cross-call cache — the model is small enough that reloading is cheap
/// and it avoids any shared-mutable-state concerns), pad both frames to a multiple of
/// 64 (`IFNet`'s coarsest scale, 1/16, needs H,W divisible by 16; padding to 64 covers
/// every `scale_list` entry with margin, matching upstream `RIFE_HDv3.Model.inference`),
/// run `RIFENet`, crop back to the original size.
nonisolated enum RIFEEngine {
    private static let scaleList: [Double] = [16, 8, 4, 2, 1]
    private static let padTo = 64

    nonisolated static func interpolate(
        imageA: CGImage,
        imageB: CGImage,
        timestep: Double = 0.5,
        modelID: String,
        progress: @Sendable (Double) -> Void
    ) async throws -> CGImage {
        let dir = ModelStore.shared.directory(forModelID: modelID)
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw StageError.modelNotInstalled(id: modelID)
        }
        guard imageA.width == imageB.width, imageA.height == imageB.height else {
            throw RIFEEngineError.sizeMismatch
        }
        let trueW = imageA.width, trueH = imageA.height

        progress(0.05)
        guard let a = cgImageToTensor(imageA), let b = cgImageToTensor(imageB) else {
            throw RIFEEngineError.imagePrepFailed
        }

        let net = try loadModel(dir: dir)
        progress(0.3)

        let padH = roundUp(trueH, to: padTo), padW = roundUp(trueW, to: padTo)
        let x = concatenated([pad(a, toHeight: padH, width: padW), pad(b, toHeight: padH, width: padW)], axis: -1)

        let merged = net(x, timestep: timestep, scaleList: scaleList)
        eval(merged)
        progress(0.9)

        let cropped = merged[0..., 0 ..< trueH, 0 ..< trueW, 0...]
        guard let out = tensorToCGImage(cropped) else {
            throw RIFEEngineError.decodeOutputFailed
        }
        progress(1.0)
        return out
    }

    // MARK: - Model loading

    private static func loadModel(dir: URL) throws -> RIFENet {
        let url = dir.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw RIFEEngineError.missingFile("model.safetensors")
        }
        let raw = try MLX.loadArrays(url: url)
        guard !raw.isEmpty else { throw RIFEEngineError.emptyWeights("model.safetensors") }
        let net = RIFENet()
        try net.update(parameters: ModuleParameters.unflattened(sanitize(raw)), verify: .none)
        eval(net.parameters())
        return net
    }

    /// PyTorch `Conv2d.weight` is `(O,I,kH,kW)`; `ConvTranspose2d.weight` swaps I/O to
    /// `(I,O,kH,kW)`. MLX's `Conv2d`/`ConvTransposed2d` both want channel-last
    /// `(O,kH,kW,I)`. `lastconv` (one per `IFBlock`) and `encode.cnn3` are the
    /// checkpoint's only `ConvTranspose2d` layers; every other 4-D weight is a plain
    /// `Conv2d`. `beta` (ResConv's learned scale) is already NHWC-native — untouched.
    private static func sanitize(_ raw: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for (key, value) in raw {
            guard value.ndim == 4 else { out[key] = value; continue }
            if key.contains("lastconv") || key.hasSuffix("cnn3.weight") {
                out[key] = value.transposed(1, 2, 3, 0)
            } else if key.hasSuffix("conv.weight") || key.hasSuffix("cnn0.weight")
                        || key.hasSuffix("cnn1.weight") || key.hasSuffix("cnn2.weight") {
                out[key] = value.transposed(0, 2, 3, 1)
            } else {
                out[key] = value
            }
        }
        return out
    }

    // MARK: - Padding

    private static func roundUp(_ n: Int, to multiple: Int) -> Int {
        (n + multiple - 1) / multiple * multiple
    }

    /// Zero-pads on the bottom/right only (matches upstream), NHWC `[N,H,W,C]`.
    private static func pad(_ x: MLXArray, toHeight h: Int, width w: Int) -> MLXArray {
        let curH = x.dim(1), curW = x.dim(2)
        guard curH != h || curW != w else { return x }
        return MLX.padded(x, widths: [.init(0), .init((0, h - curH)), .init((0, w - curW)), .init(0)])
    }

    // MARK: - Image conversion
    //
    // NHWC `[1,H,W,3]` float32 in `[0,1]` — RIFE's native layout, so (unlike
    // SeedVR2Engine's NCHW/[-1,1]/bfloat16 conversion) no transpose is needed here,
    // only the CGContext pixel extraction/write-back, adapted from
    // `SeedVR2Engine.cgImageToTensor`/`tensorToCGImage`.

    private static func cgImageToTensor(_ src: CGImage) -> MLXArray? {
        let w = src.width, h = src.height
        guard w > 0, h > 0 else { return nil }
        var bytes = [UInt8](repeating: 0, count: h * w * 4)
        guard let ctx = CGContext(
            data: &bytes, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }
        ctx.draw(src, in: CGRect(x: 0, y: 0, width: w, height: h))
        var rgb = [Float](repeating: 0, count: h * w * 3)
        for i in 0 ..< h * w {
            rgb[i * 3 + 0] = Float(bytes[i * 4 + 0]) / 255.0
            rgb[i * 3 + 1] = Float(bytes[i * 4 + 1]) / 255.0
            rgb[i * 3 + 2] = Float(bytes[i * 4 + 2]) / 255.0
        }
        return MLXArray(rgb, [1, h, w, 3])
    }

    /// `t`: `[1,H,W,3]` in `[0,1]` → `CGImage`.
    private static func tensorToCGImage(_ t: MLXArray) -> CGImage? {
        let frame = t[0, 0..., 0..., 0...]   // [H,W,3]
        let clipped = MLX.clip(frame * 255.0, min: MLXArray(Float(0)), max: MLXArray(Float(255)))
        let pixels = clipped.asType(.uint8)
        let H = pixels.dim(0), W = pixels.dim(1)
        let arr: [UInt8] = pixels.asArray(UInt8.self)
        guard let provider = CGDataProvider(data: Data(arr) as CFData) else { return nil }
        return CGImage(
            width: W, height: H,
            bitsPerComponent: 8, bitsPerPixel: 24,
            bytesPerRow: W * 3,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: 0),
            provider: provider,
            decode: nil, shouldInterpolate: false,
            intent: .defaultIntent)
    }
}
