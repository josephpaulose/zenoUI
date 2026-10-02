import Foundation
import MLX
import MLXNN

// MARK: - RIFE 4.25 (IFNet) — MLX port
//
// Transcribed from the upstream `xocialize/rife-mlx` MLX (Python) port of Practical-RIFE
// 4.25 (MIT), itself a from-scratch MLX port with no PyTorch dependency. Architecture,
// channel counts and the warp/grid-sample math below are copied 1:1 from that source
// (`rife_mlx/model/IFNet_HDv3.py`, `warplayer.py`, `grid_sample.py`), fetched and read in
// full before writing this port — not guessed. NHWC throughout, matching MLX's native
// conv layout; the upstream repo is NHWC too.

/// Conv2d + LeakyReLU(0.2). Mirrors upstream `Conv`. Checkpoint key: `<prefix>.conv.weight`.
nonisolated final class RIFEConvBlock: Module {
    @ModuleInfo var conv: Conv2d

    init(_ inputChannels: Int, _ outputChannels: Int, kernelSize: Int = 3, stride: Int = 1, padding: Int = 1) {
        _conv.wrappedValue = Conv2d(
            inputChannels: inputChannels, outputChannels: outputChannels,
            kernelSize: .init(kernelSize), stride: .init(stride), padding: .init(padding))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        leakyRelu(conv(x), negativeSlope: 0.2)
    }
}

/// `lrelu(conv(x) * beta + x)`. Mirrors upstream `ResConv`. Keys: `conv.weight/bias`, `beta`.
nonisolated final class RIFEResConv: Module {
    @ModuleInfo var conv: Conv2d
    @ParameterInfo(key: "beta") var beta: MLXArray

    init(_ channels: Int) {
        _conv.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: .init(3), stride: .init(1), padding: .init(1))
        _beta.wrappedValue = MLXArray.ones([1, 1, 1, channels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        leakyRelu(conv(x) * beta + x, negativeSlope: 0.2)
    }
}

/// Feature encoder, run once on each full-res input image. Mirrors upstream `Head`.
/// `cnn0..cnn2` are `Conv2d`, `cnn3` is a `ConvTransposed2d` that restores the resolution
/// `cnn0`'s stride-2 dropped — net effect: same H,W as the input, 4 output channels.
nonisolated final class RIFEHead: Module {
    @ModuleInfo var cnn0: Conv2d
    @ModuleInfo var cnn1: Conv2d
    @ModuleInfo var cnn2: Conv2d
    @ModuleInfo var cnn3: ConvTransposed2d

    override init() {
        _cnn0.wrappedValue = Conv2d(inputChannels: 3, outputChannels: 16, kernelSize: .init(3), stride: .init(2), padding: .init(1))
        _cnn1.wrappedValue = Conv2d(inputChannels: 16, outputChannels: 16, kernelSize: .init(3), stride: .init(1), padding: .init(1))
        _cnn2.wrappedValue = Conv2d(inputChannels: 16, outputChannels: 16, kernelSize: .init(3), stride: .init(1), padding: .init(1))
        _cnn3.wrappedValue = ConvTransposed2d(inputChannels: 16, outputChannels: 4, kernelSize: .init(4), stride: .init(2), padding: .init(1))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = leakyRelu(cnn0(x), negativeSlope: 0.2)
        h = leakyRelu(cnn1(h), negativeSlope: 0.2)
        h = leakyRelu(cnn2(h), negativeSlope: 0.2)
        return cnn3(h)   // no activation — matches upstream
    }
}

/// One coarse-to-fine flow-estimation block. Mirrors upstream `IFBlock(in_planes, c)`.
/// `conv0` downsamples ×4 (two stride-2 `RIFEConvBlock`s), 8× `RIFEResConv` at that
/// resolution, `lastconv` (`ConvTransposed2d`, stride 2) + pixel-shuffle(r=2) upsamples
/// ×4 back — net resolution-preserving relative to this block's (already `1/scale`
/// downsampled) input. The caller rescales in/out by `scale`.
nonisolated final class RIFEBlock: Module {
    @ModuleInfo var conv0: [RIFEConvBlock]
    @ModuleInfo var convblock: [RIFEResConv]
    @ModuleInfo var lastconv: ConvTransposed2d

    init(inPlanes: Int, channels: Int) {
        _conv0.wrappedValue = [
            RIFEConvBlock(inPlanes, channels / 2, kernelSize: 3, stride: 2, padding: 1),
            RIFEConvBlock(channels / 2, channels, kernelSize: 3, stride: 2, padding: 1),
        ]
        _convblock.wrappedValue = (0 ..< 8).map { _ in RIFEResConv(channels) }
        _lastconv.wrappedValue = ConvTransposed2d(
            inputChannels: channels, outputChannels: 4 * 13, kernelSize: .init(4), stride: .init(2), padding: .init(1))
        super.init()
    }

    /// `x`: the caller's concat (image/warped-feature/mask/feat bundle, NHWC).
    /// `flow`: the running flow estimate so far, or `nil` for the first (coarsest) block.
    /// `scale`: this block's `scale_list` entry — `x`/`flow` are downsampled by `1/scale`
    /// before the conv stack, and the block's raw output is upsampled by `scale` after.
    /// Returns `(flow, mask, feat)`, all at the caller's original (pre-downsample) resolution.
    func callAsFunction(_ x0: MLXArray, flow flow0: MLXArray?, scale: Double) -> (MLXArray, MLXArray, MLXArray) {
        var x = rifeResize(x0, scaleFactor: 1.0 / scale, alignCorners: false)
        if var flow = flow0 {
            flow = rifeResize(flow, scaleFactor: 1.0 / scale, alignCorners: false) * Float(1.0 / scale)
            x = concatenated([x, flow], axis: -1)
        }
        var feat = conv0[0](x)
        feat = conv0[1](feat)
        for block in convblock { feat = block(feat) }
        var tmp = rifePixelShuffle(lastconv(feat), r: 2)
        tmp = rifeResize(tmp, scaleFactor: scale, alignCorners: false)
        let flow = tmp[.ellipsis, 0 ..< 4] * Float(scale)
        let mask = tmp[.ellipsis, 4 ..< 5]
        let outFeat = tmp[.ellipsis, 5...]
        return (flow, mask, outFeat)
    }
}

/// The full IFNet: 5 `RIFEBlock`s (channels `[192,128,96,64,32]`, `in_planes
/// [15,28,28,28,28]`) refining a flow estimate coarse-to-fine, plus the `Head` feature
/// encoder run once up front. Mirrors upstream `IFNet`.
nonisolated final class RIFENet: Module {
    @ModuleInfo var block0: RIFEBlock
    @ModuleInfo var block1: RIFEBlock
    @ModuleInfo var block2: RIFEBlock
    @ModuleInfo var block3: RIFEBlock
    @ModuleInfo var block4: RIFEBlock
    @ModuleInfo var encode: RIFEHead

    override init() {
        _block0.wrappedValue = RIFEBlock(inPlanes: 7 + 8, channels: 192)
        _block1.wrappedValue = RIFEBlock(inPlanes: 8 + 4 + 8 + 8, channels: 128)
        _block2.wrappedValue = RIFEBlock(inPlanes: 8 + 4 + 8 + 8, channels: 96)
        _block3.wrappedValue = RIFEBlock(inPlanes: 8 + 4 + 8 + 8, channels: 64)
        _block4.wrappedValue = RIFEBlock(inPlanes: 8 + 4 + 8 + 8, channels: 32)
        _encode.wrappedValue = RIFEHead()
        super.init()
    }

    /// `x`: `[N,H,W,6]` — `img0`/`img1` concatenated on the channel axis, both NHWC RGB
    /// in `[0,1]`. Returns the blended mid-frame `[N,H,W,3]` at `timestep`.
    func callAsFunction(_ x: MLXArray, timestep: Double, scaleList: [Double]) -> MLXArray {
        let channel = x.dim(-1) / 2
        let img0 = x[.ellipsis, 0 ..< channel]
        let img1 = x[.ellipsis, channel...]
        let n = img0.dim(0), h = img0.dim(1), w = img0.dim(2)
        let t = MLXArray.ones([n, h, w, 1]) * Float(timestep)

        let f0 = encode(img0[.ellipsis, 0 ..< 3])
        let f1 = encode(img1[.ellipsis, 0 ..< 3])

        let blocks = [block0, block1, block2, block3, block4]
        var flow: MLXArray?
        var mask: MLXArray?
        var feat: MLXArray?
        var warpedImg0 = img0
        var warpedImg1 = img1

        for i in 0 ..< 5 {
            if flow == nil {
                let (f, m, ft) = blocks[i](
                    concatenated([img0[.ellipsis, 0 ..< 3], img1[.ellipsis, 0 ..< 3], f0, f1, t], axis: -1),
                    flow: nil, scale: scaleList[i])
                flow = f; mask = m; feat = ft
            } else {
                let wf0 = rifeWarp(f0, flow![.ellipsis, 0 ..< 2])
                let wf1 = rifeWarp(f1, flow![.ellipsis, 2 ..< 4])
                let (fd, m0, ft) = blocks[i](
                    concatenated([warpedImg0[.ellipsis, 0 ..< 3], warpedImg1[.ellipsis, 0 ..< 3],
                                  wf0, wf1, t, mask!, feat!], axis: -1),
                    flow: flow, scale: scaleList[i])
                mask = m0
                flow = flow! + fd
                feat = ft
            }
            warpedImg0 = rifeWarp(img0, flow![.ellipsis, 0 ..< 2])
            warpedImg1 = rifeWarp(img1, flow![.ellipsis, 2 ..< 4])
        }

        let finalMask = sigmoid(mask!)
        return warpedImg0 * finalMask + warpedImg1 * (1 - finalMask)
    }
}

// MARK: - Warp (backward optical-flow warp via bilinear grid-sample)
//
// Ported 1:1 from upstream `warplayer.py` + `grid_sample.py` — there is no existing
// grid-sample/backward-warp of any kind elsewhere in this codebase.

/// `input`: `[N,H,W,C]`. `flow`: `[N,H,W,2]`, pixel-space `(x,y)` displacement.
/// Backward-warps `input` by `flow`: `align_corners=true`, border-clamped bilinear.
nonisolated func rifeWarp(_ input: MLXArray, _ flow: MLXArray) -> MLXArray {
    let n = flow.dim(0), h = flow.dim(1), w = flow.dim(2)
    let xs = MLXArray.linspace(Float(-1), Float(1), count: w).reshaped([1, 1, w, 1])
    let ys = MLXArray.linspace(Float(-1), Float(1), count: h).reshaped([1, h, 1, 1])
    let grid = concatenated(
        [broadcast(xs, to: [n, h, w, 1]), broadcast(ys, to: [n, h, w, 1])], axis: -1)
    let fx = flow[.ellipsis, 0 ..< 1] / Float((w - 1 == 0 ? 1 : w - 1)) * 2
    let fy = flow[.ellipsis, 1 ..< 2] / Float((h - 1 == 0 ? 1 : h - 1)) * 2
    let g = grid + concatenated([fx, fy], axis: -1)
    return rifeGridSampleBilinear(input, g)
}

/// `input`: `[N,H,W,C]`; `grid`: `[N,gH,gW,2]` in `[-1,1]`. `align_corners=true`,
/// `padding_mode="border"` (the only mode RIFE needs). Returns `[N,gH,gW,C]`.
nonisolated func rifeGridSampleBilinear(_ input: MLXArray, _ grid: MLXArray) -> MLXArray {
    let n = input.dim(0), h = input.dim(1), w = input.dim(2), c = input.dim(3)
    let gh = grid.dim(1), gw = grid.dim(2)

    let gx = grid[.ellipsis, 0]
    let gy = grid[.ellipsis, 1]
    let ix = (gx + 1) * 0.5 * Float(w - 1)
    let iy = (gy + 1) * 0.5 * Float(h - 1)

    let x0 = floor(ix), y0 = floor(iy)
    let x1 = x0 + 1, y1 = y0 + 1
    let wx1 = ix - x0, wx0 = 1.0 - wx1
    let wy1 = iy - y0, wy0 = 1.0 - wy1

    func clampIdx(_ a: MLXArray, _ maxV: Int) -> MLXArray {
        MLX.clip(a, min: MLXArray(Float(0)), max: MLXArray(Float(maxV))).asType(.int32)
    }
    let x0c = clampIdx(x0, w - 1), x1c = clampIdx(x1, w - 1)
    let y0c = clampIdx(y0, h - 1), y1c = clampIdx(y1, h - 1)
    let wInt = MLXArray(Int32(w))

    let inputFlat = input.reshaped([n, h * w, c])
    func gather(_ yc: MLXArray, _ xc: MLXArray) -> MLXArray {
        let idx = (yc * wInt + xc).reshaped([n, gh * gw, 1])
        let gathered = takeAlong(inputFlat, idx, axis: 1)
        return gathered.reshaped([n, gh, gw, c])
    }

    let v00 = gather(y0c, x0c), v01 = gather(y0c, x1c)
    let v10 = gather(y1c, x0c), v11 = gather(y1c, x1c)

    let w00 = (wy0 * wx0).expandedDimensions(axis: -1)
    let w01 = (wy0 * wx1).expandedDimensions(axis: -1)
    let w10 = (wy1 * wx0).expandedDimensions(axis: -1)
    let w11 = (wy1 * wx1).expandedDimensions(axis: -1)
    return v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
}

// MARK: - Resize + pixel-shuffle helpers

/// Bilinear resize (up- or down-sample), NHWC, via `MLXNN.Upsample` — already used this
/// way elsewhere in the codebase for both enlarging and shrinking
/// (`DeepSeekOCRSAM.getAbsPos`'s `scale = tgt/src` can be either side of 1), so this
/// reuses the house-blessed op rather than hand-rolling a second resize implementation.
nonisolated func rifeResize(_ x: MLXArray, scaleFactor: Double, alignCorners: Bool) -> MLXArray {
    guard scaleFactor != 1.0 else { return x }
    let s = Float(scaleFactor)
    let up = Upsample(scaleFactor: [s, s], mode: .linear(alignCorners: alignCorners))
    return up(x)
}

/// `[N,H,W,C*r²] → [N,H*r,W*r,C]`, channel split `(C,r,r)` — matches `torch.nn.PixelShuffle`
/// (and upstream `pixel_shuffle_nhwc`, which documents the same ordering).
nonisolated func rifePixelShuffle(_ x: MLXArray, r: Int) -> MLXArray {
    let n = x.dim(0), h = x.dim(1), w = x.dim(2), cIn = x.dim(3)
    let c = cIn / (r * r)
    let reshaped = x.reshaped([n, h, w, c, r, r])
    let transposed = reshaped.transposed(0, 1, 4, 2, 5, 3)
    return transposed.reshaped([n, h * r, w * r, c])
}
