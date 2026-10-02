import CoreGraphics
import Foundation

/// `image → image` stage for RIFE 4.25 frame interpolation. `PipelineStage`'s contract
/// (`Core/Media.swift`) has no "two images in" shape, and — same as `SeedVR2Stage` — the
/// Run UI never actually calls `run(_:progress:)`; it calls `RIFEEngine.interpolate`
/// directly with the two picked frames. This conformance exists only so `RIFESDK.makeStage`
/// has something to return; `run` throws if it's ever invoked through the generic path.
nonisolated struct RIFEStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .image }
    var produces: MediaKind { .image }

    let modelID: String

    init(modelID: String) {
        self.id = "rife.\(modelID)"
        self.name = "RIFE 4.25 (\(modelID))"
        self.modelID = modelID
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        throw StageError.unsupportedModel(
            id: modelID, kind: .interpolate)
    }
}
