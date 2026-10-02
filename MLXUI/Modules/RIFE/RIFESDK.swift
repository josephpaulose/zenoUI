import Foundation
import SwiftUI

/// `ModelSDK` for RIFE 4.25 frame interpolation.
/// Claims models with `runnerKind == .interpolate` and "rife" in the id/hfModelId.
nonisolated struct RIFESDK: ModelSDK {
    let id = "rife"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .interpolate else { return .no }
        let hay = (model.id + " " + model.hfModelId).lowercased()
        guard hay.contains("rife") else { return .no }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        RIFEStage(modelID: model.id)
    }
}

/// `ModelUI` for RIFE — routes to `RIFERunView`.
struct RIFEUI: ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .interpolate else { return .no }
        let hay = (model.id + " " + model.hfModelId).lowercased()
        guard hay.contains("rife") else { return .no }
        return .exact
    }

    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView {
        AnyView(RIFERunView(modelDisplayName: model.displayName, modelID: model.id))
    }
}
