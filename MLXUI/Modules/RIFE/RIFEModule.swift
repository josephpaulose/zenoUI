import Foundation

/// Bundles the RIFE 4.25 frame-interpolation SDK + UI and registers them.
/// Swift/MLX port of `xocialize/rife-mlx` (itself an MLX port of Practical-RIFE 4.25,
/// MIT) — a 5-block coarse-to-fine optical-flow network (`IFNet`) that blends two
/// frames at a chosen in-between timestep via backward-warp + a learned blend mask.
enum RIFEModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "rife",
        displayName: "RIFE 4.25 (Frame Interpolation, MLX fp32)",
        modalities: ["image"],
        modelTypes: [.interpolate],
        backingPackage: "mlx-community/RIFE-4.25 (xocialize/rife-mlx Swift port)",
        packageLicense: "MIT",
        maintainers: ["hzwer (Practical-RIFE)", "xocialize (rife-mlx MLX port)"],
        notes: """
        Two-frame interpolation via a Swift/MLX port of RIFE 4.25's IFNet: 5 \
        coarse-to-fine flow-estimation blocks (channels 192/128/96/64/32), a shared \
        feature-encoder head, and backward-warp + learned blend-mask compositing. \
        Accepts two same-size images and a timestep in [0,1]; returns the blended \
        frame between them. Scoped to the still-image pair case — not multi-frame \
        video interpolation.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: RIFESDK(), ui: RIFEUI(), descriptor: descriptor)
    }
}
