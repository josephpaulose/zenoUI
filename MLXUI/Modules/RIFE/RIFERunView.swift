import AppKit
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Run model

@MainActor
@Observable
final class RIFERunModel {
    var frameA: CGImage?
    var frameB: CGImage?
    var resultImage: CGImage?
    var timestep: Double = 0.5
    var isRunning = false
    var progress = 0.0
    var progressLabel = ""
    var errorText: String?

    func setFrameA(_ cg: CGImage) {
        frameA = cg
        resultImage = nil
        errorText = nil
    }

    func setFrameB(_ cg: CGImage) {
        frameB = cg
        resultImage = nil
        errorText = nil
    }

    func interpolate(modelID: String) {
        guard let a = frameA, let b = frameB else { return }
        guard a.width == b.width, a.height == b.height else {
            errorText = "Frame A and Frame B must be the same size."
            return
        }
        isRunning = true
        progress = 0.0
        progressLabel = "Preparing frames…"
        errorText = nil
        resultImage = nil

        let t = timestep

        Task {
            do {
                let result = try await RIFEEngine.interpolate(
                    imageA: a, imageB: b, timestep: t, modelID: modelID,
                    progress: { [weak self] p in
                        Task { @MainActor [weak self] in
                            guard let self else { return }
                            self.progress = p
                            self.progressLabel = Self.label(for: p)
                        }
                    }
                )
                resultImage = result
                progress = 1.0
                progressLabel = "Done"
                isRunning = false
            } catch {
                errorText = error.localizedDescription
                isRunning = false
            }
        }
    }

    private static func label(for p: Double) -> String {
        switch p {
        case ..<0.10: return "Preparing frames…"
        case ..<0.30: return "Loading model…"
        case ..<0.90: return "Estimating flow…"
        default:      return "Done"
        }
    }
}

// MARK: - Run view

/// Bespoke Run surface for RIFE 4.25 frame interpolation. Workflow: load Frame A + Frame
/// B → pick a timestep between them → Interpolate → preview the generated in-between
/// frame → Save PNG. Scoped to the two-still-images case (mirrors the upstream CLI's
/// `--img0/--img1 -t` mode) — not multi-frame video.
struct RIFERunView: View {
    @Environment(\.dismiss) private var dismiss

    let modelDisplayName: String
    let modelID: String

    @State private var model = RIFERunModel()
    @State private var showImporterA = false
    @State private var showImporterB = false
    @State private var isTargetedA = false
    @State private var isTargetedB = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            framesRow
            timestepRow
            if let result = model.resultImage { resultRow(result) }
            if model.isRunning { progressRow }
            if let error = model.errorText { RunErrorBar(message: error) }
            Spacer(minLength: 0)
            footer
        }
        .padding(18)
        .frame(width: 700, height: 820)
        .fileImporter(
            isPresented: $showImporterA, allowedContentTypes: [.image], allowsMultipleSelection: false
        ) { result in
            if case let .success(urls) = result, let url = urls.first { loadImage(url, isA: true) }
        }
        .fileImporter(
            isPresented: $showImporterB, allowedContentTypes: [.image], allowsMultipleSelection: false
        ) { result in
            if case let .success(urls) = result, let url = urls.first { loadImage(url, isA: false) }
        }
    }

    // MARK: - Sections

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "rectangle.stack")
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(modelDisplayName).font(.headline)
                Text("Two frames → the frame between them").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var framesRow: some View {
        HStack(spacing: 12) {
            frameCanvas(
                title: "Frame A", image: model.frameA, isTargeted: $isTargetedA,
                onChoose: { showImporterA = true }, onDrop: { loadDropped($0, isA: true) })
            frameCanvas(
                title: "Frame B", image: model.frameB, isTargeted: $isTargetedB,
                onChoose: { showImporterB = true }, onDrop: { loadDropped($0, isA: false) })
        }
    }

    private func frameCanvas(
        title: String, image: CGImage?, isTargeted: Binding<Bool>,
        onChoose: @escaping () -> Void, onDrop: @escaping ([NSItemProvider]) -> Bool
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.subheadline.bold())
            ZStack {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(
                        isTargeted.wrappedValue ? Color.accentColor : .secondary.opacity(0.4),
                        style: StrokeStyle(lineWidth: 1.5, dash: [6])
                    )
                    .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))

                if let image {
                    Image(nsImage: NSImage(cgImage: image, size: .zero))
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "square.and.arrow.down").font(.title2)
                        Text("Drop an image, or choose one").foregroundStyle(.secondary)
                    }
                }
            }
            .frame(height: 260)
            .onDrop(of: [.image], isTargeted: isTargeted, perform: onDrop)
            Button("Choose…", action: onChoose).disabled(model.isRunning)
        }
    }

    private var timestepRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Timestep").font(.subheadline).foregroundStyle(.secondary)
                Spacer()
                Text("\(Int(model.timestep * 100))% from A to B")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Slider(value: $model.timestep, in: 0...1).disabled(model.isRunning)
        }
    }

    private func resultRow(_ result: CGImage) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 12) {
                Label("Interpolated frame", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green).font(.subheadline.bold())
                Text("\(result.width) × \(result.height)")
                    .font(.caption).foregroundStyle(.secondary)
                Button {
                    savePNG(result)
                } label: {
                    Label("Save PNG…", systemImage: "square.and.arrow.down")
                }
            }
            Image(nsImage: NSImage(cgImage: result, size: .zero))
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 220)
                .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))
        }
    }

    private var progressRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                ProgressView(value: model.progress).progressViewStyle(.linear)
                Text("\(Int(model.progress * 100))%")
                    .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
            }
            Text(model.progressLabel).font(.caption).foregroundStyle(.secondary)
        }
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var footer: some View {
        HStack {
            Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            Spacer()
            Button(model.isRunning ? "Interpolating…" : "Interpolate") {
                model.interpolate(modelID: modelID)
            }
            .keyboardShortcut(.defaultAction)
            .disabled(model.frameA == nil || model.frameB == nil || model.isRunning)
        }
    }

    // MARK: - Helpers

    private func loadImage(_ url: URL, isA: Bool) {
        Task {
            let cg = await Task.detached(priority: .userInitiated) {
                ImageLoader.decodedCGImage(fromSecurityScoped: url)
            }.value
            if let cg {
                if isA { model.setFrameA(cg) } else { model.setFrameB(cg) }
            } else {
                model.errorText = "Could not read image."
            }
        }
    }

    private func loadDropped(_ providers: [NSItemProvider], isA: Bool) -> Bool {
        guard let provider = providers.first else { return false }
        provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
            guard let data, let cg = ImageLoader.decodedCGImage(from: data) else { return }
            Task { @MainActor in
                if isA { model.setFrameA(cg) } else { model.setFrameB(cg) }
            }
        }
        return true
    }

    private func savePNG(_ image: CGImage) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "rife-interpolated.png"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let rep = NSBitmapImageRep(cgImage: image)
        guard let data = rep.representation(using: .png, properties: [:]) else {
            model.errorText = "Could not encode image as PNG."
            return
        }
        do {
            try data.write(to: url)
        } catch {
            model.errorText = "Save failed: \(error.localizedDescription)"
        }
    }
}
