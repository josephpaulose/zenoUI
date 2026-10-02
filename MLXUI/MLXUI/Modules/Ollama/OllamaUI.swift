import SwiftUI

/// A minimal, self-contained chat session against a local Ollama model. Deliberately not
/// routed through `ModelRunner` — that class is MLX-specific (real weight loading,
/// tool-call approval, conversation persistence) and Ollama's own server already does the
/// generation; bolting an unrelated backend onto it would be the same "stripped-down copy
/// that happens to work" trap `ProviderModelRef`'s header warns about. No tool calls, no
/// saved conversation history in this first cut — just prompt in, streamed text out.
@Observable
final class OllamaChatViewModel {
    let model: OllamaModelInfo
    var messages: [ChatMessage] = []
    var isRunning = false
    var errorMessage: String?

    private var streamTask: Task<Void, Never>?

    init(model: OllamaModelInfo) {
        self.model = model
    }

    func send(_ text: String) {
        guard !isRunning else { return }
        errorMessage = nil
        messages.append(ChatMessage(role: .user, text: text))
        let assistantID = UUID()
        messages.append(ChatMessage(id: assistantID, role: .assistant, text: ""))
        isRunning = true

        let wire = messages.dropLast().map {
            OllamaChatMessage(role: $0.role == .user ? "user" : "assistant", content: $0.text)
        }
        streamTask = Task { @MainActor [model] in
            do {
                for try await delta in OllamaClient.chatStream(model: model.name, messages: wire) {
                    if let index = messages.firstIndex(where: { $0.id == assistantID }) {
                        messages[index].text += delta
                    }
                }
            } catch is CancellationError {
                // Stop button — the partial assistant text stays, matching RunChatView's stop.
            } catch {
                errorMessage = "Ollama request failed: \(error.localizedDescription)"
            }
            isRunning = false
        }
    }

    func stop() {
        streamTask?.cancel()
        isRunning = false
    }
}

/// The "Ollama" sidebar destination: whatever `ollama list` reports on this machine right
/// now, discovered live (never a static catalog entry — see `OllamaModelInfo`'s header).
struct OllamaListView: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        Group {
            if appState.ollama.isRefreshing && appState.ollama.models.isEmpty {
                ProgressView("Looking for a local Ollama server…")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if !appState.ollama.reachable {
                unreachableState
            } else if appState.ollama.models.isEmpty {
                emptyState
            } else {
                modelList
            }
        }
        .navigationTitle("Ollama")
        .toolbar {
            ToolbarItem {
                Button {
                    Task { await appState.ollama.refresh() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Refresh")
            }
        }
        .task { await appState.ollama.refresh() }
    }

    private var unreachableState: some View {
        VStack(spacing: 10) {
            Image(systemName: "server.rack")
                .font(.system(size: 32))
                .foregroundStyle(.secondary)
            Text("No Ollama server found")
                .font(.headline)
            Text("Install Ollama and run `ollama pull <model>`, then Refresh. MLXUI looks for it at 127.0.0.1:11434.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)
            Link("Get Ollama", destination: URL(string: "https://ollama.com/download")!)
                .padding(.top, 4)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: "tray")
                .font(.system(size: 32))
                .foregroundStyle(.secondary)
            Text("Ollama is running, but no models are pulled yet")
                .font(.headline)
            Text("Run `ollama pull <model>` in Terminal, then Refresh.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var modelList: some View {
        List(appState.ollama.models) { info in
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(info.name)
                        .font(.callout.weight(.medium))
                    Text(subtitle(for: info))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if info.supportsChat {
                    Button("Chat") {
                        appState.ollama.runModel = info
                    }
                    .buttonStyle(.bordered)
                } else {
                    // `/api/show` named this an embedding-only model (e.g.
                    // nomic-embed-text) — `/api/chat` always fails for it, so there's no
                    // button to offer, only the reason there isn't one.
                    Text("Embedding model")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background(.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
                }
            }
            .padding(.vertical, 4)
        }
    }

    private func subtitle(for info: OllamaModelInfo) -> String {
        var parts: [String] = []
        if let family = info.family { parts.append(family) }
        if let params = info.parameterSize { parts.append(params) }
        if let quant = info.quantizationLevel { parts.append(quant) }
        parts.append(String(format: "%.1f GB", info.sizeGB))
        return parts.joined(separator: " · ")
    }
}

/// Chat surface for one Ollama model, matching `RunChatView`'s visual language (bubbles,
/// input bar) without any of its MLX/tool-approval/conversation-persistence plumbing.
struct OllamaRunChatView: View {
    let model: OllamaModelInfo

    @Environment(\.dismiss) private var dismiss
    @State private var viewModel: OllamaChatViewModel
    @State private var prompt = ""

    init(model: OllamaModelInfo) {
        self.model = model
        _viewModel = State(initialValue: OllamaChatViewModel(model: model))
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            transcript
            if let error = viewModel.errorMessage {
                errorBar(error)
            }
            Divider()
            inputBar
        }
        .frame(width: 640, height: 560)
        .onDisappear { viewModel.stop() }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "server.rack")
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(model.name)
                    .font(.headline)
                Text("Running locally · Ollama")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Done") { dismiss() }
                .keyboardShortcut(.cancelAction)
        }
        .padding(12)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    if viewModel.messages.isEmpty {
                        emptyState
                    }
                    ForEach(viewModel.messages) { message in
                        messageRow(message).id(message.id)
                    }
                }
                .padding(16)
            }
            .onChange(of: viewModel.messages.count) { _, _ in scrollToEnd(proxy) }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 28))
                .foregroundStyle(.secondary)
            Text("Ask \(model.name) something to get started.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 40)
    }

    private func messageRow(_ message: ChatMessage) -> some View {
        let isUser = message.role == .user
        return HStack {
            if isUser { Spacer(minLength: 40) }
            Text(message.text)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    isUser ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.12),
                    in: RoundedRectangle(cornerRadius: 10)
                )
            if !isUser { Spacer(minLength: 40) }
        }
    }

    private func errorBar(_ message: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
            Text(message)
                .font(.caption)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .foregroundStyle(.red)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.red.opacity(0.08))
    }

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Message \(model.name)…", text: $prompt, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...4)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
                .onSubmit(send)
                .disabled(viewModel.isRunning)

            if viewModel.isRunning {
                Button {
                    viewModel.stop()
                } label: {
                    Image(systemName: "stop.fill")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.bordered)
                .help("Stop generating")
            } else {
                Button {
                    send()
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 22))
                }
                .buttonStyle(.plain)
                .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(12)
    }

    private func send() {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !viewModel.isRunning else { return }
        prompt = ""
        viewModel.send(text)
    }

    private func scrollToEnd(_ proxy: ScrollViewProxy) {
        guard let last = viewModel.messages.last?.id else { return }
        withAnimation(.easeOut(duration: 0.15)) {
            proxy.scrollTo(last, anchor: .bottom)
        }
    }
}

/// A `ModelCard`-shaped card for one Ollama model, so it can sit in the same
/// `BrowseView` grid as the catalog's own cards under Browse → Infrastructure — the
/// natural place for the user to expect *every* runnable model to show up, `browser.json`
/// or not. Deliberately its own view rather than `ModelCard(model: ModelEntry)` itself:
/// `ModelCard` is built entirely around `ModelEntry`'s install/RAM-fit machinery
/// (`appState.installManager.modelStates`, `exceedsRAM`, `SourceBadge(source: ModelSource)`),
/// none of which applies here — Ollama's own server already has the weights, there is
/// nothing to install and no RAM gate MLXUI enforces. Matches `ModelCard`'s fixed height
/// and background so the two interleave in one grid without a visible seam.
struct OllamaModelCard: View {
    let model: OllamaModelInfo

    private static let cardHeight: CGFloat = 244

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Spacer()
                Text("OLLAMA")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
            .frame(height: 18)
            Image(systemName: "server.rack")
                .font(.system(size: 28))
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 6)
            VStack(spacing: 1) {
                Text(model.name)
                    .font(.headline)
                    .lineLimit(1)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Text(model.parameterSize ?? "Local model")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 32)
            Text("Served by your local Ollama install — leaves this Mac only if you use it.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(2, reservesSpace: true)
                .multilineTextAlignment(.leading)
                .frame(maxWidth: .infinity, alignment: .topLeading)
                .padding(.vertical, 3)
            VStack(alignment: .leading, spacing: 3) {
                specRow("memorychip", String(format: "%.1f GB on disk", model.sizeGB))
                if let family = model.family {
                    specRow("cpu", family)
                }
                if let quant = model.quantizationLevel {
                    specRow("gauge.with.dots.needle.33percent", quant)
                }
            }
            Spacer(minLength: 0)
            if model.supportsChat {
                Button {
                    appState.ollama.runModel = model
                } label: {
                    Label("Chat", systemImage: "bubble.left.and.bubble.right")
                        .font(.caption).fontWeight(.medium).frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .frame(height: 30)
            } else {
                // `/api/show` named this an embedding-only model — `/api/chat` always
                // fails for it, so there's no button, only the reason there isn't one.
                Label("Embedding model", systemImage: "square.grid.3x3")
                    .font(.caption).foregroundStyle(.secondary).frame(maxWidth: .infinity)
                    .padding(.vertical, 5)
                    .background(.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
                    .frame(height: 30)
            }
        }
        .frame(height: Self.cardHeight)
        .padding(10)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
    }

    @Environment(AppState.self) private var appState

    private func specRow(_ icon: String, _ label: String) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption2)
                .frame(width: 14)
                .foregroundStyle(.primary)
            Text(label)
                .font(.caption2)
        }
    }
}
