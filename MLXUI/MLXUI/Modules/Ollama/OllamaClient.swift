import Foundation

/// One model as reported by a local Ollama server's `/api/tags`. Ollama owns these
/// weights entirely — pulled, stored, and served by the user's own `ollama` install, not
/// by `InstallManager` — so this is deliberately not a `ModelEntry`: forcing a
/// non-downloadable, dynamically-discovered model into `browser.json`'s world is exactly
/// the trap `ModelSlot`'s header warns against (a fake `ramGB`/`hfModelId` that compiles
/// but is meaningless). Ollama gets its own small, parallel surface instead.
nonisolated struct OllamaModelInfo: Sendable, Equatable, Identifiable, Codable {
    var id: String { name }
    let name: String              // e.g. "llama3.2:latest" — the exact tag `/api/chat` expects back
    let sizeBytes: Int64
    let family: String?
    let parameterSize: String?
    let quantizationLevel: String?
    /// Ollama's own `/api/show` `capabilities` array (`"completion"`, `"embedding"`,
    /// `"vision"`, `"tools"`, …) — fetched once per model in `listModels()`. Empty when
    /// the server didn't report any (older Ollama builds, or the lookup failed): treated
    /// as "unknown, not as "no capabilities" — see `supportsChat`.
    var capabilities: [String] = []

    var sizeGB: Double { Double(sizeBytes) / 1_073_741_824 }

    /// Whether `/api/chat` is worth trying at all. `nomic-embed-text` and similar
    /// embedding-only models report `capabilities: ["embedding"]` with no `"completion"` —
    /// sending them a chat request always fails, so the Chat affordance is hidden rather
    /// than offering a button that's guaranteed to error. Unknown capabilities (empty
    /// array — the server didn't say, rather than said "nothing") default to `true`: an
    /// older Ollama that predates `/api/show`'s `capabilities` field must not have every
    /// model wrongly blocked.
    var supportsChat: Bool {
        capabilities.isEmpty || capabilities.contains("completion")
    }
}

nonisolated enum OllamaClientError: Error, Sendable {
    case notReachable
    case badResponse(status: Int)
    case decodeFailed
}

/// A thin `URLSession`-only client for a local Ollama server — no new package, matching
/// `ProviderExecutor`'s own standing rule. Ollama's default install always listens on
/// `127.0.0.1:11434` (owner's call for v1: hardcoded, no Settings field — the overwhelming
/// majority of installs never change it; a configurable host/port is a follow-up if asked
/// for).
nonisolated enum OllamaClient {
    static let baseURL = URL(string: "http://127.0.0.1:11434")!

    /// A short-timeout probe — absence of a running Ollama server is the common case (most
    /// Macs don't have it installed), never an error to surface, just an empty/disabled UI.
    static func isReachable() async -> Bool {
        (try? await listModels()) != nil
    }

    /// `GET /api/tags` — every model `ollama pull`ed onto this machine.
    static func listModels() async throws -> [OllamaModelInfo] {
        var request = URLRequest(url: baseURL.appendingPathComponent("api/tags"))
        request.timeoutInterval = 3
        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw OllamaClientError.notReachable
        }
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw OllamaClientError.badResponse(status: (response as? HTTPURLResponse)?.statusCode ?? 0)
        }
        guard let root = try? JSONDecoder().decode(TagsResponse.self, from: data) else {
            throw OllamaClientError.decodeFailed
        }
        // Enrich with each model's real capabilities (`/api/show`) so an embedding-only
        // model like `nomic-embed-text` never offers a Chat button that's guaranteed to
        // fail — one request per model, run concurrently; a lookup failure just leaves
        // that model's `capabilities` empty (read as "unknown," never as "blocked" — see
        // `OllamaModelInfo.supportsChat`), so a slow/older server degrades gracefully
        // rather than hiding every model.
        return await withTaskGroup(of: OllamaModelInfo.self) { group in
            for entry in root.models {
                group.addTask {
                    let capabilities = await showCapabilities(model: entry.name)
                    return OllamaModelInfo(name: entry.name, sizeBytes: entry.size,
                                            family: entry.details?.family,
                                            parameterSize: entry.details?.parameter_size,
                                            quantizationLevel: entry.details?.quantization_level,
                                            capabilities: capabilities)
                }
            }
            var results: [OllamaModelInfo] = []
            for await info in group { results.append(info) }
            return results
        }
    }

    /// `POST /api/show` — real per-model capabilities (`"completion"`, `"embedding"`,
    /// `"vision"`, `"tools"`, …). Best-effort: any failure (older Ollama without this
    /// field, a timeout, a decode miss) yields `[]`, never throws — this is enrichment,
    /// not something that should take the whole model list down with it.
    private static func showCapabilities(model: String) async -> [String] {
        var request = URLRequest(url: baseURL.appendingPathComponent("api/show"))
        request.httpMethod = "POST"
        request.timeoutInterval = 3
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        guard let body = try? JSONEncoder().encode(ShowRequest(name: model)) else { return [] }
        request.httpBody = body
        guard let (data, response) = try? await URLSession.shared.data(for: request),
              let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode),
              let show = try? JSONDecoder().decode(ShowResponse.self, from: data) else {
            return []
        }
        return show.capabilities ?? []
    }

    /// `POST /api/chat`, streamed (`stream: true`) — one JSON object per line (NDJSON),
    /// each carrying the next content delta until `"done": true`. Cancel the enclosing
    /// `Task` to stop generation early; `URLSession.bytes` tears the connection down on
    /// cancellation.
    static func chatStream(model: String, messages: [OllamaChatMessage]) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var request = URLRequest(url: baseURL.appendingPathComponent("api/chat"))
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "content-type")
                    let body = ChatRequest(model: model, messages: messages, stream: true)
                    request.httpBody = try JSONEncoder().encode(body)

                    let (bytes, response) = try await URLSession.shared.bytes(for: request)
                    guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                        throw OllamaClientError.badResponse(status: (response as? HTTPURLResponse)?.statusCode ?? 0)
                    }
                    for try await line in bytes.lines {
                        try Task.checkCancellation()
                        guard let data = line.data(using: .utf8),
                              let chunk = try? JSONDecoder().decode(ChatStreamChunk.self, from: data) else { continue }
                        if !chunk.message.content.isEmpty {
                            continuation.yield(chunk.message.content)
                        }
                        if chunk.done { break }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Wire types

    private struct TagsResponse: Decodable {
        let models: [TagEntry]
    }
    private struct TagEntry: Decodable {
        let name: String
        let size: Int64
        let details: Details?
    }
    private struct Details: Decodable {
        let family: String?
        let parameter_size: String?
        let quantization_level: String?
    }
    private struct ChatRequest: Encodable {
        let model: String
        let messages: [OllamaChatMessage]
        let stream: Bool
    }
    private struct ChatStreamChunk: Decodable {
        let message: OllamaChatMessage
        let done: Bool
    }
    private struct ShowRequest: Encodable {
        let name: String
    }
    private struct ShowResponse: Decodable {
        let capabilities: [String]?
    }
}

/// Ollama's `/api/chat` message shape — `role` is `"user"` / `"assistant"` / `"system"`.
nonisolated struct OllamaChatMessage: Codable, Sendable, Equatable {
    let role: String
    let content: String
}

/// The `AppState`-visible half of the Ollama surface: what `OllamaListView`/
/// `OllamaRunChatView` read and drive. Kept as its own small `@Observable` object rather
/// than four more stored properties on `AppState` itself — `AppState` only needs to hold
/// one `var ollama = OllamaCatalogState()`, so this feature's footprint on that
/// already-large hub is a single line.
@Observable
final class OllamaCatalogState {
    var models: [OllamaModelInfo] = []
    var reachable = false
    var isRefreshing = false
    /// Non-nil presents `OllamaRunChatView` for this model.
    var runModel: OllamaModelInfo?

    /// Probe the local server and refresh `models`/`reachable`. Safe to call repeatedly
    /// (sidebar `.task`, the manual Refresh button) — absence of a server is not an error,
    /// just an empty result, matching `OllamaClient.isReachable`'s own reasoning.
    func refresh() async {
        isRefreshing = true
        defer { isRefreshing = false }
        guard let found = try? await OllamaClient.listModels() else {
            reachable = false
            models = []
            return
        }
        reachable = true
        models = found.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }
}
