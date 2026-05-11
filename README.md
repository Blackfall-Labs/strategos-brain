# Strategos Brain

Inference runtime and autonomous agent layer for the Strategos mainframe.

## Model Requirement

**Phi-4 14B Q4_0 is the only supported model.**

```
docker model pull ai/phi4:14B-Q4_0
```

| Property | Value |
|----------|-------|
| Model | `docker.io/ai/phi4:14B-Q4_0` |
| Parameters | 14.66B |
| Quantization | Q4_0 |
| Size | 7.80 GiB |
| Architecture | phi3 |
| Docker Model ID | `7e95cd0038a4` |

The brain will refuse to start (or degrade gracefully) if this model is not loaded. Do not substitute `phi4:latest` or any other variant — the quantization and parameter count affect response quality and tool-calling behaviour in ways that have not been tested with other builds.

Docker Desktop must have GPU inference enabled and the model runner running:

```
docker model status   # should show: Docker Model Runner is running
docker model ps       # phi4:14B-Q4_0 should be listed
```

## Architecture

```
strategos-brain/
  crates/
    ops/            — integer math primitives (future CPU-native inference)
    runtime/        — model weights, tokenizer, forward passes (dormant)
    frame-catalog/  — HNSW vector store + RAG retrieval
    brain/          — autonomous agent layer (in progress)
  knowledge/        — RAG corpus (.md / .txt files indexed at startup)
  memory/           — persistent facts (gitignored, written by remember tool)
```

### Current Interface

The `strategos` binary is an interactive REPL backed by Phi-4 via Docker's
OpenAI-compatible API (`localhost:12434`). It has:

- **Prompt-based tool calling** — `<tool_call>` tags parsed from response text
- **Active RAG** — `search_knowledge` tool queries the HNSW index on demand
- **File access** — `read_file` tool reads local files (capped at 8KB)
- **Persistent memory** — `remember` tool appends facts to `memory/facts.json`,
  loaded into the system prompt on next startup
- **Streaming output** — SSE read via `reqwest::blocking` as `BufReader`

### Brain Crate (planned)

The `brain` crate will be an always-running daemon that bridges Phi-4 to the
Strategos capability system. Strategos exposes 55+ capabilities (vcs, pm,
context, documents, fc, hephaestus, plan, todo) via MCP. Brain will:

1. Query available capabilities at startup
2. Generate tool definitions from capability metadata dynamically
3. Inject them as available tools in Phi-4's system prompt
4. Dispatch `<tool_call>` responses into the capability layer
5. Handle event sources: git hooks, file watchers, system notifications

Any capability registered in Strategos automatically becomes available to
Phi-4. No separate tool definitions to maintain.

## Running

```powershell
# Ensure model is loaded
docker model ps

# Build
cargo build --release --bin strategos

# Run REPL
.\target\release\strategos.exe
```

## Performance

| Condition | Latency |
|-----------|---------|
| Cold (model evicted) | 20–25s first response |
| Warm, no tool call | ~6s |
| Warm, one tool call | ~12–15s |

Docker's model runner evicts the model after ~1 minute of inactivity.
Keep queries coming to stay warm.

## CPU-Native Path

`crates/ops` and `crates/runtime` contain an integer inference pipeline
(i16 weights, AVX2 SIMD, Rayon parallelism) that reached 9.0 tok/s on
Qwen3 1.7B. This work is paused while the system runs on Docker/Phi-4.
The target is 20+ tok/s before switching back to CPU-native inference.
