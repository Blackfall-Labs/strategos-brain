use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use reqwest::blocking::Client;
use serde_json::{json, Value};
use frame_catalog::{RetrievalSystem, RetrievalConfig};

const API_URL: &str = "http://localhost:12434/v1/chat/completions";
const MODEL: &str = "docker.io/ai/smollm3";
const KNOWLEDGE_DIR: &str = "knowledge";
const MEMORY_FILE: &str = "memory/facts.json";
const RAG_TOP_K: usize = 5;
const MAX_TOOL_ROUNDS: usize = 8;
// Buffer this many chars before deciding stream vs tool-call.
// <tool_call> is 11 chars so 64 is a comfortable detection window.
const DECIDE_WINDOW: usize = 64;

const SYSTEM_PROMPT_BASE: &str = r#"You are Strategos, a decision-making intelligence. Answer in 1–3 sentences. Never pad, speculate, or ask follow-up questions.

When you need to use a tool, your response MUST be ONLY the tool call in exactly this format — no text before or after:
<tool_call>[{"name": "tool_name", "arguments": {"param": "value"}}]</tool_call>

Available tools:
- search_knowledge — arguments: {"query": "string"} — search local knowledge base
- read_file        — arguments: {"path": "string"}  — read a local file
- remember         — arguments: {"fact": "string"}  — save a fact for future sessions

Rules:
- Use search_knowledge when asked about facts, documentation, or anything you might not know.
- Output ONLY the <tool_call> block — stop immediately after </tool_call>.
- After receiving a <tool_result>, give your final answer directly.
- If search returns nothing and you don't know the answer, say so. Never speculate."#;

fn build_system_prompt(memories: &[String]) -> String {
    if memories.is_empty() {
        return SYSTEM_PROMPT_BASE.to_string();
    }
    let facts = memories.iter()
        .map(|f| format!("- {}", f))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{}\n\n[What you remember from previous sessions:]\n{}", SYSTEM_PROMPT_BASE, facts)
}

fn load_memories() -> Vec<String> {
    let content = std::fs::read_to_string(MEMORY_FILE).unwrap_or_default();
    serde_json::from_str(&content).unwrap_or_default()
}

fn save_memory(fact: &str) -> String {
    let path = std::path::Path::new(MEMORY_FILE);
    if let Some(p) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(p) {
            return format!("Error creating memory dir: {}", e);
        }
    }
    let mut memories = load_memories();
    memories.push(fact.to_string());
    match serde_json::to_string_pretty(&memories).and_then(|j| {
        std::fs::write(path, j).map_err(|e| serde_json::Error::io(e))
    }) {
        Ok(_) => format!("Remembered: {}", fact),
        Err(e) => format!("Error saving memory: {}", e),
    }
}

// ── Result of one streaming API call ─────────────────────────────────────────

enum CallResult {
    /// Final answer — already streamed to stdout.
    Response(String),
    /// Tool call — content truncated at </tool_call>.
    ToolCall { name: String, args: Value, head: String },
    Error(String),
}

// ── Core streaming call ───────────────────────────────────────────────────────

fn call(
    client: &Client,
    messages: &[Value],
    spinning: Arc<AtomicBool>,
    spin_handle: std::thread::JoinHandle<()>,
) -> CallResult {
    let body = json!({
        "model": MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "stream": true,
    });

    let response = match client.post(API_URL).json(&body).send() {
        Ok(r) => r,
        Err(e) => {
            spinning.store(false, Ordering::Relaxed);
            spin_handle.join().ok();
            return CallResult::Error(e.to_string());
        }
    };

    let reader = std::io::BufReader::new(response);
    let mut accumulated = String::new();
    let mut stdout = io::stdout();

    enum Mode { Deciding, Streaming, ToolCall }
    let mut mode = Mode::Deciding;
    let mut decide_buf = String::new();
    let mut spin_handle = Some(spin_handle);
    // SmolLM3 emits <think>...</think> before acting — skip it silently.
    let mut in_think = false;

    for line in reader.lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if !line.starts_with("data: ") { continue; }
        let data = line["data: ".len()..].trim();
        if data == "[DONE]" { break; }

        let chunk: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let delta = chunk["choices"][0]["delta"]["content"].as_str().unwrap_or("");
        accumulated.push_str(delta);

        if delta.contains("<think>") { in_think = true; }
        if in_think {
            if delta.contains("</think>") {
                in_think = false;
                decide_buf.clear(); // restart decision window after think block
            }
            continue;
        }

        match mode {
            Mode::Deciding => {
                decide_buf.push_str(delta);
                let is_tool = decide_buf.contains("<tool_call>")
                    || decide_buf.trim_start().starts_with("{\"name\":")
                    || decide_buf.trim_start().starts_with("[{\"name\":")
                    || looks_like_naked_call(&decide_buf);
                if is_tool {
                    mode = Mode::ToolCall;
                } else if decide_buf.len() >= DECIDE_WINDOW {
                    // Stop and JOIN the spinner before touching stdout.
                    spinning.store(false, Ordering::Relaxed);
                    if let Some(h) = spin_handle.take() { h.join().ok(); }
                    print!("strategos> {}", decide_buf);
                    stdout.flush().ok();
                    decide_buf.clear();
                    mode = Mode::Streaming;
                }
            }
            Mode::Streaming => {
                print!("{}", delta);
                stdout.flush().ok();
            }
            Mode::ToolCall => { /* accumulate silently */ }
        }
    }

    // Stop spinner for all remaining cases (tool call, short response, error).
    spinning.store(false, Ordering::Relaxed);
    if let Some(h) = spin_handle.take() { h.join().ok(); }

    match mode {
        Mode::Deciding if !decide_buf.contains("<tool_call>") => {
            print!("strategos> {}", decide_buf);
            stdout.flush().ok();
            println!("\n");
            CallResult::Response(accumulated)
        }
        Mode::Streaming => {
            println!("\n");
            CallResult::Response(accumulated)
        }
        _ => match parse_tool_call(&accumulated) {
            Some((name, args)) => {
                let cut = accumulated.find("</tool_call>")
                    .map(|i| i + "</tool_call>".len())
                    .unwrap_or(accumulated.len());
                CallResult::ToolCall { name, args, head: accumulated[..cut].to_string() }
            }
            None => {
                println!("strategos> {}\n", accumulated.trim());
                CallResult::Response(accumulated)
            }
        },
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    eprintln!("=== Strategos Brain ===");
    eprintln!("Model : {}", MODEL);

    let rag = init_rag();
    let memories = load_memories();

    if memories.is_empty() {
        eprintln!("Memory: none yet");
    } else {
        eprintln!("Memory: {} facts loaded", memories.len());
    }

    eprintln!("Type anything to talk. 'quit' to exit.\n");

    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("failed to build HTTP client");

    let system_prompt = build_system_prompt(&memories);
    let mut messages: Vec<Value> = vec![
        json!({"role": "system", "content": system_prompt}),
    ];

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("you> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 { break; }
        let line = line.trim();
        if line.is_empty() { continue; }
        if line == "quit" || line == "exit" { break; }

        messages.push(json!({"role": "user", "content": line}));
        let start = std::time::Instant::now();
        let mut answered = false;

        for _ in 0..MAX_TOOL_ROUNDS {
            let spinning = Arc::new(AtomicBool::new(true));
            let spin_handle = spawn_spinner(spinning.clone());

            // call() takes ownership of spin_handle and joins it before any printing.
            match call(&client, &messages, spinning, spin_handle) {
                CallResult::Response(content) => {
                    eprintln!("  [{:.0}ms]", start.elapsed().as_secs_f64() * 1000.0);
                    messages.push(json!({"role": "assistant", "content": content}));
                    answered = true;
                    break;
                }
                CallResult::ToolCall { name, args, head } => {
                    eprintln!("  [tool] {} {}", name, args);
                    messages.push(json!({"role": "assistant", "content": head}));
                    let result = execute_tool(&name, &args, &rag);
                    messages.push(json!({
                        "role": "user",
                        "content": format!("<tool_result>\n{}\n</tool_result>", result),
                    }));
                }
                CallResult::Error(e) => {
                    eprintln!("  [error] {}", e);
                    break;
                }
            }
        }

        if !answered && messages.last().map(|m| m["role"] == "user").unwrap_or(false) {
            messages.pop();
        }
    }
}

// ── Tools ─────────────────────────────────────────────────────────────────────

const TOOL_NAMES: &[&str] = &["search_knowledge", "read_file", "remember"];

/// Returns true if the buffer starts with a known tool name followed by a JSON object.
/// Detects SmolLM3's naked call format: `search_knowledge {"query": "..."}`.
fn looks_like_naked_call(s: &str) -> bool {
    let s = s.trim_start();
    TOOL_NAMES.iter().any(|t| {
        s.starts_with(t) && s[t.len()..].trim_start().starts_with('{')
    })
}

/// Strip `<think>...</think>` blocks so tool call parsers see clean text.
fn strip_think_blocks(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(open) = rest.find("<think>") {
        out.push_str(&rest[..open]);
        rest = match rest.find("</think>") {
            Some(close) => &rest[close + "</think>".len()..],
            None => return out, // unclosed block — discard remainder
        };
    }
    out.push_str(rest);
    out
}

fn parse_tool_call(text: &str) -> Option<(String, Value)> {
    let text = &strip_think_blocks(text);
    // Format 1: <tool_call>CONTENT</tool_call>
    //   1a: [{"name":"...","arguments":{...}}]  — native SmolLM array form
    //   1b: {"name":"...","args":{...}}          — single-object fallback
    if let Some(start) = text.find("<tool_call>") {
        let inner_start = start + "<tool_call>".len();
        if let Some(end) = text.find("</tool_call>") {
            let inner = text[inner_start..end].trim();
            let obj = if inner.starts_with('[') {
                // Array form — take first element
                serde_json::from_str::<Value>(inner).ok()
                    .and_then(|v| v.as_array().and_then(|a| a.first().cloned()))
            } else {
                serde_json::from_str::<Value>(inner).ok()
            };
            if let Some(obj) = obj {
                if let Some(name) = obj["name"].as_str() {
                    let args = obj.get("arguments").or_else(|| obj.get("args"))
                        .cloned().unwrap_or(Value::Null);
                    return Some((name.to_string(), args));
                }
            }
        }
    }
    // Remaining formats operate on the trimmed text.
    let trimmed = text.trim();
    // Format 1b: raw JSON array without XML wrapper — [{"name":"...","arguments":{...}}]
    if trimmed.starts_with('[') {
        if let Ok(arr) = serde_json::from_str::<Value>(trimmed) {
            if let Some(obj) = arr.as_array().and_then(|a| a.first()) {
                if let Some(name) = obj["name"].as_str() {
                    let args = obj.get("arguments").or_else(|| obj.get("args"))
                        .cloned().unwrap_or(Value::Null);
                    return Some((name.to_string(), args));
                }
            }
        }
    }
    // Format 2 & 3: raw JSON — SmolLM3 omits the XML wrapper.
    //   Format 2: {"name": "...", "args": {...}}
    //   Format 3: {"name": "...", "query": "..."} (args inlined at top level)
    if trimmed.starts_with('{') {
        if let Ok(json) = serde_json::from_str::<Value>(trimmed) {
            if let Some(name) = json["name"].as_str() {
                // Prefer explicit "args" or "arguments" key; otherwise inline everything except "name".
                let args = if let Some(a) = json.get("args").or_else(|| json.get("arguments")) {
                    a.clone()
                } else {
                    let inlined: serde_json::Map<String, Value> = json.as_object()
                        .map(|o| o.iter()
                            .filter(|(k, _)| k.as_str() != "name")
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect())
                        .unwrap_or_default();
                    Value::Object(inlined)
                };
                return Some((name.to_string(), args));
            }
        }
    }
    // Format 4: "tool_name {args_json}" — SmolLM3 naked call (no JSON wrapper, no "name" key).
    for tool in TOOL_NAMES {
        if trimmed.starts_with(tool) {
            let rest = trimmed[tool.len()..].trim_start();
            if rest.starts_with('{') {
                if let Ok(args) = serde_json::from_str::<Value>(rest) {
                    return Some((tool.to_string(), args));
                }
            }
        }
    }
    None
}

fn execute_tool(name: &str, args: &Value, rag: &Option<RetrievalSystem>) -> String {
    match name {
        "search_knowledge" => {
            let query = args["query"].as_str()
                .or_else(|| args.as_object().and_then(|o| o.values().find_map(|v| v.as_str())))
                .unwrap_or("");
            match rag {
                Some(r) => match r.retrieve(query, RAG_TOP_K) {
                    Ok(results) if !results.is_empty() => results.iter()
                        .map(|r| format!("[{}]\n{}", r.chunk.source, r.chunk.content))
                        .collect::<Vec<_>>()
                        .join("\n\n"),
                    _ => "No relevant documents found.".to_string(),
                },
                None => "Knowledge base not available.".to_string(),
            }
        }
        "read_file" => {
            let path = args["path"].as_str().unwrap_or("");
            if path.is_empty() { return "Error: path required".to_string(); }
            match std::fs::read_to_string(path) {
                Ok(c) if c.len() > 8192 => format!("{}\n[truncated at 8KB]", &c[..8192]),
                Ok(c) => c,
                Err(e) => format!("Error reading {}: {}", path, e),
            }
        }
        "remember" => {
            let fact = args["fact"].as_str()
                .or_else(|| args.as_object().and_then(|o| o.values().find_map(|v| v.as_str())))
                .unwrap_or("");
            if fact.is_empty() { return "Error: fact required".to_string(); }
            save_memory(fact)
        }
        _ => format!("Unknown tool: {}", name),
    }
}

// ── Spinner ───────────────────────────────────────────────────────────────────

fn spawn_spinner(flag: Arc<AtomicBool>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        let mut i = 0usize;
        while flag.load(Ordering::Relaxed) {
            eprint!("\r  {} thinking...", frames[i % frames.len()]);
            io::stderr().flush().ok();
            i += 1;
            std::thread::sleep(Duration::from_millis(80));
        }
        eprint!("\r                      \r");
        io::stderr().flush().ok();
    })
}

// ── RAG init ──────────────────────────────────────────────────────────────────

fn init_rag() -> Option<RetrievalSystem> {
    let rag = RetrievalSystem::new(RetrievalConfig::default()).ok()?;

    let dir = std::path::Path::new(KNOWLEDGE_DIR);
    if !dir.exists() {
        eprintln!("RAG   : no knowledge/ directory — running without context");
        return Some(rag);
    }

    let mut indexed = 0usize;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "md" && ext != "txt" { continue; }
        if let Ok(content) = std::fs::read_to_string(&path) {
            let source = path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown").to_string();
            if let Ok(n) = rag.index_document(&source.replace('.', "_"), &content, &source, None) {
                indexed += n;
            }
        }
    }

    eprintln!("RAG   : {} chunks indexed from {}/", indexed, KNOWLEDGE_DIR);
    Some(rag)
}
