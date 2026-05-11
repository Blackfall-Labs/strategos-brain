use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use reqwest::blocking::Client;
use serde_json::{json, Value};
use frame_catalog::{RetrievalSystem, RetrievalConfig};

const API_URL: &str = "http://localhost:12434/v1/chat/completions";
const MODEL: &str = "docker.io/ai/phi4:14B-Q4_0";
const KNOWLEDGE_DIR: &str = "knowledge";
const MEMORY_FILE: &str = "memory/facts.json";
const RAG_TOP_K: usize = 5;
const MAX_TOOL_ROUNDS: usize = 8;
// Buffer this many chars before deciding stream vs tool-call.
// <tool_call> is 11 chars so 64 is a comfortable detection window.
const DECIDE_WINDOW: usize = 64;

const SYSTEM_PROMPT_BASE: &str = r#"You are Strategos, a decision-making intelligence for a local system. You reason carefully before responding and provide clear, actionable answers.

You have access to tools. When you need information or want to save something, call a tool using EXACTLY this format — write nothing after the closing tag:

<tool_call>{"name": "search_knowledge", "args": {"query": "your search terms"}}</tool_call>

Available tools:
- search_knowledge — args: {"query": "string"} — search the local knowledge base
- read_file      — args: {"path": "string"}  — read a local file by its path
- remember       — args: {"fact": "string"}  — save an important fact to persistent memory for future sessions

Rules:
- Call only one tool at a time.
- Use the exact argument names shown above (query, path, fact).
- Write nothing after </tool_call> — stop there and wait for the result.
- After receiving a tool result, give your final answer directly.
- If search_knowledge returns no results and you don't know the answer from memory, say so — do not speculate.
- If you don't need a tool, answer directly without calling one.
- Use remember proactively when the user shares something worth retaining."#;

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

fn call(client: &Client, messages: &[Value], spinning: Arc<AtomicBool>) -> CallResult {
    let body = json!({
        "model": MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "stream": true,
    });

    let response = match client.post(API_URL).json(&body).send() {
        Ok(r) => r,
        Err(e) => return CallResult::Error(e.to_string()),
    };

    let reader = std::io::BufReader::new(response);
    let mut accumulated = String::new();
    let mut stdout = io::stdout();

    enum Mode { Deciding, Streaming, ToolCall }
    let mut mode = Mode::Deciding;
    let mut decide_buf = String::new();

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

        match mode {
            Mode::Deciding => {
                decide_buf.push_str(delta);
                if decide_buf.contains("<tool_call>") {
                    mode = Mode::ToolCall;
                } else if decide_buf.len() >= DECIDE_WINDOW {
                    // No tool call in window — this is a real response.
                    spinning.store(false, Ordering::Relaxed);
                    // Clear the spinner line before printing.
                    eprint!("\r                      \r");
                    io::stderr().flush().ok();
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

    spinning.store(false, Ordering::Relaxed);

    // Handle responses shorter than DECIDE_WINDOW that never triggered Streaming.
    match mode {
        Mode::Deciding if !decide_buf.contains("<tool_call>") => {
            eprint!("\r                      \r");
            io::stderr().flush().ok();
            print!("strategos> {}", decide_buf);
            stdout.flush().ok();
            println!("\n");
            CallResult::Response(accumulated)
        }
        Mode::Streaming => {
            println!("\n");
            CallResult::Response(accumulated)
        }
        _ => {
            // ToolCall or a Deciding buffer that contains <tool_call>.
            match parse_tool_call(&accumulated) {
                Some((name, args)) => {
                    let cut = accumulated.find("</tool_call>")
                        .map(|i| i + "</tool_call>".len())
                        .unwrap_or(accumulated.len());
                    CallResult::ToolCall { name, args, head: accumulated[..cut].to_string() }
                }
                None => {
                    // Couldn't parse — treat as a response.
                    eprint!("\r                      \r");
                    println!("strategos> {}\n", accumulated.trim());
                    CallResult::Response(accumulated)
                }
            }
        }
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

            match call(&client, &messages, spinning) {
                CallResult::Response(content) => {
                    spin_handle.join().ok();
                    eprintln!("  [{:.0}ms]", start.elapsed().as_secs_f64() * 1000.0);
                    messages.push(json!({"role": "assistant", "content": content}));
                    answered = true;
                    break;
                }
                CallResult::ToolCall { name, args, head } => {
                    spin_handle.join().ok();
                    eprintln!("  [tool] {} {}", name, args);
                    messages.push(json!({"role": "assistant", "content": head}));
                    let result = execute_tool(&name, &args, &rag);
                    messages.push(json!({
                        "role": "user",
                        "content": format!("<tool_result>\n{}\n</tool_result>", result),
                    }));
                }
                CallResult::Error(e) => {
                    spin_handle.join().ok();
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

fn parse_tool_call(text: &str) -> Option<(String, Value)> {
    let start = text.find("<tool_call>")? + "<tool_call>".len();
    let end = text.find("</tool_call>")?;
    let json: Value = serde_json::from_str(text[start..end].trim()).ok()?;
    let name = json["name"].as_str()?.to_string();
    let args = json["args"].clone();
    Some((name, args))
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
