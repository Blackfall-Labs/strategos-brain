use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use reqwest::blocking::Client;
use serde_json::{json, Value};
use frame_catalog::{RetrievalSystem, RetrievalConfig};

// ── Endpoints ─────────────────────────────────────────────────────────────────
// Override via env vars in Docker Compose; fall back to local Ollama for dev.

fn dispatcher_url() -> String {
    std::env::var("DISPATCHER_URL")
        .unwrap_or_else(|_| "http://localhost:11434/v1/chat/completions".into())
}

fn reasoner_url() -> String {
    std::env::var("REASONER_URL")
        .unwrap_or_else(|_| "http://localhost:11434/v1/chat/completions".into())
}

// Models used when hitting the llama.cpp server (single model per container).
// In Docker each server runs one model so the model name is whatever llama.cpp
// loaded — by convention we use the standard completions path.
const DISPATCHER_MODEL: &str = "granite4:1b";   // local Ollama dev name
const REASONER_MODEL:   &str = "smollm3-strategos"; // local Ollama dev name

const KNOWLEDGE_DIR: &str = "knowledge";
const MEMORY_FILE:   &str = "memory/facts.json";
const RAG_TOP_K:     usize = 5;
const DECIDE_WINDOW: usize = 64;

// ── Prompts ───────────────────────────────────────────────────────────────────

const REASONER_SYSTEM: &str =
    "You are Strategos, a decision-making intelligence. \
     Answer in 1–3 sentences using only the provided context. \
     Never speculate. If the context does not contain the answer, say so.";

// ── Memory ────────────────────────────────────────────────────────────────────

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
    match serde_json::to_string_pretty(&memories)
        .and_then(|j| std::fs::write(path, j).map_err(serde_json::Error::io))
    {
        Ok(_) => format!("Remembered: {}", fact),
        Err(e) => format!("Error saving memory: {}", e),
    }
}

// ── Stage 1: Dispatcher ───────────────────────────────────────────────────────
// Calls FunctionGemma with the user query and available tools.
// Returns Some((tool_name, args)) if a tool should be called, None otherwise.

fn dispatch(
    client: &Client,
    question: &str,
    memories: &[String],
    spinning: Arc<AtomicBool>,
    spin_handle: std::thread::JoinHandle<()>,
) -> Option<(String, Value)> {
    let mut system = "You are a function dispatcher. \
        Given a user query, call search_knowledge to look up the answer. \
        Always call search_knowledge unless the user is asking you to save a fact (use remember) \
        or read a specific file (use read_file). \
        Never answer directly — always dispatch a tool.".to_string();
    if !memories.is_empty() {
        system.push_str("\n\nKnown facts from memory:\n");
        for m in memories { system.push_str(&format!("- {}\n", m)); }
    }

    let body = json!({
        "model": DISPATCHER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": question},
        ],
        "tools": [
            {"type":"function","function":{"name":"search_knowledge",
                "description":"Search the local knowledge base for documented facts",
                "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
            {"type":"function","function":{"name":"read_file",
                "description":"Read a local file by path",
                "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
            {"type":"function","function":{"name":"remember",
                "description":"Save a fact to persistent memory for future sessions",
                "parameters":{"type":"object","properties":{"fact":{"type":"string"}},"required":["fact"]}}},
        ],
        "max_tokens": 256,
        "stream": true,
    });

    let response = match client.post(&dispatcher_url()).json(&body).send() {
        Ok(r) => r,
        Err(e) => {
            spinning.store(false, Ordering::Relaxed);
            spin_handle.join().ok();
            eprintln!("  [dispatcher error] {}", e);
            return None;
        }
    };

    let reader = std::io::BufReader::new(response);
    let mut tool_name = String::new();
    let mut args_buf  = String::new();
    let mut finish_tool = false;

    for line in reader.lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if !line.starts_with("data: ") { continue; }
        let data = line["data: ".len()..].trim();
        if data == "[DONE]" { break; }
        let chunk: Value = match serde_json::from_str(data) { Ok(v) => v, Err(_) => continue };
        let choice = &chunk["choices"][0];
        if choice["finish_reason"].as_str() == Some("tool_calls") { finish_tool = true; }
        let tc = &choice["delta"]["tool_calls"][0];
        if !tc.is_null() {
            if let Some(n) = tc["function"]["name"].as_str() { tool_name = n.to_string(); }
            if let Some(a) = tc["function"]["arguments"].as_str() { args_buf.push_str(a); }
        }
    }

    spinning.store(false, Ordering::Relaxed);
    spin_handle.join().ok();

    if finish_tool || !tool_name.is_empty() {
        let args = serde_json::from_str(&args_buf).unwrap_or(Value::Null);
        Some((tool_name, args))
    } else {
        None
    }
}

// ── Stage 2: Reasoner ─────────────────────────────────────────────────────────
// Calls SmolLM3 with the original question + retrieved context.
// Streams the response to stdout.

fn reason(
    client: &Client,
    question: &str,
    context: Option<&str>,
    memories: &[String],
    spinning: Arc<AtomicBool>,
    spin_handle: std::thread::JoinHandle<()>,
) -> String {
    let mut system = REASONER_SYSTEM.to_string();
    if !memories.is_empty() {
        system.push_str("\n\nKnown facts from memory:\n");
        for m in memories { system.push_str(&format!("- {}\n", m)); }
    }

    let user_content = match context {
        Some(ctx) => format!("Context:\n{}\n\nQuestion: {}", ctx, question),
        None => question.to_string(),
    };

    // SmolLM3: /no_think disables thinking mode.
    let user_content = format!("/no_think {}", user_content);

    let body = json!({
        "model": REASONER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
        "max_tokens": 512,
        "stream": true,
    });

    let response = match client.post(&reasoner_url()).json(&body).send() {
        Ok(r) => r,
        Err(e) => {
            spinning.store(false, Ordering::Relaxed);
            spin_handle.join().ok();
            eprintln!("  [reasoner error] {}", e);
            return String::new();
        }
    };

    let reader = std::io::BufReader::new(response);
    let mut accumulated = String::new();
    let mut stdout = io::stdout();
    let mut decide_buf = String::new();
    let mut streaming = false;
    let mut in_think = false;
    let mut spin_handle = Some(spin_handle);

    for line in reader.lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if !line.starts_with("data: ") { continue; }
        let data = line["data: ".len()..].trim();
        if data == "[DONE]" { break; }
        let chunk: Value = match serde_json::from_str(data) { Ok(v) => v, Err(_) => continue };
        let delta = chunk["choices"][0]["delta"]["content"].as_str().unwrap_or("");
        accumulated.push_str(delta);

        if delta.contains("<think>") { in_think = true; }
        if in_think {
            if delta.contains("</think>") { in_think = false; decide_buf.clear(); }
            continue;
        }

        if !streaming {
            decide_buf.push_str(delta);
            if decide_buf.len() >= DECIDE_WINDOW {
                spinning.store(false, Ordering::Relaxed);
                if let Some(h) = spin_handle.take() { h.join().ok(); }
                print!("strategos> {}", decide_buf);
                stdout.flush().ok();
                streaming = true;
            }
        } else {
            print!("{}", delta);
            stdout.flush().ok();
        }
    }

    spinning.store(false, Ordering::Relaxed);
    if let Some(h) = spin_handle.take() { h.join().ok(); }

    if !streaming {
        let visible = strip_think_blocks(&decide_buf);
        print!("strategos> {}", visible.trim());
        stdout.flush().ok();
    }
    println!("\n");
    accumulated
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    eprintln!("=== Strategos Brain ===");
    eprintln!("Dispatcher : {} @ {}", DISPATCHER_MODEL, dispatcher_url());
    eprintln!("Reasoner   : {} @ {}", REASONER_MODEL, reasoner_url());

    let rag = init_rag();
    let memories = load_memories();

    if memories.is_empty() { eprintln!("Memory: none yet"); }
    else { eprintln!("Memory: {} facts loaded", memories.len()); }

    eprintln!("Type anything to talk. 'quit' to exit.\n");

    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("failed to build HTTP client");

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

        let start = std::time::Instant::now();

        // ── Stage 1: Dispatch ─────────────────────────────────────────────────
        let spinning = Arc::new(AtomicBool::new(true));
        let spin_handle = spawn_spinner(spinning.clone());

        let tool_call = dispatch(&client, line, &memories, spinning, spin_handle);

        // ── Execute tool (if dispatched) ──────────────────────────────────────
        let context: Option<String> = tool_call.map(|(name, args)| {
            eprintln!("  [tool] {} {}", name, args);
            execute_tool(&name, &args, &rag)
        });

        // ── Stage 2: Reason ───────────────────────────────────────────────────
        let spinning = Arc::new(AtomicBool::new(true));
        let spin_handle = spawn_spinner(spinning.clone());

        reason(&client, line, context.as_deref(), &memories, spinning, spin_handle);

        eprintln!("  [{:.0}ms]", start.elapsed().as_secs_f64() * 1000.0);
    }
}

// ── Tools ─────────────────────────────────────────────────────────────────────

const TOOL_NAMES: &[&str] = &["search_knowledge", "read_file", "remember"];

fn looks_like_naked_call(s: &str) -> bool {
    let s = s.trim_start();
    TOOL_NAMES.iter().any(|t| s.starts_with(t) && s[t.len()..].trim_start().starts_with('{'))
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
        eprintln!("RAG   : no knowledge/ directory");
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

// ── Think-block stripper ──────────────────────────────────────────────────────

fn strip_think_blocks(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(open) = rest.find("<think>") {
        out.push_str(&rest[..open]);
        rest = match rest.find("</think>") {
            Some(close) => &rest[close + "</think>".len()..],
            None => return out,
        };
    }
    out.push_str(rest);
    out
}
