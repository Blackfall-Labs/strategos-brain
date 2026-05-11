use std::io::{self, Write, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use reqwest::blocking::Client;
use serde_json::{json, Value};
use frame_catalog::{RetrievalSystem, RetrievalConfig};

const API_URL: &str = "http://localhost:12434/v1/chat/completions";
const MODEL: &str = "docker.io/ai/phi4:latest"; // swap to phi4:14B-Q4_0 when ready
const KNOWLEDGE_DIR: &str = "knowledge";
const RAG_TOP_K: usize = 5;
const MAX_TOOL_ROUNDS: usize = 8;

const SYSTEM_PROMPT: &str = r#"You are Strategos, a decision-making intelligence for a local system. You reason carefully before responding and provide clear, actionable answers.

You have access to tools. When you need information, call a tool using EXACTLY this format — write nothing after the closing tag:

<tool_call>{"name": "tool_name", "args": {"key": "value"}}</tool_call>

Available tools:
- search_knowledge(query: string) — search the local knowledge base for context
- read_file(path: string) — read a local file by its path

Rules:
- Call only one tool at a time.
- Write nothing after </tool_call> — stop there and wait for the result.
- After receiving a tool result, give your final answer directly.
- If you don't need a tool, answer directly without calling one."#;

fn main() {
    eprintln!("=== Strategos Brain ===");
    eprintln!("Model : {}", MODEL);

    let rag = init_rag();

    eprintln!("Type anything to talk. 'quit' to exit.\n");

    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("failed to build HTTP client");

    let mut messages: Vec<Value> = vec![
        json!({"role": "system", "content": SYSTEM_PROMPT}),
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

            let body = json!({
                "model": MODEL,
                "messages": messages,
                "max_tokens": 1024,
            });
            let result = client.post(API_URL).json(&body).send()
                .and_then(|r| r.json::<Value>());

            spinning.store(false, Ordering::Relaxed);
            spin_handle.join().ok();

            match result {
                Err(e) => {
                    eprintln!("  [error] {}", e);
                    break;
                }
                Ok(response) => {
                    let content = response["choices"][0]["message"]["content"]
                        .as_str()
                        .unwrap_or("")
                        .to_string();

                    match parse_tool_call(&content) {
                        Some((name, args)) => {
                            // Discard anything Phi-4 hallucinated after </tool_call>.
                            let cut = content.find("</tool_call>").unwrap() + "</tool_call>".len();
                            messages.push(json!({"role": "assistant", "content": &content[..cut]}));

                            eprintln!("  [tool] {} {}", name, args);
                            let tool_result = execute_tool(&name, &args, &rag);

                            messages.push(json!({
                                "role": "user",
                                "content": format!("<tool_result>\n{}\n</tool_result>", tool_result),
                            }));
                        }
                        None => {
                            eprintln!("  [{:.0}ms]", start.elapsed().as_secs_f64() * 1000.0);
                            println!("strategos> {}\n", content.trim());
                            messages.push(json!({"role": "assistant", "content": content}));
                            answered = true;
                            break;
                        }
                    }
                }
            }
        }

        // If the loop exhausted without a final answer, clean up the dangling user message.
        if !answered && messages.last().map(|m| m["role"] == "user").unwrap_or(false) {
            messages.pop();
        }
    }
}

/// Extract (name, args) from a `<tool_call>...</tool_call>` block in the response text.
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
            let query = args["query"].as_str().unwrap_or("");
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
            if path.is_empty() {
                return "Error: path required".to_string();
            }
            match std::fs::read_to_string(path) {
                Ok(c) if c.len() > 8192 => format!("{}\n[truncated at 8KB]", &c[..8192]),
                Ok(c) => c,
                Err(e) => format!("Error reading {}: {}", path, e),
            }
        }
        _ => format!("Unknown tool: {}", name),
    }
}

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
            let source = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();
            if let Ok(n) = rag.index_document(&source.replace('.', "_"), &content, &source, None) {
                indexed += n;
            }
        }
    }

    eprintln!("RAG   : {} chunks indexed from {}/", indexed, KNOWLEDGE_DIR);
    Some(rag)
}
