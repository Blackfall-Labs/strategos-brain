use std::io::{self, Write, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use reqwest::blocking::Client;
use serde_json::{json, Value};
use frame_catalog::{RetrievalSystem, RetrievalConfig};

const API_URL: &str = "http://localhost:12434/v1/chat/completions";
const MODEL: &str = "docker.io/ai/phi4:latest";
const KNOWLEDGE_DIR: &str = "knowledge";
const RAG_TOP_K: usize = 3;

const SYSTEM_PROMPT: &str = "\
You are Strategos, a decision-making intelligence for a local system. \
You reason carefully before responding and provide clear, actionable answers. \
Be concise but thorough.";

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
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "quit" || line == "exit" {
            break;
        }

        // RAG: retrieve relevant context and prepend to user message.
        let user_content = match &rag {
            Some(r) if r.chunk_count() > 0 => {
                match r.retrieve(line, RAG_TOP_K) {
                    Ok(results) if !results.is_empty() => {
                        let context: String = results.iter()
                            .map(|r| format!("[{}]\n{}", r.chunk.source, r.chunk.content))
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        format!("Relevant context:\n---\n{}\n---\n\n{}", context, line)
                    }
                    _ => line.to_string(),
                }
            }
            _ => line.to_string(),
        };

        messages.push(json!({"role": "user", "content": user_content}));

        // Spinner while waiting.
        let spinning = Arc::new(AtomicBool::new(true));
        let spin_handle = {
            let flag = spinning.clone();
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
        };

        let start = std::time::Instant::now();
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
                messages.pop();
            }
            Ok(json) => {
                let content = json["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("[no response]");

                eprintln!("  [{:.0}ms]", start.elapsed().as_secs_f64() * 1000.0);
                println!("strategos> {}\n", content.trim());

                // Store original user input (not the RAG-augmented version) in history.
                *messages.last_mut().unwrap() = json!({"role": "user", "content": line});
                messages.push(json!({"role": "assistant", "content": content}));
            }
        }
    }
}

/// Load and index all .md and .txt files from the knowledge directory.
fn init_rag() -> Option<RetrievalSystem> {
    let rag = RetrievalSystem::new(RetrievalConfig::default()).ok()?;

    let dir = std::path::Path::new(KNOWLEDGE_DIR);
    if !dir.exists() {
        eprintln!("RAG   : no knowledge/ directory found — running without context");
        return Some(rag);
    }

    let entries = std::fs::read_dir(dir).ok()?;
    let mut indexed = 0usize;

    for entry in entries.flatten() {
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "md" && ext != "txt" {
            continue;
        }
        if let Ok(content) = std::fs::read_to_string(&path) {
            let source = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();
            let doc_id = source.replace('.', "_");
            if let Ok(chunks) = rag.index_document(&doc_id, &content, &source, None) {
                indexed += chunks;
            }
        }
    }

    eprintln!("RAG   : {} chunks indexed from {}/", indexed, KNOWLEDGE_DIR);
    Some(rag)
}
