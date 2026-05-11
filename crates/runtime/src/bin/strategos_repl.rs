use std::io::{self, Write, BufRead};
use reqwest::blocking::Client;
use serde_json::{json, Value};

const API_URL: &str = "http://localhost:12434/v1/chat/completions";
const MODEL: &str = "ai/phi4:latest";

const SYSTEM_PROMPT: &str = "\
You are Strategos, a decision-making intelligence for a local system. \
You reason carefully before responding and provide clear, actionable answers. \
Be concise but thorough.";

fn main() {
    eprintln!("=== Strategos Brain ===");
    eprintln!("Model : {}", MODEL);
    eprintln!("API   : {}", API_URL);
    eprintln!("Type anything to talk. 'quit' to exit.\n");

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(120))
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

        messages.push(json!({"role": "user", "content": line}));

        let start = std::time::Instant::now();

        let body = json!({
            "model": MODEL,
            "messages": messages,
            "max_tokens": 1024,
        });

        let result = client
            .post(API_URL)
            .json(&body)
            .send()
            .and_then(|r| r.json::<Value>());

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

                messages.push(json!({"role": "assistant", "content": content}));
            }
        }
    }
}
