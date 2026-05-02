//! Strategos REPL — interactive routing + tokenization + prompt formatting.
//! Shows the full pipeline without running model inference.

use std::io::{self, Write, BufRead};
use strategos_runtime::tokenizer::Tokenizer;
use strategos_runtime::router::{Strategos, Expert};
use strategos_runtime::prompt::{self, ChatFormat, Message};

fn main() {
    eprintln!("=== Strategos Brain ===\n");

    let tokenizer = Tokenizer::load("models/qwen3/tokenizer.json")
        .expect("failed to load tokenizer");

    let mut brain = Strategos::new();

    eprintln!("tokenizer: {} vocab", tokenizer.vocab_size());
    eprintln!("type anything to talk. 'quit' to exit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    // Track conversation for context.
    let mut conversation: Vec<Message> = Vec::new();

    loop {
        print!("you> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 { break; }
        let line = line.trim();
        if line.is_empty() { continue; }
        if line == "quit" || line == "exit" { break; }

        if line == "context" {
            eprintln!("  {} messages, {} turns", conversation.len(), brain.context.turn);
            for (i, msg) in conversation.iter().enumerate() {
                let preview: String = msg.content.chars().take(60).collect();
                eprintln!("  [{}] {}: {}...", i, msg.role, preview);
            }
            eprintln!();
            continue;
        }

        // 1. Tokenize input.
        let input_tokens = tokenizer.encode(line);

        // 2. Route.
        let (expert, _mock_response) = brain.process(line);

        // 3. Build the prompt for the selected expert.
        let (format, expert_name) = match &expert {
            Expert::Thinker => (ChatFormat::Phi, "Phi-4 (Thinker)"),
            Expert::Communicator => (ChatFormat::ChatML, "SmolLM3 (Communicator)"),
            Expert::Router => (ChatFormat::ChatML, "Qwen3 (Router)"),
            Expert::Memory => (ChatFormat::ChatML, "Memory → Qwen3"),
        };

        // Build context summary from recent conversation.
        let ctx_summary = if conversation.len() > 2 {
            let recent: Vec<String> = conversation.iter()
                .rev().take(4).rev()
                .map(|m| format!("{}: {}", m.role, &m.content[..m.content.len().min(80)]))
                .collect();
            Some(recent.join("\n"))
        } else {
            None
        };

        let formatted = prompt::format_strategos_prompt(
            line,
            ctx_summary.as_deref(),
            format,
        );
        let prompt_tokens = tokenizer.encode(&formatted);

        // 4. Show results.
        eprintln!("  input: {} tokens", input_tokens.len());
        eprintln!("  route: {:?} → {}", expert, expert_name);
        eprintln!("  prompt: {} tokens", prompt_tokens.len());

        // Show what the model would receive (first 100 chars).
        let prompt_preview: String = formatted.chars().take(120).collect();
        eprintln!("  prompt preview: {}...", prompt_preview.replace('\n', "\\n"));

        // Mock response based on expert.
        let response = match &expert {
            Expert::Thinker => {
                format!("[Phi-4 reasoning about: \"{}\"]", line)
            }
            Expert::Communicator => {
                format!("[SmolLM3 responding to: \"{}\"]", line)
            }
            Expert::Router => {
                format!("[Qwen3 tool calling: \"{}\"]", line)
            }
            Expert::Memory => {
                format!("[Memory search → then route: \"{}\"]", line)
            }
        };

        eprintln!("  {}\n", response);

        // Track conversation.
        conversation.push(Message::user(line));
        conversation.push(Message::assistant(&response));
    }

    eprintln!("\n{} turns total.", brain.context.turn);
}
