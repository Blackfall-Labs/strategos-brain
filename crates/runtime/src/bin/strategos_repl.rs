//! Strategos REPL — interactive conversation with MoE routing.
//!
//! Loads all three experts at startup and routes each input to the right model:
//!   Router/tool calls  → Qwen3 1.7B   (ChatML, EOS 151645)
//!   General response   → SmolLM3 3B   (ChatML, EOS 128012)
//!   Reasoning/analysis → Phi-4 Mini   (Phi format, EOS 200018)

use std::io::{self, Write, BufRead};
use std::path::Path;
use strategos_runtime::tokenizer::Tokenizer;
use strategos_runtime::router::{Strategos, Expert};
use strategos_runtime::prompt::{self, ChatFormat, Message};
use strategos_runtime::loader::ModelWeights;
use strategos_runtime::qwen3::Qwen3;
use strategos_runtime::smollm3::SmolLM3;
use strategos_runtime::phi4::Phi4;

const MAX_NEW_TOKENS: usize = 256;

fn main() {
    eprintln!("=== Strategos Brain ===\n");

    // --- Load tokenizers (each model has its own vocab) ---
    let tok_qwen  = Tokenizer::load("models/qwen3/tokenizer.json")
        .expect("failed to load Qwen3 tokenizer");
    let tok_smol  = Tokenizer::load("models/smollm3/tokenizer.json")
        .expect("failed to load SmolLM3 tokenizer");
    let tok_phi4  = Tokenizer::load("models/phi4/tokenizer.json")
        .expect("failed to load Phi-4 tokenizer");

    // --- Load models ---
    eprintln!("loading Qwen3 1.7B...");
    let w_qwen = ModelWeights::load(Path::new("models/qwen3"))
        .expect("failed to load Qwen3 weights");
    let mut qwen3 = Qwen3::load(&w_qwen).expect("failed to init Qwen3");

    eprintln!("loading SmolLM3 3B...");
    let w_smol = ModelWeights::load(Path::new("models/smollm3"))
        .expect("failed to load SmolLM3 weights");
    let mut smollm3 = SmolLM3::load(&w_smol).expect("failed to init SmolLM3");

    eprintln!("loading Phi-4 Mini 3.8B...");
    let w_phi4 = ModelWeights::load(Path::new("models/phi4"))
        .expect("failed to load Phi-4 weights");
    let mut phi4 = Phi4::load(&w_phi4).expect("failed to init Phi-4");

    eprintln!("\nAll experts ready. Type anything to talk. 'quit' to exit.\n");

    let mut brain = Strategos::new();
    let mut conversation: Vec<Message> = Vec::new();

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

        // Route the input.
        let (expert, _) = brain.process(line);
        conversation.push(Message::user(line));

        let start = std::time::Instant::now();

        let response = match expert {
            Expert::Router => {
                // Qwen3 handles tool calls and routing decisions directly.
                let prompt = prompt::format_router_prompt(line, &[]);
                generate_response(&mut qwen3, &tok_qwen, &prompt, 151645, ChatFormat::ChatML)
            }
            Expert::Communicator | Expert::Memory => {
                // SmolLM3 generates the response. Memory falls through until RAG is wired.
                let prompt = prompt::format_strategos_prompt(line, None, ChatFormat::ChatML);
                generate_smollm3(&mut smollm3, &tok_smol, &prompt, 128012)
            }
            Expert::Thinker => {
                // Phi-4 reasons through the answer.
                let prompt = prompt::format_strategos_prompt(line, None, ChatFormat::Phi);
                generate_phi4(&mut phi4, &tok_phi4, &prompt, 200018)
            }
        };

        let elapsed = start.elapsed();
        let total_ms = elapsed.as_secs_f64() * 1000.0;

        eprintln!("  [{:?}] {:.0}ms", expert, total_ms);

        println!("strategos> {}", response.trim());
        println!();

        conversation.push(Message::assistant(response.trim()));
    }
}

/// Run Qwen3 on a pre-formatted prompt string, return decoded response.
fn generate_response(
    model: &mut Qwen3,
    tokenizer: &Tokenizer,
    prompt: &str,
    eos: u32,
    _format: ChatFormat,
) -> String {
    model.reset();
    let tokens = tokenizer.encode(prompt);

    for &tok in &tokens {
        let _ = model.forward_one(tok);
    }

    let mut out = Vec::new();
    let mut last = *tokens.last().unwrap_or(&1);

    for _ in 0..MAX_NEW_TOKENS {
        let logits = model.forward_one(last);
        last = argmax(&logits);
        if last == eos { break; }
        out.push(last);
    }

    tokenizer.decode(&out)
}

/// Run SmolLM3 on a pre-formatted prompt string.
fn generate_smollm3(
    model: &mut SmolLM3,
    tokenizer: &Tokenizer,
    prompt: &str,
    eos: u32,
) -> String {
    model.reset();
    let tokens = tokenizer.encode(prompt);

    for &tok in &tokens {
        let _ = model.forward_one(tok);
    }

    let mut out = Vec::new();
    let mut last = *tokens.last().unwrap_or(&1);

    for _ in 0..MAX_NEW_TOKENS {
        let logits = model.forward_one(last);
        last = argmax(&logits);
        if last == eos { break; }
        out.push(last);
    }

    tokenizer.decode(&out)
}

/// Run Phi-4 on a pre-formatted prompt string.
fn generate_phi4(
    model: &mut Phi4,
    tokenizer: &Tokenizer,
    prompt: &str,
    eos: u32,
) -> String {
    model.reset();
    let tokens = tokenizer.encode(prompt);

    for &tok in &tokens {
        let _ = model.forward_one(tok);
    }

    let mut out = Vec::new();
    let mut last = *tokens.last().unwrap_or(&1);

    for _ in 0..MAX_NEW_TOKENS {
        let logits = model.forward_one(last);
        last = argmax(&logits);
        if last == eos { break; }
        out.push(last);
    }

    tokenizer.decode(&out)
}

#[inline]
fn argmax(logits: &[f32]) -> u32 {
    logits.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
