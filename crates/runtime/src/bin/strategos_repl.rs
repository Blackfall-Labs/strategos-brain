//! Strategos REPL — interactive routing + tokenization test.
//! No model inference. Tests the routing logic and tokenizer.

use std::io::{self, Write, BufRead};
use strategos_runtime::tokenizer::Tokenizer;
use strategos_runtime::router::Strategos;

fn main() {
    eprintln!("=== Strategos REPL ===\n");

    // Load Qwen3 tokenizer (smallest, fastest to load).
    let tokenizer = Tokenizer::load("models/qwen3/tokenizer.json")
        .expect("failed to load tokenizer");

    let mut brain = Strategos::new();

    eprintln!("tokenizer: {} vocab", tokenizer.vocab_size());
    eprintln!("commands: type anything to see routing + tokenization");
    eprintln!("          'quit' to exit\n");

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

        // Tokenize.
        let tokens = tokenizer.encode(line);
        let decoded: Vec<String> = tokens.iter()
            .map(|&id| format!("{}:{}", id, tokenizer.decode_token(id)))
            .collect();

        eprintln!("  tokens ({}): {}", tokens.len(), decoded.join(" "));

        // Route.
        let (expert, response) = brain.process(line);
        eprintln!("  route: {:?}", expert);
        eprintln!("  {}", response);
        eprintln!("  context: {} turns, {} total tokens\n",
            brain.context.turn / 2, brain.context.total_tokens());
    }
}
