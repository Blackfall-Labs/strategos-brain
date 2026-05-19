use std::io::{BufRead, BufReader};
use std::fs::File;
use strategos_runtime::tokenizer::Tokenizer;

/// Read up to `max_tokens` tokens from a plain-text corpus file.
/// Each line is tokenized and concatenated with a newline separator token.
pub fn load_tokens(corpus_path: &str, tokenizer_path: &str, max_tokens: usize) -> Result<Vec<u32>, String> {
    let tokenizer = Tokenizer::load(tokenizer_path)?;
    let file = File::open(corpus_path)
        .map_err(|e| format!("corpus: {e}"))?;

    let mut tokens: Vec<u32> = Vec::with_capacity(max_tokens.min(1_000_000));
    let reader = BufReader::new(file);

    for line in reader.lines() {
        if tokens.len() >= max_tokens { break; }
        let line = line.map_err(|e| format!("read line: {e}"))?;
        let line = line.trim();
        if line.is_empty() { continue; }
        let mut line_tokens = tokenizer.encode(line);
        let remaining = max_tokens - tokens.len();
        line_tokens.truncate(remaining);
        tokens.extend(line_tokens);
    }

    eprintln!("[corpus] {} tokens loaded from {}", tokens.len(), corpus_path);
    Ok(tokens)
}
