//! BPE Tokenizer — loads HuggingFace tokenizer.json format.
//!
//! Supports encode (text → tokens) and decode (tokens → text).
//! No Python, no external dependencies beyond serde_json.

use std::collections::HashMap;

/// A BPE tokenizer loaded from tokenizer.json.
pub struct Tokenizer {
    /// Token string → token ID.
    vocab: HashMap<String, u32>,
    /// Token ID → token string.
    vocab_inv: Vec<String>,
    /// Merge rules: (pair_a, pair_b) → merged token, ordered by priority.
    merges: Vec<(String, String)>,
    /// Special tokens.
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub pad_id: Option<u32>,
}

impl Tokenizer {
    /// Load from a tokenizer.json file.
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("tokenizer.json: {}", e))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| format!("parse tokenizer: {}", e))?;

        let model = json.get("model")
            .ok_or("missing 'model' in tokenizer.json")?;

        // Vocab: string → id.
        let vocab_obj = model.get("vocab")
            .and_then(|v| v.as_object())
            .ok_or("missing model.vocab")?;

        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        let mut max_id = 0u32;

        for (token, id_val) in vocab_obj {
            let id = id_val.as_u64().ok_or("vocab id not integer")? as u32;
            vocab.insert(token.clone(), id);
            if id > max_id { max_id = id; }
        }

        // Added tokens (special tokens with their IDs).
        if let Some(added) = json.get("added_tokens").and_then(|v| v.as_array()) {
            for entry in added {
                if let (Some(content), Some(id)) = (
                    entry.get("content").and_then(|v| v.as_str()),
                    entry.get("id").and_then(|v| v.as_u64()),
                ) {
                    let id = id as u32;
                    vocab.insert(content.to_string(), id);
                    if id > max_id { max_id = id; }
                }
            }
        }

        // Build inverse vocab.
        let mut vocab_inv = vec![String::new(); (max_id + 1) as usize];
        for (token, &id) in &vocab {
            if (id as usize) < vocab_inv.len() {
                vocab_inv[id as usize] = token.clone();
            }
        }

        // Merges: list of "tokenA tokenB" strings.
        let merges_arr = model.get("merges")
            .and_then(|v| v.as_array())
            .ok_or("missing model.merges")?;

        let mut merges = Vec::with_capacity(merges_arr.len());
        for m in merges_arr {
            if let Some(s) = m.as_str() {
                let parts: Vec<&str> = s.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    merges.push((parts[0].to_string(), parts[1].to_string()));
                }
            }
        }

        // Find special token IDs.
        let bos_id = vocab.get("<|im_start|>").or(vocab.get("<s>")).or(vocab.get("<|begin_of_text|>")).copied();
        let eos_id = vocab.get("<|im_end|>").or(vocab.get("</s>")).or(vocab.get("<|end_of_text|>")).or(vocab.get("<|endoftext|>")).copied();
        let pad_id = vocab.get("<|pad|>").or(vocab.get("<pad>")).copied();

        eprintln!("[tokenizer] loaded: {} vocab, {} merges, bos={:?}, eos={:?}",
            vocab.len(), merges.len(), bos_id, eos_id);

        Ok(Self { vocab, vocab_inv, merges, bos_id, eos_id, pad_id })
    }

    /// Encode text to token IDs.
    ///
    /// 1. Extract special tokens (like <|im_start|>) as atomic units.
    /// 2. Split remaining text on whitespace, preserving Ġ prefix.
    /// 3. Greedy longest-match against vocab for each piece.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();

        // First pass: split on special tokens (anything matching <|...|>).
        // These must be looked up as whole units.
        let mut remaining = text;
        while !remaining.is_empty() {
            // Look for special token pattern.
            if let Some(start) = remaining.find("<|") {
                // Encode text before the special token.
                if start > 0 {
                    self.encode_text_piece(&remaining[..start], &mut result);
                }
                // Find the end of the special token.
                if let Some(end) = remaining[start..].find("|>") {
                    let special = &remaining[start..start + end + 2];
                    if let Some(&id) = self.vocab.get(special) {
                        result.push(id);
                    }
                    remaining = &remaining[start + end + 2..];
                } else {
                    // No closing |>, treat rest as text.
                    self.encode_text_piece(remaining, &mut result);
                    break;
                }
            } else {
                // No more special tokens, encode rest as text.
                self.encode_text_piece(remaining, &mut result);
                break;
            }
        }

        result
    }

    /// Encode a text segment (no special tokens) into token IDs.
    fn encode_text_piece(&self, text: &str, result: &mut Vec<u32>) {
        if text.is_empty() { return; }

        // Split on whitespace and newlines, preserving Ġ/Ċ prefixes.
        let mut pieces: Vec<String> = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if ch == ' ' {
                if !current.is_empty() {
                    pieces.push(current.clone());
                    current.clear();
                }
                current.push('Ġ');
            } else if ch == '\n' {
                if !current.is_empty() {
                    pieces.push(current.clone());
                    current.clear();
                }
                // Try Ċ first, else literal newline.
                if self.vocab.contains_key("Ċ") {
                    pieces.push("Ċ".to_string());
                } else if let Some(&id) = self.vocab.get("\n") {
                    // Will be picked up by the greedy matcher.
                    pieces.push("\n".to_string());
                }
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() {
            pieces.push(current);
        }

        // Greedy longest-match for each piece.
        for piece in &pieces {
            if let Some(&id) = self.vocab.get(piece) {
                result.push(id);
            } else {
                let chars: Vec<char> = piece.chars().collect();
                let mut pos = 0;
                while pos < chars.len() {
                    let mut best_len = 0;
                    let mut best_id = None;

                    for end in (pos + 1..=chars.len()).rev() {
                        let substr: String = chars[pos..end].iter().collect();
                        if let Some(&id) = self.vocab.get(&substr) {
                            best_len = end - pos;
                            best_id = Some(id);
                            break;
                        }
                    }

                    if let Some(id) = best_id {
                        result.push(id);
                        pos += best_len;
                    } else {
                        let s = chars[pos].to_string();
                        if let Some(&id) = self.vocab.get(&s) {
                            result.push(id);
                        }
                        pos += 1;
                    }
                }
            }
        }
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut text = String::new();
        for &id in tokens {
            if (id as usize) < self.vocab_inv.len() {
                let token = &self.vocab_inv[id as usize];
                // Handle byte-level encoding: Ġ = space prefix in GPT-2 style.
                let cleaned = token
                    .replace("Ġ", " ")
                    .replace("Ċ", "\n")
                    .replace("ĉ", "\t");
                text.push_str(&cleaned);
            }
        }
        text
    }

    /// Decode a single token to its string representation.
    pub fn decode_token(&self, id: u32) -> &str {
        if (id as usize) < self.vocab_inv.len() {
            &self.vocab_inv[id as usize]
        } else {
            "<unk>"
        }
    }

    /// Vocab size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Look up a token string to get its ID.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }
}
