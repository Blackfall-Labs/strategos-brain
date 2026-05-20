use std::collections::HashMap;
use serde::Deserialize;

/// Byte-level BPE tokenizer loaded from HuggingFace tokenizer.json.
pub struct Tokenizer {
    vocab:  HashMap<String, u32>,   // token_str → token_id
    merges: HashMap<(String, String), u32>,  // (a, b) → merge rank
    byte_table: [char; 256],
}

#[derive(Deserialize)]
struct TokenizerJson {
    model: BpeModel,
}

#[derive(Deserialize)]
struct BpeModel {
    vocab:  HashMap<String, u32>,
    merges: Vec<String>,
}

impl Tokenizer {
    pub fn load(path: &str) -> Self {
        let s = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read tokenizer: {e}"));
        let tj: TokenizerJson = serde_json::from_str(&s)
            .unwrap_or_else(|e| panic!("bad tokenizer JSON: {e}"));

        let merges = tj.model.merges.into_iter().enumerate()
            .filter_map(|(rank, line)| {
                let mut parts = line.splitn(2, ' ');
                let a = parts.next()?.to_string();
                let b = parts.next()?.to_string();
                Some(((a, b), rank as u32))
            })
            .collect();

        Self {
            vocab:      tj.model.vocab,
            merges,
            byte_table: build_byte_table(),
        }
    }

    pub fn vocab_size(&self) -> usize { self.vocab.len() }

    /// Tokenize a text string into token IDs using byte-level BPE.
    /// Splits on whitespace boundaries; word-initial tokens get Ġ prefix.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut first = true;
        for word in text.split_whitespace() {
            let piece = if first {
                first = false;
                self.bytes_to_chars(word)
            } else {
                let mut s = String::from('\u{0120}'); // Ġ prefix
                s.push_str(&self.bytes_to_chars(word));
                s
            };
            self.bpe_encode_word(&piece, &mut out);
        }
        out
    }

    fn bytes_to_chars(&self, s: &str) -> String {
        s.bytes().map(|b| self.byte_table[b as usize]).collect()
    }

    fn bpe_encode_word(&self, word: &str, out: &mut Vec<u32>) {
        if word.is_empty() { return; }

        // Collect grapheme clusters (each BPE "char" may be multi-byte unicode)
        let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        loop {
            if chars.len() < 2 { break; }
            let mut best_rank: Option<u32> = None;
            let mut best_idx  = 0;

            for i in 0..chars.len() - 1 {
                if let Some(&rank) = self.merges.get(&(chars[i].clone(), chars[i + 1].clone())) {
                    if best_rank.map_or(true, |r| rank < r) {
                        best_rank = Some(rank);
                        best_idx  = i;
                    }
                }
            }

            match best_rank {
                None => break,
                Some(_) => {
                    let merged = format!("{}{}", chars[best_idx], chars[best_idx + 1]);
                    chars.remove(best_idx + 1);
                    chars[best_idx] = merged;
                }
            }
        }

        for ch in &chars {
            if let Some(&id) = self.vocab.get(ch) {
                out.push(id);
            }
        }
    }
}

/// GPT-2 / SmolLM2 byte-level encoding: maps each byte 0-255 to a unicode char.
fn build_byte_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let printable: Vec<u8> = (b'!'..=b'~').chain(b'\xA1'..=b'\xAC').chain(b'\xAE'..=b'\xFF').collect();
    let mut extra_char = 256u32;

    for b in 0u8..=255 {
        if printable.contains(&b) {
            table[b as usize] = char::from(b);
        } else {
            table[b as usize] = char::from_u32(extra_char).unwrap();
            extra_char += 1;
        }
    }
    table
}
