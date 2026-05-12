use crate::vector_store::EMBEDDING_DIM;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Model error: {0}")]
    Model(String),
}

pub type Result<T> = std::result::Result<T, EmbeddingError>;

pub trait EmbeddingGenerator: Send + Sync {
    fn generate(&self, text: &str) -> Result<Vec<f32>>;
    fn dimension(&self) -> usize;
}

/// Bag-of-words embedding. Each word is hashed to a dimension slot and its
/// frequency accumulated, then L2-normalised. Query words overlap with
/// document words — much better than hashing the whole string as one token.
pub struct SimpleEmbeddingGenerator {
    dimension: usize,
}

impl SimpleEmbeddingGenerator {
    pub fn new() -> Self {
        Self { dimension: EMBEDDING_DIM }
    }

    fn hash_word(word: &str) -> u64 {
        let mut h = DefaultHasher::new();
        word.hash(&mut h);
        h.finish()
    }

    fn bow_embedding(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dimension];
        let dim = self.dimension as u64;

        let words: Vec<String> = text
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() > 2)
            .collect();

        if words.is_empty() {
            return embedding;
        }

        for w in &words {
            // Unigram
            let idx = (Self::hash_word(w) % dim) as usize;
            embedding[idx] += 1.0;

            // Stemmed prefix (first 4 chars) — helps "inference" match "infer"
            if w.len() >= 4 {
                let prefix = &w[..4];
                let idx2 = (Self::hash_word(prefix).wrapping_add(0xdeadbeef) % dim) as usize;
                embedding[idx2] += 0.5;
            }
        }

        // Bigrams — captures phrase-level matching
        for pair in words.windows(2) {
            let bigram = format!("{} {}", pair[0], pair[1]);
            let idx = (Self::hash_word(&bigram) % dim) as usize;
            embedding[idx] += 0.75;
        }

        // TF normalise
        let total: f32 = embedding.iter().sum();
        if total > 0.0 {
            embedding.iter_mut().for_each(|x| *x /= total);
        }

        // L2 normalise
        let mag: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag > 0.0 {
            embedding.iter_mut().for_each(|x| *x /= mag);
        }

        embedding
    }
}

impl Default for SimpleEmbeddingGenerator {
    fn default() -> Self { Self::new() }
}

impl EmbeddingGenerator for SimpleEmbeddingGenerator {
    fn generate(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(EmbeddingError::InvalidInput("Empty text".to_string()));
        }
        Ok(self.bow_embedding(text))
    }

    fn dimension(&self) -> usize { self.dimension }
}
