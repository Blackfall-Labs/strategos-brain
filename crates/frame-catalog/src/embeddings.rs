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

/// Hash-based embedding generator — deterministic, no external deps.
/// Not semantically meaningful, but sufficient for keyword-level RAG.
pub struct SimpleEmbeddingGenerator {
    dimension: usize,
}

impl SimpleEmbeddingGenerator {
    pub fn new() -> Self {
        Self { dimension: EMBEDDING_DIM }
    }

    fn hash_to_embedding(&self, text: &str) -> Vec<f32> {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let base_hash = hasher.finish();

        let mut embedding = Vec::with_capacity(self.dimension);
        let mut seed = base_hash;

        for i in 0..self.dimension {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let val = ((seed >> 16) as f32) / 65536.0;
            let normalized = (val * 2.0 - 1.0) * (1.0 + (i as f32 / self.dimension as f32).sin());
            embedding.push(normalized);
        }

        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            embedding.iter_mut().for_each(|x| *x /= magnitude);
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
        Ok(self.hash_to_embedding(text))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}
