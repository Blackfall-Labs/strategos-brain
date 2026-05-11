use crate::embeddings::{EmbeddingGenerator, SimpleEmbeddingGenerator};
use crate::vector_store::{DocumentChunk, SearchResult, VectorStore, VectorStoreConfig};
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum RetrievalError {
    #[error("Vector store error: {0}")]
    VectorStore(#[from] crate::vector_store::VectorStoreError),
    #[error("Embedding error: {0}")]
    Embedding(#[from] crate::embeddings::EmbeddingError),
}

pub type Result<T> = std::result::Result<T, RetrievalError>;

#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    pub max_chunk_size: usize,
    pub chunk_overlap: usize,
    pub vector_config: VectorStoreConfig,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 512,
            chunk_overlap: 50,
            vector_config: VectorStoreConfig::default(),
        }
    }
}

pub struct RetrievalSystem {
    vector_store: VectorStore,
    embedder: Arc<dyn EmbeddingGenerator>,
    config: RetrievalConfig,
}

impl RetrievalSystem {
    pub fn new(config: RetrievalConfig) -> Result<Self> {
        let vector_store = VectorStore::new(config.vector_config.clone())?;
        let embedder: Arc<dyn EmbeddingGenerator> = Arc::new(SimpleEmbeddingGenerator::new());
        Ok(Self { vector_store, embedder, config })
    }

    pub fn index_document(
        &self,
        document_id: &str,
        content: &str,
        source: &str,
        metadata: Option<String>,
    ) -> Result<usize> {
        let chunks = self.split_into_chunks(content);
        let mut count = 0;
        for (i, chunk_text) in chunks.iter().enumerate() {
            let chunk = DocumentChunk {
                id: format!("{}:{}", document_id, i),
                content: chunk_text.clone(),
                source: source.to_string(),
                metadata: metadata.clone(),
            };
            let embedding = self.embedder.generate(chunk_text)?;
            self.vector_store.add_chunk(chunk, &embedding)?;
            count += 1;
        }
        Ok(count)
    }

    pub fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let query_embedding = self.embedder.generate(query)?;
        Ok(self.vector_store.search(&query_embedding, top_k)?)
    }

    pub fn chunk_count(&self) -> usize {
        self.vector_store.len()
    }

    fn split_into_chunks(&self, text: &str) -> Vec<String> {
        let max_size = self.config.max_chunk_size;
        let overlap = self.config.chunk_overlap;

        if text.len() <= max_size {
            return vec![text.to_string()];
        }

        let mut chunks = Vec::new();
        let mut start = 0;
        while start < text.len() {
            let end = (start + max_size).min(text.len());
            chunks.push(text[start..end].to_string());
            if end >= text.len() { break; }
            start += max_size - overlap;
        }
        chunks
    }
}
