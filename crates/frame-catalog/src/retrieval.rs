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

        // Split on paragraph boundaries first (double newline).
        // This keeps logically related lines together (e.g. table rows with headers).
        let paragraphs: Vec<&str> = text
            .split("\n\n")
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect();

        let mut chunks: Vec<String> = Vec::new();
        let mut current = String::new();

        for para in paragraphs {
            // If para alone exceeds max, split it by lines instead.
            if para.len() > max_size {
                if !current.is_empty() {
                    chunks.push(current.trim().to_string());
                    current = String::new();
                }
                for line in para.lines() {
                    let line = line.trim();
                    if line.is_empty() { continue; }
                    if current.len() + line.len() + 1 > max_size && !current.is_empty() {
                        chunks.push(current.trim().to_string());
                        current = String::new();
                    }
                    if !current.is_empty() { current.push('\n'); }
                    current.push_str(line);
                }
            } else if current.len() + para.len() + 2 > max_size && !current.is_empty() {
                chunks.push(current.trim().to_string());
                current = para.to_string();
            } else {
                if !current.is_empty() { current.push_str("\n\n"); }
                current.push_str(para);
            }
        }
        if !current.trim().is_empty() {
            chunks.push(current.trim().to_string());
        }

        if chunks.is_empty() {
            vec![text.to_string()]
        } else {
            chunks
        }
    }
}
