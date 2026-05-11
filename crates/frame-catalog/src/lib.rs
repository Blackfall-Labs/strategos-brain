pub mod vector_store;
pub mod embeddings;
pub mod retrieval;

pub use vector_store::{
    VectorStore, VectorStoreConfig, VectorStoreError,
    DocumentChunk, SearchResult, EMBEDDING_DIM,
};

pub use embeddings::{EmbeddingGenerator, EmbeddingError, SimpleEmbeddingGenerator};

pub use retrieval::{RetrievalSystem, RetrievalConfig, RetrievalError};
