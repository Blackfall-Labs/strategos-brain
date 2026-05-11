use hnsw_rs::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub const EMBEDDING_DIM: usize = 384;

#[derive(Debug, thiserror::Error)]
pub enum VectorStoreError {
    #[error("HNSW error: {0}")]
    Hnsw(String),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Invalid embedding dimension: expected {EMBEDDING_DIM}, got {0}")]
    InvalidDimension(usize),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, VectorStoreError>;

#[derive(Debug, Clone)]
pub struct DocumentChunk {
    pub id: String,
    pub content: String,
    pub source: String,
    pub metadata: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: DocumentChunk,
    pub score: f32,
}

pub struct VectorStore {
    hnsw: Arc<RwLock<Hnsw<'static, f32, DistL2>>>,
    documents: Arc<RwLock<HashMap<usize, DocumentChunk>>>,
    next_id: Arc<RwLock<usize>>,
    config: VectorStoreConfig,
}

#[derive(Debug, Clone)]
pub struct VectorStoreConfig {
    pub max_connections: usize,
    pub ef_construction: usize,
    pub max_elements: usize,
    pub ef_search: usize,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            max_connections: 16,
            ef_construction: 200,
            max_elements: 100_000,
            ef_search: 100,
        }
    }
}

impl VectorStore {
    pub fn new(config: VectorStoreConfig) -> Result<Self> {
        let hnsw = Hnsw::<f32, DistL2>::new(
            config.max_connections,
            config.max_elements,
            EMBEDDING_DIM,
            config.ef_construction,
            DistL2 {},
        );
        Ok(Self {
            hnsw: Arc::new(RwLock::new(hnsw)),
            documents: Arc::new(RwLock::new(HashMap::new())),
            next_id: Arc::new(RwLock::new(0)),
            config,
        })
    }

    pub fn add_chunk(&self, chunk: DocumentChunk, embedding: &[f32]) -> Result<usize> {
        if embedding.len() != EMBEDDING_DIM {
            return Err(VectorStoreError::InvalidDimension(embedding.len()));
        }
        let mut next_id = self.next_id.write().unwrap();
        let id = *next_id;
        *next_id += 1;
        drop(next_id);

        self.hnsw.write().unwrap().insert((embedding, id));
        self.documents.write().unwrap().insert(id, chunk);
        Ok(id)
    }

    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        if query_embedding.len() != EMBEDDING_DIM {
            return Err(VectorStoreError::InvalidDimension(query_embedding.len()));
        }
        let results = self.hnsw.read().unwrap().search(query_embedding, top_k, self.config.ef_search);
        let documents = self.documents.read().unwrap();
        Ok(results.iter().filter_map(|n| {
            documents.get(&n.d_id).map(|chunk| SearchResult {
                chunk: chunk.clone(),
                score: 1.0 / (1.0 + n.distance),
            })
        }).collect())
    }

    pub fn len(&self) -> usize {
        self.documents.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct VectorStoreStats {
    pub num_chunks: usize,
    pub embedding_dim: usize,
}

impl VectorStore {
    pub fn stats(&self) -> VectorStoreStats {
        VectorStoreStats {
            num_chunks: self.documents.read().unwrap().len(),
            embedding_dim: EMBEDDING_DIM,
        }
    }
}
