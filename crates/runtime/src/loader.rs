//! Weight loader — manifest.json + .bin files → tensors.
//!
//! Same format proven in ternsig/deepspeech/conformer_native.rs.
//! Each tensor stored as raw f32 little-endian bytes.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Manifest entry for one tensor.
#[derive(Debug, Deserialize)]
pub struct ManifestEntry {
    pub name: String,
    pub file: String,
    pub shape: Vec<usize>,
    pub numel: usize,
}

/// A loaded tensor — raw f32 data with shape.
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Collection of loaded tensors, indexed by name.
pub struct ModelWeights {
    pub tensors: HashMap<String, Tensor>,
}

impl ModelWeights {
    /// Load all tensors from a manifest directory.
    ///
    /// Expected files:
    ///   manifest.json — array of ManifestEntry
    ///   *.bin — raw f32 LE bytes per tensor
    pub fn load(manifest_dir: &Path) -> Result<Self, String> {
        let manifest_path = manifest_dir.join("manifest.json");
        let manifest_text = std::fs::read_to_string(&manifest_path)
            .map_err(|e| format!("manifest.json: {}", e))?;

        let entries: Vec<ManifestEntry> = serde_json::from_str(&manifest_text)
            .map_err(|e| format!("parse manifest: {}", e))?;

        eprintln!("[loader] {} tensors in manifest", entries.len());

        let mut tensors = HashMap::new();
        for entry in &entries {
            let bin_path = manifest_dir.join(&entry.file);
            let raw = std::fs::read(&bin_path)
                .map_err(|e| format!("{}: {}", entry.file, e))?;

            let expected_bytes = entry.numel * 4;
            if raw.len() != expected_bytes {
                return Err(format!(
                    "{}: expected {} bytes, got {}",
                    entry.name, expected_bytes, raw.len()
                ));
            }

            let data: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            tensors.insert(
                entry.name.clone(),
                Tensor {
                    data,
                    shape: entry.shape.clone(),
                },
            );
        }

        eprintln!("[loader] {} tensors loaded", tensors.len());
        Ok(Self { tensors })
    }

    /// Get a tensor by name.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Get a tensor's f32 data by name, or error.
    pub fn require(&self, name: &str) -> Result<&[f32], String> {
        self.tensors.get(name)
            .map(|t| t.data.as_slice())
            .ok_or_else(|| format!("missing tensor: {}", name))
    }

    /// List all tensor names.
    pub fn names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
}
