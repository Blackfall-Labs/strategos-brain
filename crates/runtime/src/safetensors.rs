//! Safetensors reader — pure Rust, no Python, no torch.
//!
//! Format:
//!   [8 bytes LE] header_size
//!   [header_size bytes] JSON header mapping tensor_name → {dtype, shape, data_offsets}
//!   [remaining bytes] raw tensor data at specified offsets
//!
//! Supports BF16 → f32 conversion (bit shift, no library needed).
//! Writes to our manifest.json + .bin format.

use std::collections::HashMap;
use std::path::Path;

/// Tensor metadata from safetensors header.
#[derive(Debug)]
struct TensorMeta {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    offset_start: usize,
    offset_end: usize,
}

/// Read a safetensors file and return all tensors as f32.
pub fn read_safetensors(path: &Path) -> Result<Vec<(String, Vec<usize>, Vec<f32>)>, String> {
    let data = std::fs::read(path)
        .map_err(|e| format!("{}: {}", path.display(), e))?;

    if data.len() < 8 {
        return Err("file too small".into());
    }

    // Header size (8 bytes LE).
    let header_size = u64::from_le_bytes([
        data[0], data[1], data[2], data[3],
        data[4], data[5], data[6], data[7],
    ]) as usize;

    if 8 + header_size > data.len() {
        return Err(format!("header_size {} exceeds file size {}", header_size, data.len()));
    }

    // Parse JSON header.
    let header_str = std::str::from_utf8(&data[8..8 + header_size])
        .map_err(|e| format!("header UTF-8: {}", e))?;

    let header: serde_json::Value = serde_json::from_str(header_str)
        .map_err(|e| format!("header JSON: {}", e))?;

    let header_obj = header.as_object()
        .ok_or("header is not an object")?;

    // Data region starts after header.
    let data_base = 8 + header_size;

    // Parse tensor metadata.
    let mut tensors = Vec::new();

    for (name, meta) in header_obj {
        if name == "__metadata__" { continue; }

        let meta_obj = meta.as_object()
            .ok_or(format!("{}: meta is not object", name))?;

        let dtype = meta_obj.get("dtype")
            .and_then(|v| v.as_str())
            .ok_or(format!("{}: missing dtype", name))?
            .to_string();

        let shape: Vec<usize> = meta_obj.get("shape")
            .and_then(|v| v.as_array())
            .ok_or(format!("{}: missing shape", name))?
            .iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect();

        let offsets = meta_obj.get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or(format!("{}: missing data_offsets", name))?;

        let offset_start = offsets.get(0).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let offset_end = offsets.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        tensors.push(TensorMeta { name: name.clone(), dtype, shape, offset_start, offset_end });
    }

    // Sort by offset for sequential reading.
    tensors.sort_by_key(|t| t.offset_start);

    // Convert tensors to f32.
    let mut result = Vec::with_capacity(tensors.len());

    for tm in &tensors {
        let raw = &data[data_base + tm.offset_start..data_base + tm.offset_end];
        let numel: usize = tm.shape.iter().product();

        let f32_data = match tm.dtype.as_str() {
            "BF16" => bf16_to_f32(raw, numel)?,
            "F16" => f16_to_f32(raw, numel)?,
            "F32" => raw_to_f32(raw, numel)?,
            other => return Err(format!("{}: unsupported dtype {}", tm.name, other)),
        };

        result.push((tm.name.clone(), tm.shape.clone(), f32_data));
    }

    Ok(result)
}

/// Convert BF16 bytes to f32.
/// BF16 is the upper 16 bits of an f32. Conversion: (bf16 as u32) << 16.
fn bf16_to_f32(raw: &[u8], numel: usize) -> Result<Vec<f32>, String> {
    if raw.len() != numel * 2 {
        return Err(format!("BF16: expected {} bytes, got {}", numel * 2, raw.len()));
    }
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        let bf16_bits = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
        let f32_bits = (bf16_bits as u32) << 16;
        out.push(f32::from_bits(f32_bits));
    }
    Ok(out)
}

/// Convert F16 (IEEE 754 half) to f32.
fn f16_to_f32(raw: &[u8], numel: usize) -> Result<Vec<f32>, String> {
    if raw.len() != numel * 2 {
        return Err(format!("F16: expected {} bytes, got {}", numel * 2, raw.len()));
    }
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        let h = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
        out.push(half_to_f32(h));
    }
    Ok(out)
}

/// IEEE 754 half-precision to single-precision.
fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero.
            f32::from_bits(sign << 31)
        } else {
            // Subnormal.
            let mut m = mant;
            let mut e = 0u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF;
            let f32_exp = 127 - 15 - e + 1;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN.
        let f32_mant = mant << 13;
        f32::from_bits((sign << 31) | (0xFF << 23) | f32_mant)
    } else {
        // Normal.
        let f32_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

/// Raw f32 LE bytes.
fn raw_to_f32(raw: &[u8], numel: usize) -> Result<Vec<f32>, String> {
    if raw.len() != numel * 4 {
        return Err(format!("F32: expected {} bytes, got {}", numel * 4, raw.len()));
    }
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        let bits = u32::from_le_bytes([raw[i*4], raw[i*4+1], raw[i*4+2], raw[i*4+3]]);
        out.push(f32::from_bits(bits));
    }
    Ok(out)
}

/// Export safetensors files to our manifest.json + .bin format.
/// Reads all .safetensors files in a directory, writes manifest + bins to output_dir.
pub fn export_to_bins(
    safetensors_dir: &Path,
    output_dir: &Path,
) -> Result<usize, String> {
    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("create dir: {}", e))?;

    // Find all .safetensors files.
    let mut st_files: Vec<std::path::PathBuf> = Vec::new();
    for entry in std::fs::read_dir(safetensors_dir)
        .map_err(|e| format!("read dir: {}", e))?
    {
        let entry = entry.map_err(|e| format!("dir entry: {}", e))?;
        let path = entry.path();
        if path.extension().map(|e| e == "safetensors").unwrap_or(false) {
            st_files.push(path);
        }
    }
    st_files.sort();

    if st_files.is_empty() {
        return Err("no .safetensors files found".into());
    }

    eprintln!("[safetensors] {} files to process", st_files.len());

    let mut manifest = Vec::new();
    let mut tensor_idx = 0usize;

    for st_path in &st_files {
        eprintln!("[safetensors] reading {}...", st_path.display());
        let tensors = read_safetensors(st_path)?;

        for (name, shape, data) in tensors {
            let bin_name = format!("weight_{:05}.bin", tensor_idx);
            let bin_path = output_dir.join(&bin_name);

            // Write raw f32 LE bytes.
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for &v in &data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&bin_path, &bytes)
                .map_err(|e| format!("write {}: {}", bin_name, e))?;

            manifest.push(serde_json::json!({
                "name": name,
                "file": bin_name,
                "shape": shape,
                "numel": data.len(),
            }));

            tensor_idx += 1;
            if tensor_idx % 50 == 0 {
                eprintln!("[safetensors] {} tensors exported...", tensor_idx);
            }
        }
    }

    // Write manifest.
    let manifest_path = output_dir.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&serde_json::Value::Array(manifest))
        .map_err(|e| format!("serialize manifest: {}", e))?;
    std::fs::write(&manifest_path, manifest_json)
        .map_err(|e| format!("write manifest: {}", e))?;

    eprintln!("[safetensors] exported {} tensors to {}", tensor_idx, output_dir.display());
    Ok(tensor_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_conversion() {
        // BF16 for 1.0 = 0x3F80 (same as upper 16 bits of f32 1.0)
        let raw = [0x80u8, 0x3F]; // LE
        let result = bf16_to_f32(&raw, 1).unwrap();
        assert_eq!(result[0], 1.0);

        // BF16 for -2.0 = 0xC000
        let raw = [0x00u8, 0xC0];
        let result = bf16_to_f32(&raw, 1).unwrap();
        assert_eq!(result[0], -2.0);

        // BF16 for 0.0 = 0x0000
        let raw = [0x00u8, 0x00];
        let result = bf16_to_f32(&raw, 1).unwrap();
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn f16_conversion() {
        // F16 for 1.0 = 0x3C00
        let raw = [0x00u8, 0x3C];
        let result = f16_to_f32(&raw, 1).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6);

        // F16 for 0.0 = 0x0000
        let raw = [0x00u8, 0x00];
        let result = f16_to_f32(&raw, 1).unwrap();
        assert_eq!(result[0], 0.0);
    }
}
