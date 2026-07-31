use crate::quant::{QuantTensor, QuantType};
use anyhow::{bail, ensure, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: usize = 32;

#[derive(Debug, Clone)]
pub enum MetadataValue {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    /// Array payloads are parsed and skipped. Model loading only needs their length.
    Array {
        element_type: u32,
        len: usize,
    },
}

impl MetadataValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F64(v) => Some(*v as f32),
            Self::U64(v) => Some(*v as f32),
            Self::I64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn array_len(&self) -> Option<usize> {
        match self {
            Self::Array { len, .. } => Some(*len),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    /// GGML dimension order: dimensions[0] is the contiguous row width.
    pub dimensions: Vec<usize>,
    pub ggml_type: u32,
    pub offset: usize,
}

#[derive(Debug)]
pub struct GgufFile {
    mmap: Arc<Mmap>,
    pub version: u32,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    data_offset: usize,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file)? });
        let mut reader = Reader::new(&mmap);
        ensure!(
            reader.take(4)? == GGUF_MAGIC,
            "{} is not a GGUF file",
            path.display()
        );
        let version = reader.u32()?;
        ensure!(
            (2..=3).contains(&version),
            "unsupported GGUF version {version}; versions 2 and 3 are supported"
        );
        let tensor_count =
            usize::try_from(reader.u64()?).context("GGUF tensor count does not fit usize")?;
        let metadata_count =
            usize::try_from(reader.u64()?).context("GGUF metadata count does not fit usize")?;
        ensure!(
            tensor_count <= 1_000_000 && metadata_count <= 1_000_000,
            "unreasonable GGUF header counts"
        );

        let mut metadata = HashMap::with_capacity(metadata_count);
        for _ in 0..metadata_count {
            let key = reader.string()?;
            let value_type = reader.u32()?;
            let value = reader.metadata_value(value_type)?;
            ensure!(
                metadata.insert(key.clone(), value).is_none(),
                "duplicate GGUF metadata key {key}"
            );
        }

        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = reader.string()?;
            let dimensions_count = reader.u32()? as usize;
            ensure!(
                (1..=4).contains(&dimensions_count),
                "tensor {name} has invalid dimension count {dimensions_count}"
            );
            let mut dimensions = Vec::with_capacity(dimensions_count);
            for _ in 0..dimensions_count {
                dimensions.push(
                    usize::try_from(reader.u64()?).context("GGUF dimension does not fit usize")?,
                );
            }
            let ggml_type = reader.u32()?;
            let offset =
                usize::try_from(reader.u64()?).context("GGUF tensor offset does not fit usize")?;
            let info = GgufTensorInfo {
                name: name.clone(),
                dimensions,
                ggml_type,
                offset,
            };
            ensure!(
                tensors.insert(name.clone(), info).is_none(),
                "duplicate GGUF tensor {name}"
            );
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(MetadataValue::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT as u64) as usize;
        ensure!(
            alignment.is_power_of_two() && alignment <= 1 << 20,
            "invalid GGUF alignment {alignment}"
        );
        let data_offset = align_up(reader.position(), alignment)?;
        ensure!(
            data_offset <= mmap.len(),
            "GGUF tensor data offset is outside the file"
        );

        let result = Self {
            mmap,
            version,
            metadata,
            tensors,
            data_offset,
        };
        // Validate every range eagerly so malformed files cannot panic later in a kernel.
        for info in result.tensors.values() {
            result.tensor_range(info)?;
        }
        Ok(result)
    }

    pub fn metadata_u64(&self, key: &str) -> Result<u64> {
        self.metadata
            .get(key)
            .and_then(MetadataValue::as_u64)
            .with_context(|| format!("missing or non-integer GGUF metadata {key}"))
    }

    pub fn metadata_f32(&self, key: &str) -> Result<f32> {
        self.metadata
            .get(key)
            .and_then(MetadataValue::as_f32)
            .with_context(|| format!("missing or non-numeric GGUF metadata {key}"))
    }

    pub fn metadata_str(&self, key: &str) -> Result<&str> {
        self.metadata
            .get(key)
            .and_then(MetadataValue::as_str)
            .with_context(|| format!("missing or non-string GGUF metadata {key}"))
    }

    pub fn tensor_info(&self, name: &str) -> Result<&GgufTensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing GGUF tensor {name}"))
    }

    pub fn tensor(&self, name: &str) -> Result<QuantTensor> {
        let info = self.tensor_info(name)?;
        ensure!(
            info.dimensions.len() <= 2,
            "tensor {name} has {} dimensions; Qwen2 loader supports 1D/2D weights",
            info.dimensions.len()
        );
        let ty =
            QuantType::from_ggml_type(info.ggml_type).with_context(|| format!("tensor {name}"))?;
        let (start, len) = self.tensor_range(info)?;
        let shape = match info.dimensions.as_slice() {
            [len] => vec![*len],
            [cols, rows] => vec![*rows, *cols],
            _ => unreachable!(),
        };
        QuantTensor::from_mmap(self.mmap.clone(), start, len, shape, ty)
            .with_context(|| format!("invalid GGUF tensor {name}"))
    }

    fn tensor_range(&self, info: &GgufTensorInfo) -> Result<(usize, usize)> {
        let ty = QuantType::from_ggml_type(info.ggml_type)
            .with_context(|| format!("tensor {}", info.name))?;
        let elements = info
            .dimensions
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .with_context(|| format!("tensor {} element count overflow", info.name))?;
        ensure!(
            elements % ty.block_size() == 0,
            "tensor {} element count is incompatible with {}",
            info.name,
            ty.name()
        );
        // Quantization blocks may not cross rows.
        ensure!(
            info.dimensions[0].is_multiple_of(ty.block_size()),
            "tensor {} row width {} is incompatible with {}",
            info.name,
            info.dimensions[0],
            ty.name()
        );
        let len = elements / ty.block_size() * ty.type_size();
        let start = self
            .data_offset
            .checked_add(info.offset)
            .with_context(|| format!("tensor {} offset overflow", info.name))?;
        ensure!(
            start <= self.mmap.len() && len <= self.mmap.len() - start,
            "tensor {} is outside the GGUF file",
            info.name
        );
        Ok((start, len))
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .context("GGUF alignment overflow")
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        ensure!(
            self.pos <= self.data.len() && len <= self.data.len() - self.pos,
            "truncated GGUF file at byte {}",
            self.pos
        );
        let value = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn string(&mut self) -> Result<String> {
        let len = usize::try_from(self.u64()?).context("GGUF string length does not fit usize")?;
        ensure!(len <= 1 << 30, "unreasonable GGUF string length {len}");
        let bytes = self.take(len)?;
        Ok(std::str::from_utf8(bytes)
            .context("invalid UTF-8 in GGUF string")?
            .to_owned())
    }

    fn metadata_value(&mut self, ty: u32) -> Result<MetadataValue> {
        match ty {
            0 => Ok(MetadataValue::U64(self.u8()? as u64)),
            1 => Ok(MetadataValue::I64(self.i8()? as i64)),
            2 => Ok(MetadataValue::U64(self.u16()? as u64)),
            3 => Ok(MetadataValue::I64(self.i16()? as i64)),
            4 => Ok(MetadataValue::U64(self.u32()? as u64)),
            5 => Ok(MetadataValue::I64(self.i32()? as i64)),
            6 => Ok(MetadataValue::F64(self.f32()? as f64)),
            7 => Ok(MetadataValue::Bool(self.u8()? != 0)),
            8 => Ok(MetadataValue::String(self.string()?)),
            9 => {
                let element_type = self.u32()?;
                ensure!(
                    element_type != 9,
                    "nested GGUF arrays are not supported by the format"
                );
                let len =
                    usize::try_from(self.u64()?).context("GGUF array length does not fit usize")?;
                ensure!(len <= 100_000_000, "unreasonable GGUF array length {len}");
                for _ in 0..len {
                    let _ = self.metadata_value(element_type)?;
                }
                Ok(MetadataValue::Array { element_type, len })
            }
            10 => Ok(MetadataValue::U64(self.u64()?)),
            11 => Ok(MetadataValue::I64(self.i64()?)),
            12 => Ok(MetadataValue::F64(self.f64()?)),
            _ => bail!("unknown GGUF metadata type {ty}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn put_string(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }

    #[test]
    fn parses_a_minimal_v3_file() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // tensors
        bytes.extend_from_slice(&1u64.to_le_bytes()); // metadata
        put_string(&mut bytes, "general.alignment");
        bytes.extend_from_slice(&4u32.to_le_bytes()); // UINT32
        bytes.extend_from_slice(&4u32.to_le_bytes());
        put_string(&mut bytes, "x");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes()); // columns
        bytes.extend_from_slice(&2u64.to_le_bytes()); // rows
        bytes.extend_from_slice(&0u32.to_le_bytes()); // F32
        bytes.extend_from_slice(&0u64.to_le_bytes());
        while bytes.len() % 4 != 0 {
            bytes.push(0);
        }
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let path = std::env::temp_dir().join(format!("llama-rs-gguf-{}.gguf", std::process::id()));
        let mut file = File::create(&path).unwrap();
        file.write_all(&bytes).unwrap();
        drop(file);

        {
            let gguf = GgufFile::open(&path).unwrap();
            assert_eq!(gguf.version, 3);
            let tensor = gguf.tensor("x").unwrap();
            assert_eq!(tensor.shape, [2, 2]);
            assert_eq!(tensor.to_vec_f32(), [1.0, 2.0, 3.0, 4.0]);
        }
        std::fs::remove_file(path).unwrap();
    }
}
