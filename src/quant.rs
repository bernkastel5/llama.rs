use crate::backend::{BackendConfig, CpuBackend, KernelBackend, KernelKind};
use crate::simd::{dot_row_avx2, dot_row_neon};
use anyhow::{bail, ensure, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use std::sync::{Arc, OnceLock};

pub(crate) const QK_K: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    /// F32. The name is retained for compatibility with the original prototype.
    None,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}

impl QuantType {
    pub fn from_ggml_type(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            30 => Ok(Self::BF16),
            other => bail!(
                "unsupported GGML tensor type {other}; supported: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K"
            ),
        }
    }

    pub fn block_size(self) -> usize {
        match self {
            Self::None | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => 32,
            Self::Q4K | Self::Q5K | Self::Q6K => QK_K,
        }
    }

    pub fn type_size(self) -> usize {
        match self {
            Self::None => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
        }
    }

    pub fn row_bytes(self, cols: usize) -> usize {
        cols / self.block_size() * self.type_size()
    }
}

#[derive(Debug, Clone)]
pub enum TensorData {
    Owned(Arc<[u8]>),
    Mapped {
        mmap: Arc<Mmap>,
        start: usize,
        len: usize,
    },
}

impl TensorData {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Owned(data) => data,
            Self::Mapped { mmap, start, len } => &mmap[*start..*start + *len],
        }
    }
}

/// A row-major matrix. `shape` is `[rows, columns]` (or `[length]` for a vector).
#[derive(Debug, Clone)]
pub struct QuantTensor {
    pub data: TensorData,
    pub shape: Vec<usize>,
    pub quant_type: QuantType,
}

impl QuantTensor {
    pub fn from_owned(data: Vec<u8>, shape: Vec<usize>, quant_type: QuantType) -> Result<Self> {
        Self::from_data(TensorData::Owned(data.into()), shape, quant_type)
    }

    pub fn from_mmap(
        mmap: Arc<Mmap>,
        start: usize,
        len: usize,
        shape: Vec<usize>,
        quant_type: QuantType,
    ) -> Result<Self> {
        ensure!(
            start <= mmap.len() && len <= mmap.len() - start,
            "tensor mmap range is out of bounds"
        );
        Self::from_data(TensorData::Mapped { mmap, start, len }, shape, quant_type)
    }

    fn from_data(data: TensorData, shape: Vec<usize>, quant_type: QuantType) -> Result<Self> {
        ensure!(
            !shape.is_empty() && shape.len() <= 2,
            "only 1D and 2D tensors are supported"
        );
        let cols = *shape.last().unwrap();
        ensure!(
            cols.is_multiple_of(quant_type.block_size()),
            "last dimension {cols} is not divisible by {} for {}",
            quant_type.block_size(),
            quant_type.name()
        );
        let elements: usize = shape.iter().product();
        let expected = elements / quant_type.block_size() * quant_type.type_size();
        ensure!(
            data.as_bytes().len() == expected,
            "bad {} tensor byte length: got {}, expected {expected}",
            quant_type.name(),
            data.as_bytes().len()
        );
        Ok(Self {
            data,
            shape,
            quant_type,
        })
    }

    pub fn rows(&self) -> usize {
        if self.shape.len() == 1 {
            1
        } else {
            self.shape[0]
        }
    }

    pub fn cols(&self) -> usize {
        *self.shape.last().unwrap()
    }

    pub fn row_bytes(&self) -> usize {
        self.cols() / self.quant_type.block_size() * self.quant_type.type_size()
    }

    pub fn from_f32(values: &[f32], shape: Vec<usize>) -> Result<Self> {
        ensure!(
            values.len() == shape.iter().product::<usize>(),
            "F32 tensor shape/data mismatch"
        );
        let mut data = Vec::with_capacity(values.len() * 4);
        for &value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        Self::from_owned(data, shape, QuantType::None)
    }

    /// Quantizes a 2D matrix to GGML-compatible Q4_K blocks. As in llama.cpp,
    /// rows not divisible by 256 fall back to Q5_0 (block size 32), and unusual
    /// rows not divisible by 32 stay F32.
    pub fn quantize_q4k(values: &[f32], rows: usize, cols: usize) -> Result<Self> {
        ensure!(
            values.len() == rows * cols,
            "quantization shape/data mismatch"
        );
        if cols.is_multiple_of(QK_K) {
            let mut data = Vec::with_capacity(rows * cols / QK_K * 144);
            for row in values.chunks_exact(cols) {
                for block in row.chunks_exact(QK_K) {
                    quantize_block_q4k(block, &mut data);
                }
            }
            Self::from_owned(data, vec![rows, cols], QuantType::Q4K)
        } else if cols.is_multiple_of(32) {
            Self::quantize_q5_0(values, rows, cols)
        } else {
            Self::from_f32(values, vec![rows, cols])
        }
    }

    pub fn quantize_q5_0(values: &[f32], rows: usize, cols: usize) -> Result<Self> {
        ensure!(
            values.len() == rows * cols,
            "quantization shape/data mismatch"
        );
        if !cols.is_multiple_of(32) {
            return Self::from_f32(values, vec![rows, cols]);
        }
        let mut data = Vec::with_capacity(rows * cols / 32 * 22);
        for row in values.chunks_exact(cols) {
            for block in row.chunks_exact(32) {
                quantize_block_q5_0(block, &mut data);
            }
        }
        Self::from_owned(data, vec![rows, cols], QuantType::Q5_0)
    }

    pub fn quantize_q8_0(values: &[f32], rows: usize, cols: usize) -> Result<Self> {
        ensure!(
            values.len() == rows * cols,
            "quantization shape/data mismatch"
        );
        if !cols.is_multiple_of(32) {
            return Self::from_f32(values, vec![rows, cols]);
        }
        let mut data = Vec::with_capacity(rows * cols / 32 * 34);
        for row in values.chunks_exact(cols) {
            for block in row.chunks_exact(32) {
                quantize_block_q8_0(block, &mut data);
            }
        }
        Self::from_owned(data, vec![rows, cols], QuantType::Q8_0)
    }

    pub fn quantize_q6k(values: &[f32], rows: usize, cols: usize) -> Result<Self> {
        ensure!(
            values.len() == rows * cols,
            "quantization shape/data mismatch"
        );
        if !cols.is_multiple_of(QK_K) {
            return Self::quantize_q8_0(values, rows, cols);
        }
        let mut data = Vec::with_capacity(rows * cols / QK_K * 210);
        for row in values.chunks_exact(cols) {
            for block in row.chunks_exact(QK_K) {
                quantize_block_q6k(block, &mut data);
            }
        }
        Self::from_owned(data, vec![rows, cols], QuantType::Q6K)
    }

    pub fn row_into(&self, row: usize, output: &mut [f32]) -> Result<()> {
        ensure!(row < self.rows(), "row {row} out of bounds");
        ensure!(
            output.len() == self.cols(),
            "embedding output has wrong length"
        );
        let row_data = self.row_data(row);
        for (col, value) in output.iter_mut().enumerate() {
            *value = value_at(row_data, self.quant_type, col);
        }
        Ok(())
    }

    pub fn to_vec_f32(&self) -> Vec<f32> {
        let mut output = vec![0.0; self.rows() * self.cols()];
        for row in 0..self.rows() {
            let row_data = self.row_data(row);
            for col in 0..self.cols() {
                output[row * self.cols() + col] = value_at(row_data, self.quant_type, col);
            }
        }
        output
    }

    /// Convenience entry point using an auto-detected CPU backend. Model inference
    /// keeps a persistent backend/thread-pool and calls `matvec_with` instead.
    /// Reference matrix-vector product, computed entirely in f32.
    ///
    /// This is the accuracy baseline, so it deliberately does not quantize the
    /// activation: callers use it to check other paths. Inference goes through
    /// [`QuantTensor::matvec_with`] with the engine's backend, which may trade
    /// ~1e-3 of relative error for a large speedup.
    pub fn matvec(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        static DEFAULT_BACKEND: OnceLock<CpuBackend> = OnceLock::new();
        DEFAULT_BACKEND
            .get_or_init(|| {
                CpuBackend::new(&BackendConfig::reference())
                    .expect("reference CPU backend must be constructible")
            })
            .matvec(self, input, output)
    }

    pub fn matvec_with(
        &self,
        backend: &dyn KernelBackend,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        backend.matvec(self, input, output)
    }

    pub fn matvec_quantized_with(
        &self,
        backend: &dyn KernelBackend,
        input: &[f32],
        quant_act: &crate::activation::QuantizedActivation,
        output: &mut [f32],
    ) -> Result<()> {
        backend.matvec_quantized(self, input, quant_act, output)
    }

    pub fn matvec_batch_with(
        &self,
        backend: &dyn KernelBackend,
        inputs: &[f32],
        outputs: &mut [f32],
        batch_size: usize,
    ) -> Result<()> {
        backend.matvec_batch(self, inputs, outputs, batch_size)
    }

    pub(crate) fn validate_matvec(&self, input: &[f32], output: &[f32]) -> Result<()> {
        ensure!(self.shape.len() == 2, "matvec requires a 2D tensor");
        ensure!(
            input.len() == self.cols(),
            "matvec input mismatch: got {}, expected {}",
            input.len(),
            self.cols()
        );
        ensure!(
            output.len() == self.rows(),
            "matvec output mismatch: got {}, expected {}",
            output.len(),
            self.rows()
        );
        Ok(())
    }

    pub(crate) fn row_data(&self, row: usize) -> &[u8] {
        let stride = self.row_bytes();
        &self.data.as_bytes()[row * stride..(row + 1) * stride]
    }
}

fn read_f16(data: &[u8], offset: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([data[offset], data[offset + 1]])).to_f32()
}

fn read_bf16(data: &[u8], offset: usize) -> f32 {
    bf16::from_bits(u16::from_le_bytes([data[offset], data[offset + 1]])).to_f32()
}

fn write_f16(value: f32, output: &mut Vec<u8>) {
    output.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
}

fn scale_min_k4(index: usize, scales: &[u8]) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        (
            (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

pub(crate) fn value_at(row: &[u8], ty: QuantType, col: usize) -> f32 {
    match ty {
        QuantType::None => {
            let o = col * 4;
            f32::from_le_bytes([row[o], row[o + 1], row[o + 2], row[o + 3]])
        }
        QuantType::F16 => read_f16(row, col * 2),
        QuantType::BF16 => read_bf16(row, col * 2),
        QuantType::Q4_0 => {
            let block = &row[col / 32 * 18..][..18];
            let j = col % 32;
            let q = if j < 16 {
                block[2 + j] & 0x0f
            } else {
                block[2 + j - 16] >> 4
            };
            read_f16(block, 0) * (q as f32 - 8.0)
        }
        QuantType::Q4_1 => {
            let block = &row[col / 32 * 20..][..20];
            let j = col % 32;
            let q = if j < 16 {
                block[4 + j] & 0x0f
            } else {
                block[4 + j - 16] >> 4
            };
            read_f16(block, 0) * q as f32 + read_f16(block, 2)
        }
        QuantType::Q5_0 | QuantType::Q5_1 => {
            let block_size = ty.type_size();
            let block = &row[col / 32 * block_size..][..block_size];
            let j = col % 32;
            let (qh_off, qs_off) = if ty == QuantType::Q5_0 {
                (2, 6)
            } else {
                (4, 8)
            };
            let low = if j < 16 {
                block[qs_off + j] & 0x0f
            } else {
                block[qs_off + j - 16] >> 4
            };
            let high = (block[qh_off + j / 8] >> (j % 8)) & 1;
            let q = low | (high << 4);
            if ty == QuantType::Q5_0 {
                read_f16(block, 0) * (q as f32 - 16.0)
            } else {
                read_f16(block, 0) * q as f32 + read_f16(block, 2)
            }
        }
        QuantType::Q8_0 => {
            let block = &row[col / 32 * 34..][..34];
            read_f16(block, 0) * (block[2 + col % 32] as i8 as f32)
        }
        QuantType::Q4K => {
            let block = &row[col / QK_K * 144..][..144];
            let j = col % QK_K;
            let group = j / 32;
            let (sc, min) = scale_min_k4(group, &block[4..16]);
            let q_group = group / 2;
            let q_index = j % 32;
            let packed = block[16 + q_group * 32 + q_index];
            let q = if group.is_multiple_of(2) {
                packed & 0x0f
            } else {
                packed >> 4
            };
            read_f16(block, 0) * sc as f32 * q as f32 - read_f16(block, 2) * min as f32
        }
        QuantType::Q5K => {
            let block = &row[col / QK_K * 176..][..176];
            let j = col % QK_K;
            let group = j / 32;
            let (sc, min) = scale_min_k4(group, &block[4..16]);
            let packed = block[48 + group / 2 * 32 + j % 32];
            let low = if group.is_multiple_of(2) {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let mask = 1u8 << group;
            let high = if block[16 + j % 32] & mask != 0 {
                16
            } else {
                0
            };
            read_f16(block, 0) * sc as f32 * (low + high) as f32 - read_f16(block, 2) * min as f32
        }
        QuantType::Q6K => {
            let block = &row[col / QK_K * 210..][..210];
            let j = col % QK_K;
            let half = j / 128;
            let within = j % 128;
            let lane = within % 32;
            let quarter = within / 32;
            let ql = match quarter {
                0 => block[half * 64 + lane] & 0x0f,
                1 => block[half * 64 + 32 + lane] & 0x0f,
                2 => block[half * 64 + lane] >> 4,
                _ => block[half * 64 + 32 + lane] >> 4,
            };
            let qh = (block[128 + half * 32 + lane] >> (quarter * 2)) & 3;
            let q = ((ql | (qh << 4)) as i16 - 32) as f32;
            let scale = block[192 + half * 8 + quarter * 2 + lane / 16] as i8 as f32;
            read_f16(block, 208) * scale * q
        }
    }
}

pub(crate) fn dot_row_scalar(row: &[u8], ty: QuantType, input: &[f32]) -> f32 {
    match ty {
        QuantType::None => row
            .chunks_exact(4)
            .zip(input)
            .map(|(b, &x)| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * x)
            .sum(),
        QuantType::F16 => row
            .chunks_exact(2)
            .zip(input)
            .map(|(b, &x)| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32() * x)
            .sum(),
        QuantType::BF16 => row
            .chunks_exact(2)
            .zip(input)
            .map(|(b, &x)| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32() * x)
            .sum(),
        QuantType::Q4_0 => {
            let mut sum = 0.0;
            for (bi, block) in row.chunks_exact(18).enumerate() {
                let d = read_f16(block, 0);
                let x = &input[bi * 32..][..32];
                for j in 0..16 {
                    let q = block[2 + j];
                    sum += d * ((q & 0x0f) as f32 - 8.0) * x[j];
                    sum += d * ((q >> 4) as f32 - 8.0) * x[j + 16];
                }
            }
            sum
        }
        QuantType::Q4_1 => {
            let mut sum = 0.0;
            for (bi, block) in row.chunks_exact(20).enumerate() {
                let d = read_f16(block, 0);
                let m = read_f16(block, 2);
                let x = &input[bi * 32..][..32];
                for j in 0..16 {
                    let q = block[4 + j];
                    sum += (d * (q & 0x0f) as f32 + m) * x[j];
                    sum += (d * (q >> 4) as f32 + m) * x[j + 16];
                }
            }
            sum
        }
        QuantType::Q5_0 | QuantType::Q5_1 => {
            let mut sum = 0.0;
            let size = ty.type_size();
            for (bi, block) in row.chunks_exact(size).enumerate() {
                let d = read_f16(block, 0);
                let m = if ty == QuantType::Q5_1 {
                    read_f16(block, 2)
                } else {
                    0.0
                };
                let (qh_off, qs_off) = if ty == QuantType::Q5_0 {
                    (2, 6)
                } else {
                    (4, 8)
                };
                let qh = u32::from_le_bytes([
                    block[qh_off],
                    block[qh_off + 1],
                    block[qh_off + 2],
                    block[qh_off + 3],
                ]);
                let x = &input[bi * 32..][..32];
                for j in 0..16 {
                    let packed = block[qs_off + j];
                    let q0 = (packed & 0x0f) as u32 | (((qh >> j) & 1) << 4);
                    let q1 = (packed >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4);
                    if ty == QuantType::Q5_0 {
                        sum += d * (q0 as f32 - 16.0) * x[j];
                        sum += d * (q1 as f32 - 16.0) * x[j + 16];
                    } else {
                        sum += (d * q0 as f32 + m) * x[j];
                        sum += (d * q1 as f32 + m) * x[j + 16];
                    }
                }
            }
            sum
        }
        QuantType::Q8_0 => {
            let mut sum = 0.0;
            for (bi, block) in row.chunks_exact(34).enumerate() {
                let d = read_f16(block, 0);
                let x = &input[bi * 32..][..32];
                for j in 0..32 {
                    sum += d * block[2 + j] as i8 as f32 * x[j];
                }
            }
            sum
        }
        QuantType::Q4K => dot_q4k(row, input),
        QuantType::Q5K => dot_q5k(row, input),
        QuantType::Q6K => dot_q6k(row, input),
    }
}

pub(crate) fn dot_row_for_kernel(
    kernel: KernelKind,
    row: &[u8],
    ty: QuantType,
    input: &[f32],
) -> f32 {
    match kernel {
        KernelKind::Scalar => dot_row_scalar(row, ty, input),
        KernelKind::Avx2 => dot_row_avx2(row, ty, input),
        KernelKind::Neon => dot_row_neon(row, ty, input),
    }
}

fn dot_q4k(row: &[u8], input: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (bi, block) in row.chunks_exact(144).enumerate() {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..16];
        let qs = &block[16..144];
        let x = &input[bi * QK_K..][..QK_K];
        for group in 0..8 {
            let (sc, min) = scale_min_k4(group, scales);
            let ds = d * sc as f32;
            let dm = dmin * min as f32;
            for j in 0..32 {
                let packed = qs[group / 2 * 32 + j];
                let q = if group.is_multiple_of(2) {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                sum += (ds * q as f32 - dm) * x[group * 32 + j];
            }
        }
    }
    sum
}

fn dot_q5k(row: &[u8], input: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (bi, block) in row.chunks_exact(176).enumerate() {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let x = &input[bi * QK_K..][..QK_K];
        for group in 0..8 {
            let (sc, min) = scale_min_k4(group, scales);
            let ds = d * sc as f32;
            let dm = dmin * min as f32;
            let mask = 1u8 << group;
            for j in 0..32 {
                let packed = qs[group / 2 * 32 + j];
                let low = if group.is_multiple_of(2) {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                let q = low + if qh[j] & mask != 0 { 16 } else { 0 };
                sum += (ds * q as f32 - dm) * x[group * 32 + j];
            }
        }
    }
    sum
}

fn dot_q6k(row: &[u8], input: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (bi, block) in row.chunks_exact(210).enumerate() {
        let d = read_f16(block, 208);
        let x = &input[bi * QK_K..][..QK_K];
        for half in 0..2 {
            let ql = &block[half * 64..half * 64 + 64];
            let qh = &block[128 + half * 32..128 + half * 32 + 32];
            let sc = &block[192 + half * 8..192 + half * 8 + 8];
            let base = half * 128;
            for lane in 0..32 {
                let si = lane / 16;
                let q1 = ((ql[lane] & 0x0f) | ((qh[lane] & 3) << 4)) as i16 - 32;
                let q2 = ((ql[32 + lane] & 0x0f) | (((qh[lane] >> 2) & 3) << 4)) as i16 - 32;
                let q3 = ((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) as i16 - 32;
                let q4 = ((ql[32 + lane] >> 4) | (((qh[lane] >> 6) & 3) << 4)) as i16 - 32;
                sum += d * sc[si] as i8 as f32 * q1 as f32 * x[base + lane];
                sum += d * sc[si + 2] as i8 as f32 * q2 as f32 * x[base + 32 + lane];
                sum += d * sc[si + 4] as i8 as f32 * q3 as f32 * x[base + 64 + lane];
                sum += d * sc[si + 6] as i8 as f32 * q4 as f32 * x[base + 96 + lane];
            }
        }
    }
    sum
}

fn quantize_block_q8_0(values: &[f32], output: &mut Vec<u8>) {
    debug_assert_eq!(values.len(), 32);
    let abs_max = values
        .iter()
        .fold(0.0f32, |max, value| max.max(value.abs()));
    let d = abs_max / 127.0;
    let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
    write_f16(d, output);
    for &value in values {
        let q = (value * inv).round().clamp(-127.0, 127.0) as i8;
        output.push(q as u8);
    }
}

fn quantize_block_q6k(values: &[f32], output: &mut Vec<u8>) {
    debug_assert_eq!(values.len(), QK_K);
    let mut codes = [0u8; QK_K];
    let mut scales = [0.0f32; 16];
    let mut max_scale = 0.0f32;
    let mut max_abs_scale = 0.0f32;

    for group in 0..16 {
        let scale = make_qx_quants_16(
            &values[group * 16..group * 16 + 16],
            &mut codes[group * 16..group * 16 + 16],
        );
        scales[group] = scale;
        if scale.abs() > max_abs_scale {
            max_abs_scale = scale.abs();
            max_scale = scale;
        }
    }
    if max_abs_scale < 1e-15 {
        output.extend_from_slice(&[0; 210]);
        return;
    }

    let inv_scale = -128.0 / max_scale;
    let d = 1.0 / inv_scale;
    let d_f16 = f16::from_f32(d).to_f32();
    let mut scale_codes = [0i8; 16];
    for group in 0..16 {
        scale_codes[group] = (inv_scale * scales[group]).round().min(127.0) as i8;
        let group_d = d_f16 * scale_codes[group] as f32;
        if group_d == 0.0 {
            continue;
        }
        for i in 0..16 {
            let level = (values[group * 16 + i] / group_d)
                .round()
                .clamp(-32.0, 31.0) as i32;
            codes[group * 16 + i] = (level + 32) as u8;
        }
    }

    let mut low = [0u8; 128];
    let mut high = [0u8; 64];
    for half in 0..2 {
        let base = half * 128;
        for lane in 0..32 {
            let q1 = codes[base + lane];
            let q2 = codes[base + 32 + lane];
            let q3 = codes[base + 64 + lane];
            let q4 = codes[base + 96 + lane];
            low[half * 64 + lane] = (q1 & 0x0f) | ((q3 & 0x0f) << 4);
            low[half * 64 + 32 + lane] = (q2 & 0x0f) | ((q4 & 0x0f) << 4);
            high[half * 32 + lane] =
                (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
        }
    }
    output.extend_from_slice(&low);
    output.extend_from_slice(&high);
    output.extend(scale_codes.iter().map(|v| *v as u8));
    write_f16(d, output);
}

/// Port of llama.cpp `make_qx_quants(..., n=16, nmax=32, rmse_type=1)`.
fn make_qx_quants_16(values: &[f32], codes: &mut [u8]) -> f32 {
    let mut max = 0.0f32;
    let mut abs_max = 0.0f32;
    for &value in values {
        if value.abs() > abs_max {
            abs_max = value.abs();
            max = value;
        }
    }
    if abs_max < 1e-15 {
        codes.fill(0);
        return 0.0;
    }

    let mut inv_scale = -32.0 / max;
    let mut sum_lx = 0.0;
    let mut sum_l2 = 0.0;
    for i in 0..16 {
        let level = (inv_scale * values[i]).round().clamp(-32.0, 31.0) as i32;
        codes[i] = (level + 32) as u8;
        let weight = values[i] * values[i];
        sum_lx += weight * values[i] * level as f32;
        sum_l2 += weight * (level * level) as f32;
    }
    let mut scale = if sum_l2 > 0.0 { sum_lx / sum_l2 } else { 0.0 };
    let mut best = scale * sum_lx;

    for step in -9..=9 {
        if step == 0 {
            continue;
        }
        inv_scale = -(32.0 + 0.1 * step as f32) / max;
        sum_lx = 0.0;
        sum_l2 = 0.0;
        for &value in values {
            let level = (inv_scale * value).round().clamp(-32.0, 31.0) as i32;
            let weight = value * value;
            sum_lx += weight * value * level as f32;
            sum_l2 += weight * (level * level) as f32;
        }
        if sum_l2 > 0.0 && sum_lx * sum_lx > best * sum_l2 {
            for i in 0..16 {
                let level = (inv_scale * values[i]).round().clamp(-32.0, 31.0) as i32;
                codes[i] = (level + 32) as u8;
            }
            scale = sum_lx / sum_l2;
            best = scale * sum_lx;
        }
    }
    scale
}

fn quantize_block_q5_0(values: &[f32], output: &mut Vec<u8>) {
    debug_assert_eq!(values.len(), 32);
    let mut max_value = 0.0f32;
    let mut abs_max = 0.0f32;
    for &value in values {
        if value.abs() > abs_max {
            abs_max = value.abs();
            max_value = value;
        }
    }
    let d = if abs_max == 0.0 {
        0.0
    } else {
        max_value / -16.0
    };
    let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
    write_f16(d, output);
    let qh_pos = output.len();
    output.extend_from_slice(&[0; 4]);
    let mut high = 0u32;
    for j in 0..16 {
        let q0 = (values[j] * inv + 16.5).floor().clamp(0.0, 31.0) as u8;
        let q1 = (values[j + 16] * inv + 16.5).floor().clamp(0.0, 31.0) as u8;
        output.push((q0 & 0x0f) | ((q1 & 0x0f) << 4));
        high |= ((q0 as u32 >> 4) & 1) << j;
        high |= ((q1 as u32 >> 4) & 1) << (j + 16);
    }
    output[qh_pos..qh_pos + 4].copy_from_slice(&high.to_le_bytes());
}

/// Deterministic port of llama.cpp's reference Q4_K encoder. It emits the exact
/// GGML block layout and applies the same weighted least-squares sub-block fitting.
fn quantize_block_q4k(values: &[f32], output: &mut Vec<u8>) {
    debug_assert_eq!(values.len(), QK_K);
    let mut local_scales = [0.0f32; 8];
    let mut local_mins = [0.0f32; 8];
    let mut max_scale = 0.0f32;
    let mut max_min = 0.0f32;
    let mut codes = [0u8; QK_K];
    let mut aux = [0u8; 32];

    for group in 0..8 {
        let group_values = &values[group * 32..group * 32 + 32];
        let sum_sq: f32 = group_values.iter().map(|v| v * v).sum();
        let average = (sum_sq / 32.0).sqrt();
        let mut weights = [0.0f32; 32];
        for i in 0..32 {
            weights[i] = average + group_values[i].abs();
        }
        let mut min = 0.0;
        let scale = make_qkx2_quants(
            group_values,
            &weights,
            &mut codes[group * 32..group * 32 + 32],
            &mut min,
            &mut aux,
        );
        local_scales[group] = scale;
        local_mins[group] = min;
        max_scale = max_scale.max(scale);
        max_min = max_min.max(min);
    }

    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let mut scales = [0u8; 12];
    let mut sc_codes = [0u8; 8];
    let mut min_codes = [0u8; 8];
    for group in 0..8 {
        sc_codes[group] = (inv_scale * local_scales[group]).round().clamp(0.0, 63.0) as u8;
        min_codes[group] = (inv_min * local_mins[group]).round().clamp(0.0, 63.0) as u8;
        if group < 4 {
            scales[group] = sc_codes[group];
            scales[group + 4] = min_codes[group];
        } else {
            scales[group + 4] = (sc_codes[group] & 0x0f) | ((min_codes[group] & 0x0f) << 4);
            scales[group - 4] |= (sc_codes[group] >> 4) << 6;
            scales[group] |= (min_codes[group] >> 4) << 6;
        }
    }

    let d = max_scale / 63.0;
    let dmin = max_min / 63.0;
    write_f16(d, output);
    write_f16(dmin, output);
    output.extend_from_slice(&scales);
    let d = f16::from_f32(d).to_f32();
    let dmin = f16::from_f32(dmin).to_f32();

    // Recompute final codes using the scales after six-bit and FP16 rounding.
    for group in 0..8 {
        let scale = d * sc_codes[group] as f32;
        if scale == 0.0 {
            continue;
        }
        let min = dmin * min_codes[group] as f32;
        for i in 0..32 {
            codes[group * 32 + i] = ((values[group * 32 + i] + min) / scale)
                .round()
                .clamp(0.0, 15.0) as u8;
        }
    }
    for pair in 0..4 {
        for i in 0..32 {
            output.push(codes[pair * 64 + i] | (codes[pair * 64 + 32 + i] << 4));
        }
    }
}

/// Port of llama.cpp `make_qkx2_quants` specialized for Q4_K's 32-value,
/// 15-level sub-blocks and weighted squared-error objective.
fn make_qkx2_quants(
    values: &[f32],
    weights: &[f32],
    codes: &mut [u8],
    output_min: &mut f32,
    aux: &mut [u8],
) -> f32 {
    debug_assert_eq!(values.len(), 32);
    debug_assert_eq!(weights.len(), 32);
    let mut min = values[0];
    let mut max = values[0];
    let mut sum_weight = weights[0];
    let mut sum_value = weights[0] * values[0];
    for i in 1..32 {
        min = min.min(values[i]);
        max = max.max(values[i]);
        sum_weight += weights[i];
        sum_value += weights[i] * values[i];
    }
    min = min.min(0.0);
    if max == min {
        codes.fill(0);
        *output_min = -min;
        return 0.0;
    }

    let range = max - min;
    let mut inv_scale = 15.0 / range;
    let mut scale = 1.0 / inv_scale;
    let mut best_error = 0.0;
    for i in 0..32 {
        let code = (inv_scale * (values[i] - min)).round().clamp(0.0, 15.0) as u8;
        codes[i] = code;
        let diff = scale * code as f32 + min - values[i];
        best_error += weights[i] * diff * diff;
    }

    for step in 0..=20 {
        inv_scale = (-1.0 + 0.1 * step as f32 + 15.0) / range;
        let mut sum_l = 0.0;
        let mut sum_l2 = 0.0;
        let mut sum_xl = 0.0;
        for i in 0..32 {
            let code = (inv_scale * (values[i] - min)).round().clamp(0.0, 15.0) as u8;
            aux[i] = code;
            let weighted_code = weights[i] * code as f32;
            sum_l += weighted_code;
            sum_l2 += weighted_code * code as f32;
            sum_xl += weighted_code * values[i];
        }
        let determinant = sum_weight * sum_l2 - sum_l * sum_l;
        if determinant > 0.0 {
            let mut candidate_scale = (sum_weight * sum_xl - sum_value * sum_l) / determinant;
            let mut candidate_min = (sum_l2 * sum_value - sum_l * sum_xl) / determinant;
            if candidate_min > 0.0 {
                candidate_min = 0.0;
                candidate_scale = sum_xl / sum_l2;
            }
            let mut error = 0.0;
            for i in 0..32 {
                let diff = candidate_scale * aux[i] as f32 + candidate_min - values[i];
                error += weights[i] * diff * diff;
            }
            if error < best_error {
                codes.copy_from_slice(&aux[..32]);
                best_error = error;
                scale = candidate_scale;
                min = candidate_min;
            }
        }
    }
    *output_min = -min;
    scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4k_roundtrip_and_dot_are_consistent() {
        let values: Vec<f32> = (0..512)
            .map(|i| ((i as f32 * 0.17).sin() + (i as f32 * 0.03).cos()) * 0.3)
            .collect();
        let tensor = QuantTensor::quantize_q4k(&values, 2, 256).unwrap();
        assert_eq!(tensor.quant_type, QuantType::Q4K);
        let mut row = vec![0.0; 256];
        tensor.row_into(1, &mut row).unwrap();
        let mae: f32 = row
            .iter()
            .zip(&values[256..])
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / 256.0;
        assert!(mae < 0.06, "mae={mae}");

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.11).cos()).collect();
        let expected: f32 = row.iter().zip(&input).map(|(a, b)| a * b).sum();
        let mut output = [0.0; 2];
        tensor.matvec(&input, &mut output).unwrap();
        assert!((output[1] - expected).abs() < 1e-4);
    }

    #[test]
    fn q4k_shape_falls_back_like_llama_cpp() {
        let tensor = QuantTensor::quantize_q4k(&vec![0.0; 896], 1, 896).unwrap();
        assert_eq!(tensor.quant_type, QuantType::Q5_0);
    }

    #[test]
    fn avx2_quant_kernels_match_scalar() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.037).sin()).collect();
            let values: Vec<f32> = (0..256)
                .map(|i| ((i as f32 * 0.071).sin() - (i as f32 * 0.19).cos()) * 0.2)
                .collect();
            let q4 = QuantTensor::quantize_q4k(&values, 1, 256).unwrap();
            let q6 = QuantTensor::quantize_q6k(&values, 1, 256).unwrap();
            let q8 = QuantTensor::quantize_q8_0(&values, 1, 256).unwrap();
            let q5_0 = QuantTensor::quantize_q5_0(&values, 1, 256).unwrap();

            // Deterministic, structurally valid Q5_K block.
            let mut q5k_data = vec![0u8; 176];
            q5k_data[0..2].copy_from_slice(&f16::from_f32(0.003).to_bits().to_le_bytes());
            q5k_data[2..4].copy_from_slice(&f16::from_f32(0.001).to_bits().to_le_bytes());
            for (i, byte) in q5k_data[4..].iter_mut().enumerate() {
                *byte = (i as u8).wrapping_mul(73).wrapping_add(19);
            }
            let q5k = QuantTensor::from_owned(q5k_data, vec![1, 256], QuantType::Q5K).unwrap();

            for tensor in [&q4, &q5_0, &q5k, &q6, &q8] {
                let row = tensor.row_data(0);
                let scalar = dot_row_scalar(row, tensor.quant_type, &input);
                let simd = dot_row_avx2(row, tensor.quant_type, &input);
                let tolerance = 2e-4 * scalar.abs().max(1.0);
                assert!(
                    (scalar - simd).abs() <= tolerance,
                    "{} scalar={scalar}, simd={simd}",
                    tensor.quant_type.name()
                );
            }
        }
    }

    #[test]
    fn q8k_activation_kernel_matches_scalar() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            use crate::activation::quantize_activation_q8k;
            use crate::simd::dot_row_avx2_q8k;

            // 512 columns exercises the multi-super-block path.
            let values: Vec<f32> = (0..512)
                .map(|i| ((i as f32 * 0.041).sin() + (i as f32 * 0.011).cos()) * 0.35)
                .collect();
            let q4 = QuantTensor::quantize_q4k(&values, 1, 512).unwrap();
            let q6 = QuantTensor::quantize_q6k(&values, 1, 512).unwrap();

            // Deterministic, structurally valid Q5_K block (512 elements = 2 blocks = 352 bytes).
            let mut q5k_data = vec![0u8; 352];
            for block in 0..2 {
                let off = block * 176;
                q5k_data[off..off + 2].copy_from_slice(&f16::from_f32(0.003).to_bits().to_le_bytes());
                q5k_data[off + 2..off + 4].copy_from_slice(&f16::from_f32(0.001).to_bits().to_le_bytes());
                for (i, byte) in q5k_data[off + 4..off + 176].iter_mut().enumerate() {
                    *byte = (i as u8).wrapping_mul(73).wrapping_add(19 + block as u8);
                }
            }
            let q5k = QuantTensor::from_owned(q5k_data, vec![1, 512], QuantType::Q5K).unwrap();

            for tensor in [&q4, &q5k, &q6] {
                for scale in [1.0f32, 0.001, 40.0] {
                    let input: Vec<f32> = (0..512)
                        .map(|i| (i as f32 * 0.023).cos() * scale)
                        .collect();
                    let mut blocks = Vec::new();
                    quantize_activation_q8k(&input, &mut blocks);

                    let row = tensor.row_data(0);
                    let reference = dot_row_scalar(row, tensor.quant_type, &input);
                    let integer = dot_row_avx2_q8k(row, tensor.quant_type, &blocks)
                        .expect("integer kernel must exist on AVX2");

                    // Q8_K activations carry ~7 bits of mantissa, so this compares
                    // against the f32 reference with a quantization-sized budget
                    // rather than a floating-point epsilon.
                    let magnitude: f32 = input.iter().map(|v| v.abs()).sum::<f32>() * 0.02;
                    let tolerance = magnitude.max(2e-3 * reference.abs().max(1.0));
                    assert!(
                        (reference - integer).abs() <= tolerance,
                        "{} scale={scale}: scalar={reference}, int8={integer}",
                        tensor.quant_type.name()
                    );
                }
            }
        }
    }

    /// `QuantTensor::matvec` is the accuracy baseline and must stay exact.
    ///
    /// The integer path is a large speedup but costs ~1e-3 relative error, so
    /// enabling it by default here would silently change the meaning of the
    /// public API. This test is what catches that.
    #[test]
    fn reference_matvec_stays_exact_for_every_quant_type() {
        let values: Vec<f32> = (0..512)
            .map(|i| ((i as f32 * 0.17).sin() + (i as f32 * 0.03).cos()) * 0.3)
            .collect();
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.11).cos()).collect();

        let tensors = [
            QuantTensor::quantize_q4k(&values, 2, 256).unwrap(),
            QuantTensor::quantize_q6k(&values, 2, 256).unwrap(),
            QuantTensor::quantize_q8_0(&values, 2, 256).unwrap(),
            QuantTensor::quantize_q5_0(&values, 2, 256).unwrap(),
        ];

        for tensor in &tensors {
            let mut row = vec![0.0; 256];
            tensor.row_into(1, &mut row).unwrap();
            let expected: f32 = row.iter().zip(&input).map(|(a, b)| a * b).sum();

            let mut output = [0.0; 2];
            tensor.matvec(&input, &mut output).unwrap();
            assert!(
                (output[1] - expected).abs() < 1e-4,
                "{}: reference matvec drifted, got {} want {}",
                tensor.quant_type.name(),
                output[1],
                expected
            );
        }
    }

    /// The integer path must be close to the reference, but is not expected to
    /// be exact. This pins the size of the error so a real bug cannot hide
    /// behind "8-bit is approximate".
    #[test]
    fn integer_activation_backend_tracks_the_reference() {
        let values: Vec<f32> = (0..512)
            .map(|i| ((i as f32 * 0.17).sin() + (i as f32 * 0.03).cos()) * 0.3)
            .collect();
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.11).cos()).collect();

        let fast = CpuBackend::new(&BackendConfig::default()).unwrap();

        for tensor in [
            QuantTensor::quantize_q4k(&values, 2, 256).unwrap(),
            QuantTensor::quantize_q6k(&values, 2, 256).unwrap(),
            QuantTensor::quantize_q8_0(&values, 2, 256).unwrap(),
            QuantTensor::quantize_q5_0(&values, 2, 256).unwrap(),
        ] {
            let mut reference = [0.0; 2];
            tensor.matvec(&input, &mut reference).unwrap();

            let mut actual = [0.0; 2];
            tensor.matvec_with(&fast, &input, &mut actual).unwrap();

            for row in 0..2 {
                let scale = reference[row].abs().max(1.0);
                assert!(
                    (actual[row] - reference[row]).abs() < 5e-2 * scale,
                    "{} row {row}: integer={} reference={}",
                    tensor.quant_type.name(),
                    actual[row],
                    reference[row]
                );
            }
        }
    }

    #[test]
    fn reference_config_disables_the_integer_path() {
        let backend = CpuBackend::new(&BackendConfig::reference()).unwrap();
        let values: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.09).cos()).collect();
        let tensor = QuantTensor::quantize_q8_0(&values, 1, 256).unwrap();

        let mut row = vec![0.0; 256];
        tensor.row_into(0, &mut row).unwrap();
        let expected: f32 = row.iter().zip(&input).map(|(a, b)| a * b).sum();

        let mut output = [0.0; 1];
        tensor.matvec_with(&backend, &input, &mut output).unwrap();
        assert!((output[0] - expected).abs() < 1e-4);
    }

    #[test]
    fn q8_32_kernels_match_scalar() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            use crate::activation::quantize_activation_q8_32;
            use crate::simd::dot_row_avx2_q8_32;

            // 896 columns is the real Qwen2.5-0.5B projection width, and it is
            // deliberately not a multiple of QK_K.
            let cols = 896;
            let values: Vec<f32> = (0..cols)
                .map(|i| ((i as f32 * 0.053).sin() + (i as f32 * 0.017).cos()) * 0.4)
                .collect();

            let tensors = [
                QuantTensor::quantize_q8_0(&values, 1, cols).unwrap(),
                QuantTensor::quantize_q5_0(&values, 1, cols).unwrap(),
            ];

            for tensor in &tensors {
                for scale in [1.0f32, 0.002, 25.0] {
                    let input: Vec<f32> = (0..cols)
                        .map(|i| (i as f32 * 0.031).sin() * scale)
                        .collect();
                    let mut blocks = Vec::new();
                    quantize_activation_q8_32(&input, &mut blocks);

                    let row = tensor.row_data(0);
                    let reference = dot_row_scalar(row, tensor.quant_type, &input);
                    let integer = dot_row_avx2_q8_32(row, tensor.quant_type, &blocks)
                        .expect("integer kernel must exist for this type");

                    let budget: f32 = input.iter().map(|v| v.abs()).sum::<f32>() * 0.02;
                    let tolerance = budget.max(2e-3 * reference.abs().max(1.0));
                    assert!(
                        (reference - integer).abs() <= tolerance,
                        "{} scale={scale}: scalar={reference}, int8={integer}",
                        tensor.quant_type.name()
                    );
                }
            }
        }
    }

    #[test]
    fn integer_kernels_are_absent_for_unsupported_types() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            use crate::activation::quantize_activation_q8_32;
            use crate::simd::dot_row_avx2_q8_32;

            // Q4_1 has no integer kernel yet; it must report absence so the
            // caller falls back rather than producing a wrong number.
            let values: Vec<f32> = (0..256).map(|i| (i as f32 * 0.03).sin()).collect();
            let tensor = QuantTensor::quantize_q6k(&values, 1, 256).unwrap();
            let mut blocks = Vec::new();
            quantize_activation_q8_32(&values, &mut blocks);
            assert!(dot_row_avx2_q8_32(tensor.row_data(0), QuantType::Q4_1, &blocks).is_none());
        }
    }

    #[test]
    fn q8k_zero_activation_is_zero() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            use crate::activation::quantize_activation_q8k;
            use crate::simd::dot_row_avx2_q8k;

            let values: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();
            let tensor = QuantTensor::quantize_q4k(&values, 1, 256).unwrap();
            let mut blocks = Vec::new();
            quantize_activation_q8k(&vec![0.0; 256], &mut blocks);
            let result =
                dot_row_avx2_q8k(tensor.row_data(0), QuantType::Q4K, &blocks).unwrap();
            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn q6k_and_q8_roundtrip() {
        let values: Vec<f32> = (0..512)
            .map(|i| ((i as f32 * 0.071).sin() - (i as f32 * 0.19).cos()) * 0.2)
            .collect();
        for (tensor, max_mae) in [
            (QuantTensor::quantize_q6k(&values, 2, 256).unwrap(), 0.008),
            (QuantTensor::quantize_q8_0(&values, 2, 256).unwrap(), 0.003),
        ] {
            let decoded = tensor.to_vec_f32();
            let mae = decoded
                .iter()
                .zip(&values)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / values.len() as f32;
            assert!(mae < max_mae, "{} mae={mae}", tensor.quant_type.name());

            let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.13).sin()).collect();
            let expected: f32 = decoded[..256].iter().zip(&input).map(|(a, b)| a * b).sum();
            let mut output = [0.0; 2];
            tensor.matvec(&input, &mut output).unwrap();
            assert!((output[0] - expected).abs() < 1e-4);
        }
    }
}
