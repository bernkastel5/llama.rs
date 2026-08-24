//! Quantized activations for integer matvec kernels.
//!
//! The kernels in `simd.rs` historically multiplied a quantized weight by an
//! `f32` activation, which forced every weight code back into a float lane:
//! eight values per AVX2 register and a horizontal reduction per 32-value
//! group. Quantizing the activation once per matvec lets the same registers
//! carry 32 int8 lanes and defers the reduction to one per row.
//!
//! This module owns only the activation side. Weight storage, load-time
//! quantization and the `--quantize` choice are untouched: a Q4_K tensor is
//! still read exactly as it sits in the GGUF file.

use crate::quant::QuantType;

/// Super-block length shared by the K-quants and by `BlockQ8K`.
///
/// Re-exported from `quant` so the block geometry has exactly one definition;
/// a second `256` here could silently drift from the weight side.
pub const QK_K: usize = crate::quant::QK_K;

/// One super-block of activations quantized to 8 bits.
///
/// `bsums` holds per-16-lane sums of `qs`. K-quant weights carry a per-group
/// minimum that is subtracted from every code, so the kernel needs
/// `sum(activation)` per group; precomputing it here keeps that term out of
/// the row loop, where it would otherwise be recomputed for every row.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ8K {
    /// Scale such that `value ≈ d * qs[i]`.
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}

impl Default for BlockQ8K {
    fn default() -> Self {
        Self {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        }
    }
}

/// Quantize `input` into Q8_K super-blocks, reusing `out`'s allocation.
///
/// `input.len()` must be a multiple of [`QK_K`]; callers gate on
/// [`supports_q8k_activations`] and on the tensor's column count, both of which
/// guarantee that. Any trailing remainder is ignored rather than silently
/// mis-scaled, so a caller that violates the contract loses accuracy instead of
/// reading out of bounds.
pub fn quantize_activation_q8k(input: &[f32], out: &mut Vec<BlockQ8K>) {
    out.clear();
    out.reserve(input.len() / QK_K);
    for chunk in input.chunks_exact(QK_K) {
        let mut block = BlockQ8K::default();
        let amax = chunk.iter().fold(0.0f32, |max, v| max.max(v.abs()));
        let d = amax / 127.0;
        let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
        block.d = d;
        for (slot, &value) in block.qs.iter_mut().zip(chunk) {
            *slot = (value * inv).round().clamp(-127.0, 127.0) as i8;
        }
        for (group, sum) in block.bsums.iter_mut().enumerate() {
            // At most 16 * 127 = 2032, so the i16 accumulator cannot overflow.
            let mut total = 0i32;
            for &q in &block.qs[group * 16..group * 16 + 16] {
                total += q as i32;
            }
            *sum = total as i16;
        }
        out.push(block);
    }
}

/// One 32-value block of activations quantized to 8 bits.
///
/// Q4_0/Q5_0/Q8_0 use 32-value blocks with one scale each, so they need a
/// matching activation layout rather than the 256-value [`BlockQ8K`].
/// `sum` is `sum(qs)`: Q4_0 and Q5_0 subtract a constant (8 and 16) from every
/// code, and that constant factors out into a single multiply per block.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
#[allow(non_camel_case_types)] // matches the Q8_32 format name
pub struct BlockQ8_32 {
    pub d: f32,
    pub qs: [i8; 32],
    pub sum: i32,
}

/// Quantize `input` into 32-value Q8 blocks, reusing `out`'s allocation.
pub fn quantize_activation_q8_32(input: &[f32], out: &mut Vec<BlockQ8_32>) {
    out.clear();
    out.reserve(input.len() / 32);
    for chunk in input.chunks_exact(32) {
        let mut block = BlockQ8_32::default();
        let amax = chunk.iter().fold(0.0f32, |max, v| max.max(v.abs()));
        let d = amax / 127.0;
        let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
        block.d = d;
        let mut sum = 0i32;
        for (slot, &value) in block.qs.iter_mut().zip(chunk) {
            *slot = (value * inv).round().clamp(-127.0, 127.0) as i8;
            sum += *slot as i32;
        }
        block.sum = sum;
        out.push(block);
    }
}

/// Activation layout an integer kernel expects for a given weight type.
///
/// This is the single place that decides whether a format has an integer path.
/// Adding a format means adding a kernel and one arm here; the backend, the
/// architecture layer and the engine are untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationLayout {
    /// No integer kernel; use the existing f32 path.
    None,
    /// 256-value super-blocks, for the K-quants.
    Q8K,
    /// 32-value blocks, for the legacy quants.
    Q8_32,
}

pub fn activation_layout(ty: QuantType) -> ActivationLayout {
    match ty {
        QuantType::Q4K => ActivationLayout::Q8K,
        QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 => ActivationLayout::Q8_32,
        // Q4_1/Q5_1 carry a per-block offset, and Q5_K/Q6_K need their own
        // kernels; all still take the f32 path.
        _ => ActivationLayout::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8k_activation_roundtrip_is_accurate() {
        let input: Vec<f32> = (0..QK_K).map(|i| (i as f32 * 0.037).sin() * 2.5).collect();
        let mut blocks = Vec::new();
        quantize_activation_q8k(&input, &mut blocks);
        assert_eq!(blocks.len(), 1);

        let block = &blocks[0];
        for (i, &original) in input.iter().enumerate() {
            let restored = block.d * block.qs[i] as f32;
            assert!(
                (restored - original).abs() <= block.d,
                "index {i}: {restored} vs {original}"
            );
        }

        for (group, &sum) in block.bsums.iter().enumerate() {
            let expected: i32 = block.qs[group * 16..group * 16 + 16]
                .iter()
                .map(|&q| q as i32)
                .sum();
            assert_eq!(sum as i32, expected, "bsums[{group}]");
        }
    }

    #[test]
    fn zero_activation_block_is_well_defined() {
        let mut blocks = Vec::new();
        quantize_activation_q8k(&vec![0.0; QK_K], &mut blocks);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].d, 0.0);
        assert!(blocks[0].qs.iter().all(|&q| q == 0));
        assert!(blocks[0].bsums.iter().all(|&s| s == 0));
    }

    #[test]
    fn partial_trailing_block_is_not_emitted() {
        let mut blocks = Vec::new();
        quantize_activation_q8k(&vec![1.0; QK_K + 5], &mut blocks);
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn q8_32_roundtrip_and_sum_are_correct() {
        let input: Vec<f32> = (0..96).map(|i| (i as f32 * 0.11).cos() * 3.0).collect();
        let mut blocks = Vec::new();
        quantize_activation_q8_32(&input, &mut blocks);
        assert_eq!(blocks.len(), 3);

        for (b, block) in blocks.iter().enumerate() {
            for i in 0..32 {
                let restored = block.d * block.qs[i] as f32;
                let original = input[b * 32 + i];
                assert!(
                    (restored - original).abs() <= block.d,
                    "block {b} index {i}: {restored} vs {original}"
                );
            }
            let expected: i32 = block.qs.iter().map(|&q| q as i32).sum();
            assert_eq!(block.sum, expected, "block {b} sum");
        }
    }

    #[test]
    fn layout_selection_covers_the_supported_formats() {
        assert_eq!(activation_layout(QuantType::Q4K), ActivationLayout::Q8K);
        assert_eq!(activation_layout(QuantType::Q5_0), ActivationLayout::Q8_32);
        assert_eq!(activation_layout(QuantType::Q8_0), ActivationLayout::Q8_32);
        assert_eq!(activation_layout(QuantType::Q4_0), ActivationLayout::Q8_32);
        // Formats without an integer kernel must fall back, not silently differ.
        assert_eq!(activation_layout(QuantType::Q5K), ActivationLayout::None);
        assert_eq!(activation_layout(QuantType::Q6K), ActivationLayout::None);
        assert_eq!(activation_layout(QuantType::Q4_1), ActivationLayout::None);
        assert_eq!(activation_layout(QuantType::F16), ActivationLayout::None);
    }

    #[test]
    fn scales_are_reused_not_appended() {
        let mut blocks = Vec::new();
        quantize_activation_q8_32(&vec![1.0; 64], &mut blocks);
        quantize_activation_q8_32(&vec![1.0; 32], &mut blocks);
        assert_eq!(blocks.len(), 1, "buffer must be cleared between calls");
    }
}
