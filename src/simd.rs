use crate::activation::{BlockQ8K, BlockQ8_32};
use crate::quant::{dot_row_scalar, value_at, QuantType};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn hsum(value: __m256) -> f32 {
        let low = _mm256_castps256_ps128(value);
        let high = _mm256_extractf128_ps(value, 1);
        let sum = _mm_add_ps(low, high);
        let sum = _mm_hadd_ps(sum, sum);
        let sum = _mm_hadd_ps(sum, sum);
        _mm_cvtss_f32(sum)
    }

    #[inline]
    unsafe fn bytes8_to_i32(ptr: *const u8) -> __m256i {
        let bytes = _mm_loadl_epi64(ptr.cast::<__m128i>());
        _mm256_cvtepu8_epi32(bytes)
    }

    #[inline]
    unsafe fn i8x8_to_i32(ptr: *const u8) -> __m256i {
        let bytes = _mm_loadl_epi64(ptr.cast::<__m128i>());
        _mm256_cvtepi8_epi32(bytes)
    }

    #[inline]
    fn f16(data: &[u8], offset: usize) -> f32 {
        half::f16::from_bits(u16::from_le_bytes([data[offset], data[offset + 1]])).to_f32()
    }

    #[inline]
    fn scale_min(index: usize, scales: &[u8]) -> (u8, u8) {
        if index < 4 {
            (scales[index] & 63, scales[index + 4] & 63)
        } else {
            (
                (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
                (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
            )
        }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_f32(row: &[u8], input: &[f32]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let mut index = 0;
        while index + 8 <= input.len() {
            let weights = _mm256_loadu_ps(row.as_ptr().add(index * 4).cast::<f32>());
            let values = _mm256_loadu_ps(input.as_ptr().add(index));
            acc = _mm256_fmadd_ps(weights, values, acc);
            index += 8;
        }
        let mut sum = hsum(acc);
        while index < input.len() {
            let offset = index * 4;
            sum += f32::from_le_bytes([
                row[offset],
                row[offset + 1],
                row[offset + 2],
                row[offset + 3],
            ]) * input[index];
            index += 1;
        }
        sum
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q4(row: &[u8], input: &[f32], affine: bool) -> f32 {
        let block_size = if affine { 20 } else { 18 };
        let quant_offset = if affine { 4 } else { 2 };
        let mut sum = 0.0;
        for (block_index, block) in row.chunks_exact(block_size).enumerate() {
            let d = f16(block, 0);
            let min = if affine { f16(block, 2) } else { -8.0 * d };
            let x = input.as_ptr().add(block_index * 32);
            let mut qdot = _mm256_setzero_ps();
            let mut xsum = _mm256_setzero_ps();
            for half in 0..2 {
                for chunk in 0..2 {
                    let packed = bytes8_to_i32(block.as_ptr().add(quant_offset + chunk * 8));
                    let codes = if half == 0 {
                        _mm256_and_si256(packed, _mm256_set1_epi32(15))
                    } else {
                        _mm256_srli_epi32(packed, 4)
                    };
                    let values = _mm256_loadu_ps(x.add(half * 16 + chunk * 8));
                    qdot = _mm256_fmadd_ps(_mm256_cvtepi32_ps(codes), values, qdot);
                    xsum = _mm256_add_ps(xsum, values);
                }
            }
            sum += d * hsum(qdot) + min * hsum(xsum);
        }
        sum
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q5(row: &[u8], input: &[f32], affine: bool) -> f32 {
        let block_size = if affine { 24 } else { 22 };
        let qh_offset = if affine { 4 } else { 2 };
        let qs_offset = if affine { 8 } else { 6 };
        let mut sum = 0.0;
        for (block_index, block) in row.chunks_exact(block_size).enumerate() {
            let d = f16(block, 0);
            let min = if affine { f16(block, 2) } else { -16.0 * d };
            let qh = u32::from_le_bytes([
                block[qh_offset],
                block[qh_offset + 1],
                block[qh_offset + 2],
                block[qh_offset + 3],
            ]);
            let x = input.as_ptr().add(block_index * 32);
            let mut qdot = _mm256_setzero_ps();
            let mut xsum = _mm256_setzero_ps();
            for half in 0..2 {
                for chunk in 0..2 {
                    let base = chunk * 8;
                    let packed = bytes8_to_i32(block.as_ptr().add(qs_offset + base));
                    let low = if half == 0 {
                        _mm256_and_si256(packed, _mm256_set1_epi32(15))
                    } else {
                        _mm256_srli_epi32(packed, 4)
                    };
                    let bit = half * 16 + base;
                    let high_values = [
                        (((qh >> bit) & 1) << 4) as i32,
                        (((qh >> (bit + 1)) & 1) << 4) as i32,
                        (((qh >> (bit + 2)) & 1) << 4) as i32,
                        (((qh >> (bit + 3)) & 1) << 4) as i32,
                        (((qh >> (bit + 4)) & 1) << 4) as i32,
                        (((qh >> (bit + 5)) & 1) << 4) as i32,
                        (((qh >> (bit + 6)) & 1) << 4) as i32,
                        (((qh >> (bit + 7)) & 1) << 4) as i32,
                    ];
                    let high = _mm256_loadu_si256(high_values.as_ptr().cast::<__m256i>());
                    let codes = _mm256_add_epi32(low, high);
                    let values = _mm256_loadu_ps(x.add(half * 16 + base));
                    qdot = _mm256_fmadd_ps(_mm256_cvtepi32_ps(codes), values, qdot);
                    xsum = _mm256_add_ps(xsum, values);
                }
            }
            sum += d * hsum(qdot) + min * hsum(xsum);
        }
        sum
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q8(row: &[u8], input: &[f32]) -> f32 {
        let mut sum = 0.0;
        for (block_index, block) in row.chunks_exact(34).enumerate() {
            let d = f16(block, 0);
            let mut acc = _mm256_setzero_ps();
            for chunk in 0..4 {
                let codes = i8x8_to_i32(block.as_ptr().add(2 + chunk * 8));
                let values = _mm256_loadu_ps(input.as_ptr().add(block_index * 32 + chunk * 8));
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(codes), values, acc);
            }
            sum += d * hsum(acc);
        }
        sum
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q4k(row: &[u8], input: &[f32]) -> f32 {
        let mut sum = 0.0;
        for (block_index, block) in row.chunks_exact(144).enumerate() {
            let d = f16(block, 0);
            let dmin = f16(block, 2);
            let scales = &block[4..16];
            let qs = &block[16..144];
            let x = input.as_ptr().add(block_index * 256);
            for group in 0..8 {
                let (scale, min) = scale_min(group, scales);
                let mut qdot = _mm256_setzero_ps();
                let mut xsum = _mm256_setzero_ps();
                for chunk in 0..4 {
                    let packed = bytes8_to_i32(qs.as_ptr().add(group / 2 * 32 + chunk * 8));
                    let codes = if group % 2 == 0 {
                        _mm256_and_si256(packed, _mm256_set1_epi32(15))
                    } else {
                        _mm256_srli_epi32(packed, 4)
                    };
                    let values = _mm256_loadu_ps(x.add(group * 32 + chunk * 8));
                    qdot = _mm256_fmadd_ps(_mm256_cvtepi32_ps(codes), values, qdot);
                    xsum = _mm256_add_ps(xsum, values);
                }
                sum += d * scale as f32 * hsum(qdot) - dmin * min as f32 * hsum(xsum);
            }
        }
        sum
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q5k(row: &[u8], input: &[f32]) -> f32 {
        let mut sum = 0.0;
        for (block_index, block) in row.chunks_exact(176).enumerate() {
            let d = f16(block, 0);
            let dmin = f16(block, 2);
            let scales = &block[4..16];
            let qh = &block[16..48];
            let qs = &block[48..176];
            let x = input.as_ptr().add(block_index * 256);
            for group in 0..8 {
                let (scale, min) = scale_min(group, scales);
                let mask = _mm256_set1_epi32(1 << group);
                let mut qdot = _mm256_setzero_ps();
                let mut xsum = _mm256_setzero_ps();
                for chunk in 0..4 {
                    let packed = bytes8_to_i32(qs.as_ptr().add(group / 2 * 32 + chunk * 8));
                    let low = if group % 2 == 0 {
                        _mm256_and_si256(packed, _mm256_set1_epi32(15))
                    } else {
                        _mm256_srli_epi32(packed, 4)
                    };
                    let high_bytes = bytes8_to_i32(qh.as_ptr().add(chunk * 8));
                    let present = _mm256_cmpgt_epi32(
                        _mm256_and_si256(high_bytes, mask),
                        _mm256_setzero_si256(),
                    );
                    let high = _mm256_and_si256(present, _mm256_set1_epi32(16));
                    let codes = _mm256_add_epi32(low, high);
                    let values = _mm256_loadu_ps(x.add(group * 32 + chunk * 8));
                    qdot = _mm256_fmadd_ps(_mm256_cvtepi32_ps(codes), values, qdot);
                    xsum = _mm256_add_ps(xsum, values);
                }
                sum += d * scale as f32 * hsum(qdot) - dmin * min as f32 * hsum(xsum);
            }
        }
        sum
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q6k(row: &[u8], input: &[f32]) -> f32 {
        let mut sum = 0.0;
        for (block_index, block) in row.chunks_exact(210).enumerate() {
            let d = f16(block, 208);
            let x = input.as_ptr().add(block_index * 256);
            let mut acc = _mm256_setzero_ps();
            for half in 0..2 {
                let ql = block.as_ptr().add(half * 64);
                let qh = block.as_ptr().add(128 + half * 32);
                let scales = &block[192 + half * 8..192 + half * 8 + 8];
                for quarter in 0..4 {
                    let ql_base = if quarter % 2 == 0 { 0 } else { 32 };
                    for chunk in 0..4 {
                        let packed_low = bytes8_to_i32(ql.add(ql_base + chunk * 8));
                        let low = if quarter < 2 {
                            _mm256_and_si256(packed_low, _mm256_set1_epi32(15))
                        } else {
                            _mm256_srli_epi32(packed_low, 4)
                        };
                        let packed_high = bytes8_to_i32(qh.add(chunk * 8));
                        let shifted_high = match quarter {
                            0 => packed_high,
                            1 => _mm256_srli_epi32(packed_high, 2),
                            2 => _mm256_srli_epi32(packed_high, 4),
                            _ => _mm256_srli_epi32(packed_high, 6),
                        };
                        let high = _mm256_and_si256(shifted_high, _mm256_set1_epi32(3));
                        let codes = _mm256_sub_epi32(
                            _mm256_or_si256(low, _mm256_slli_epi32(high, 4)),
                            _mm256_set1_epi32(32),
                        );
                        let scale = scales[quarter * 2 + (chunk * 8) / 16] as i8 as f32 * d;
                        let values = _mm256_loadu_ps(x.add(half * 128 + quarter * 32 + chunk * 8));
                        let weights =
                            _mm256_mul_ps(_mm256_cvtepi32_ps(codes), _mm256_set1_ps(scale));
                        acc = _mm256_fmadd_ps(weights, values, acc);
                    }
                }
            }
            sum += hsum(acc);
        }
        sum
    }

    pub fn dot(row: &[u8], ty: QuantType, input: &[f32]) -> f32 {
        // SAFETY: CpuBackend selects this function only after runtime AVX2+FMA detection.
        unsafe {
            match ty {
                QuantType::None => dot_f32(row, input),
                QuantType::Q4_0 => dot_q4(row, input, false),
                QuantType::Q4_1 => dot_q4(row, input, true),
                QuantType::Q5_0 => dot_q5(row, input, false),
                QuantType::Q5_1 => dot_q5(row, input, true),
                QuantType::Q8_0 => dot_q8(row, input),
                QuantType::Q4K => dot_q4k(row, input),
                QuantType::Q5K => dot_q5k(row, input),
                QuantType::Q6K => dot_q6k(row, input),
                QuantType::F16 | QuantType::BF16 => dot_row_scalar(row, ty, input),
            }
        }
    }

    /// Q4_K weights against Q8_K activations, entirely in integer lanes.
    ///
    /// Both operands stay quantized: `_mm256_maddubs_epi16` consumes 32 weight
    /// codes per register instead of the 8 an f32 lane holds, and the only
    /// horizontal reduction happens once per row rather than eight times per
    /// 256-value super-block.
    ///
    /// Ranges are chosen so nothing saturates: weight nibbles are 0..15 and
    /// activations -128..127, so each `maddubs` pair reaches at most 3810,
    /// well inside i16; scaling by a 6-bit group scale keeps the `madd_epi16`
    /// result inside i32.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q4k_q8k(row: &[u8], activation: &[BlockQ8K]) -> f32 {
        let low_mask = _mm256_set1_epi8(0x0f);
        let mut acc = _mm256_setzero_ps();
        let mut mins_total = 0.0f32;
        for (block, act) in row.chunks_exact(144).zip(activation) {
            let d = f16(block, 0) * act.d;
            let dmin = -f16(block, 2) * act.d;
            let scales = &block[4..16];
            let qs = block.as_ptr().add(16);
            let q8 = act.qs.as_ptr();
            let mut sumi = _mm256_setzero_si256();
            let mut mins_acc = 0i32;
            for pair in 0..4 {
                // 32 bytes carry the low nibbles of group 2*pair and the high
                // nibbles of group 2*pair + 1.
                let packed = _mm256_loadu_si256(qs.add(pair * 32).cast::<__m256i>());
                let low = _mm256_and_si256(packed, low_mask);
                let high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
                let (scale_low, min_low) = scale_min(pair * 2, scales);
                let (scale_high, min_high) = scale_min(pair * 2 + 1, scales);
                let a_low = _mm256_loadu_si256(q8.add(pair * 64).cast::<__m256i>());
                let a_high = _mm256_loadu_si256(q8.add(pair * 64 + 32).cast::<__m256i>());
                sumi = _mm256_add_epi32(
                    sumi,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(low, a_low),
                        _mm256_set1_epi16(scale_low as i16),
                    ),
                );
                sumi = _mm256_add_epi32(
                    sumi,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(high, a_high),
                        _mm256_set1_epi16(scale_high as i16),
                    ),
                );
                // The per-group minimum multiplies the plain activation sum,
                // which `bsums` already holds at 16-lane granularity.
                mins_acc += min_low as i32
                    * (act.bsums[pair * 4] as i32 + act.bsums[pair * 4 + 1] as i32)
                    + min_high as i32
                        * (act.bsums[pair * 4 + 2] as i32 + act.bsums[pair * 4 + 3] as i32);
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            mins_total += dmin * mins_acc as f32;
        }
        hsum(acc) + mins_total
    }

    pub(super) fn dot_q8k(row: &[u8], ty: QuantType, activation: &[BlockQ8K]) -> Option<f32> {
        match ty {
            // SAFETY: reached only after runtime AVX2+FMA detection in
            // `CpuBackend`; lengths are validated by `QuantTensor`.
            QuantType::Q4K => Some(unsafe { dot_q4k_q8k(row, activation) }),
            _ => None,
        }
    }

    /// Expand 32 bits into 32 lanes of 0x00/0xFF, bit `j` selecting byte `j`.
    ///
    /// Q5_0 keeps the fifth bit of each code in a separate 32-bit field; the
    /// existing f32 kernel unpacked it with eight scalar shifts per 8 lanes.
    #[inline]
    unsafe fn bytes_from_bits_32(data: &[u8], offset: usize) -> __m256i {
        let bits = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        let shuffle = _mm256_set_epi64x(
            0x0303030303030303u64 as i64,
            0x0202020202020202u64 as i64,
            0x0101010101010101u64 as i64,
            0x0000000000000000u64 as i64,
        );
        let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(bits as i32), shuffle);
        let selector = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeu64 as i64);
        _mm256_cmpeq_epi8(
            _mm256_or_si256(bytes, selector),
            _mm256_set1_epi64x(-1i64),
        )
    }

    /// Unpack one 32-value nibble block into bytes ordered
    /// `[low0..low15, high0..high15]`, matching the scalar `x[j]`/`x[j+16]` pairing.
    #[inline]
    unsafe fn nibbles_32(qs: *const u8) -> __m256i {
        let packed = _mm_loadu_si128(qs.cast::<__m128i>());
        let low = _mm_and_si128(packed, _mm_set1_epi8(0x0f));
        let high = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0f));
        // `insert` rather than `_mm256_set_m128i`, which is not available in
        // every std::arch version this crate may be built against.
        _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1)
    }

    /// Signed 8-bit dot product. `maddubs` needs an unsigned left operand, so
    /// the sign of the weight is moved onto the activation.
    #[inline]
    unsafe fn mul_sum_i8(weight: __m256i, activation: __m256i) -> __m256i {
        let magnitude = _mm256_sign_epi8(weight, weight);
        let signed = _mm256_sign_epi8(activation, weight);
        _mm256_madd_epi16(
            _mm256_maddubs_epi16(magnitude, signed),
            _mm256_set1_epi16(1),
        )
    }

    /// Unsigned codes (0..31) against a signed activation.
    #[inline]
    unsafe fn mul_sum_u8(codes: __m256i, activation: __m256i) -> __m256i {
        _mm256_madd_epi16(
            _mm256_maddubs_epi16(codes, activation),
            _mm256_set1_epi16(1),
        )
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q8_0_q8_32(row: &[u8], activation: &[BlockQ8_32]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        for (block, act) in row.chunks_exact(34).zip(activation) {
            let d = f16(block, 0) * act.d;
            let weight = _mm256_loadu_si256(block.as_ptr().add(2).cast::<__m256i>());
            let values = _mm256_loadu_si256(act.qs.as_ptr().cast::<__m256i>());
            acc = _mm256_fmadd_ps(
                _mm256_set1_ps(d),
                _mm256_cvtepi32_ps(mul_sum_i8(weight, values)),
                acc,
            );
        }
        hsum(acc)
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q5_0_q8_32(row: &[u8], activation: &[BlockQ8_32]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let mut offset = 0.0f32;
        for (block, act) in row.chunks_exact(22).zip(activation) {
            let d = f16(block, 0) * act.d;
            let high = _mm256_and_si256(bytes_from_bits_32(block, 2), _mm256_set1_epi8(16));
            // Codes land in 0..31, so the unsigned `maddubs` operand is exact.
            let codes = _mm256_add_epi8(nibbles_32(block.as_ptr().add(6)), high);
            let values = _mm256_loadu_si256(act.qs.as_ptr().cast::<__m256i>());
            acc = _mm256_fmadd_ps(
                _mm256_set1_ps(d),
                _mm256_cvtepi32_ps(mul_sum_u8(codes, values)),
                acc,
            );
            // Every code carries a -16 bias, which factors out of the block.
            offset += d * 16.0 * act.sum as f32;
        }
        hsum(acc) - offset
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q4_0_q8_32(row: &[u8], activation: &[BlockQ8_32]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let mut offset = 0.0f32;
        for (block, act) in row.chunks_exact(18).zip(activation) {
            let d = f16(block, 0) * act.d;
            let codes = nibbles_32(block.as_ptr().add(2));
            let values = _mm256_loadu_si256(act.qs.as_ptr().cast::<__m256i>());
            acc = _mm256_fmadd_ps(
                _mm256_set1_ps(d),
                _mm256_cvtepi32_ps(mul_sum_u8(codes, values)),
                acc,
            );
            offset += d * 8.0 * act.sum as f32;
        }
        hsum(acc) - offset
    }

    pub(super) fn dot_q8_32(row: &[u8], ty: QuantType, activation: &[BlockQ8_32]) -> Option<f32> {
        // SAFETY: reached only after runtime AVX2+FMA detection in
        // `CpuBackend`; lengths are validated by `QuantTensor`.
        unsafe {
            match ty {
                QuantType::Q8_0 => Some(dot_q8_0_q8_32(row, activation)),
                QuantType::Q5_0 => Some(dot_q5_0_q8_32(row, activation)),
                QuantType::Q4_0 => Some(dot_q4_0_q8_32(row, activation)),
                _ => None,
            }
        }
    }
}

pub(crate) fn dot_row_avx2(row: &[u8], ty: QuantType, input: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        x86::dot(row, ty, input)
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        dot_row_scalar(row, ty, input)
    }
}

/// Integer-lane row product against Q8_K activations, or `None` when this
/// build has no kernel for `ty`.
///
/// Returning `None` keeps the caller's f32 path as the single fallback, so a
/// missing kernel degrades to the previous behaviour instead of failing.
pub(crate) fn dot_row_avx2_q8k(row: &[u8], ty: QuantType, activation: &[BlockQ8K]) -> Option<f32> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        x86::dot_q8k(row, ty, activation)
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = (row, ty, activation);
        None
    }
}

/// Integer-lane row product against 32-value Q8 activation blocks.
pub(crate) fn dot_row_avx2_q8_32(
    row: &[u8],
    ty: QuantType,
    activation: &[BlockQ8_32],
) -> Option<f32> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        x86::dot_q8_32(row, ty, activation)
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = (row, ty, activation);
        None
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon_impl(row: &[u8], ty: QuantType, input: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    // Decode in cache-sized chunks, then use four-wide NEON FMA. Quantized data
    // remains compressed in memory; only a small stack tile is materialized.
    let mut total = 0.0;
    let mut offset = 0;
    let mut tile = [0.0f32; 64];
    while offset < input.len() {
        let len = (input.len() - offset).min(tile.len());
        for i in 0..len {
            tile[i] = value_at(row, ty, offset + i);
        }
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 4 <= len {
            let weights = vld1q_f32(tile.as_ptr().add(i));
            let values = vld1q_f32(input.as_ptr().add(offset + i));
            acc = vfmaq_f32(acc, weights, values);
            i += 4;
        }
        total += vaddvq_f32(acc);
        while i < len {
            total += tile[i] * input[offset + i];
            i += 1;
        }
        offset += len;
    }
    total
}

pub(crate) fn dot_row_neon(row: &[u8], ty: QuantType, input: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: CpuBackend selects this function only after runtime NEON detection.
        unsafe { dot_neon_impl(row, ty, input) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = value_at;
        dot_row_scalar(row, ty, input)
    }
}
