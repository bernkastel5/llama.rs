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

    /// Q5_K weights against Q8_K activations, entirely in integer lanes.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q5k_q8k(row: &[u8], activation: &[BlockQ8K]) -> f32 {
        let low_mask = _mm256_set1_epi8(0x0f);
        let mut acc = _mm256_setzero_ps();
        let mut mins_total = 0.0f32;

        for (block, act) in row.chunks_exact(176).zip(activation) {
            let d = f16(block, 0) * act.d;
            let dmin = -f16(block, 2) * act.d;
            let scales = &block[4..16];
            let qh = block.as_ptr().add(16);
            let qs = block.as_ptr().add(48);
            let q8 = act.qs.as_ptr();

            let mut sumi = _mm256_setzero_si256();
            let mut mins_acc = 0i32;
            let qh_val = _mm256_loadu_si256(qh.cast::<__m256i>());

            for pair in 0..4 {
                let packed = _mm256_loadu_si256(qs.add(pair * 32).cast::<__m256i>());
                let low = _mm256_and_si256(packed, low_mask);
                let high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);

                let g0 = pair * 2;
                let g1 = pair * 2 + 1;

                let mask0 = _mm256_set1_epi8(1 << g0);
                let mask1 = _mm256_set1_epi8(1 << g1);

                let h0 = _mm256_and_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(qh_val, mask0), mask0),
                    _mm256_set1_epi8(16),
                );
                let h1 = _mm256_and_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(qh_val, mask1), mask1),
                    _mm256_set1_epi8(16),
                );

                let q0 = _mm256_or_si256(low, h0);
                let q1 = _mm256_or_si256(high, h1);

                let (sc0, m0) = scale_min(g0, scales);
                let (sc1, m1) = scale_min(g1, scales);

                let a0 = _mm256_loadu_si256(q8.add(pair * 64).cast::<__m256i>());
                let a1 = _mm256_loadu_si256(q8.add(pair * 64 + 32).cast::<__m256i>());

                sumi = _mm256_add_epi32(
                    sumi,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(q0, a0),
                        _mm256_set1_epi16(sc0 as i16),
                    ),
                );
                sumi = _mm256_add_epi32(
                    sumi,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(q1, a1),
                        _mm256_set1_epi16(sc1 as i16),
                    ),
                );

                mins_acc += m0 as i32
                    * (act.bsums[pair * 4] as i32 + act.bsums[pair * 4 + 1] as i32)
                    + m1 as i32
                        * (act.bsums[pair * 4 + 2] as i32 + act.bsums[pair * 4 + 3] as i32);
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            mins_total += dmin * mins_acc as f32;
        }
        hsum(acc) + mins_total
    }

    /// Q6_K weights against Q8_K activations, entirely in integer lanes.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q6k_q8k(row: &[u8], activation: &[BlockQ8K]) -> f32 {
        let low_mask = _mm256_set1_epi8(0x0f);
        let high_mask = _mm256_set1_epi8(0x03);
        let mut acc = _mm256_setzero_ps();
        let mut mins_total = 0.0f32;

        for (block, act) in row.chunks_exact(210).zip(activation) {
            let d = f16(block, 208) * act.d;
            let mut sumi = _mm256_setzero_si256();
            let mut mins_acc = 0i32;

            for half in 0..2 {
                let ql = block.as_ptr().add(half * 64);
                let qh = block.as_ptr().add(128 + half * 32);
                let scales = block.as_ptr().add(192 + half * 8);
                let q8 = act.qs.as_ptr().add(half * 128);
                let bsums = act.bsums.as_ptr().add(half * 8);

                let qh_val = _mm256_loadu_si256(qh.cast::<__m256i>());

                for quarter in 0..4 {
                    let ql_off = if quarter % 2 == 0 { 0 } else { 32 };
                    let ql_val = _mm256_loadu_si256(ql.add(ql_off).cast::<__m256i>());
                    let low = if quarter < 2 {
                        _mm256_and_si256(ql_val, low_mask)
                    } else {
                        _mm256_and_si256(_mm256_srli_epi16(ql_val, 4), low_mask)
                    };

                    let high_shifted = match quarter {
                        0 => qh_val,
                        1 => _mm256_srli_epi16(qh_val, 2),
                        2 => _mm256_srli_epi16(qh_val, 4),
                        _ => _mm256_srli_epi16(qh_val, 6),
                    };
                    let high = _mm256_and_si256(high_shifted, high_mask);

                    let q = _mm256_or_si256(low, _mm256_slli_epi16(high, 4));
                    let a = _mm256_loadu_si256(q8.add(quarter * 32).cast::<__m256i>());

                    let sc0 = *scales.add(quarter * 2) as i8 as i16;
                    let sc1 = *scales.add(quarter * 2 + 1) as i8 as i16;

                    let prod = _mm256_maddubs_epi16(q, a);
                    let sc_vec = _mm256_set_epi16(
                        sc1, sc1, sc1, sc1, sc1, sc1, sc1, sc1,
                        sc0, sc0, sc0, sc0, sc0, sc0, sc0, sc0,
                    );
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(prod, sc_vec));

                    mins_acc += sc0 as i32 * *bsums.add(quarter * 2) as i32
                        + sc1 as i32 * *bsums.add(quarter * 2 + 1) as i32;
                }
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            mins_total += d * 32.0 * mins_acc as f32;
        }
        hsum(acc) - mins_total
    }

    pub(super) fn dot_q8k(row: &[u8], ty: QuantType, activation: &[BlockQ8K]) -> Option<f32> {
        match ty {
            // SAFETY: reached only after runtime AVX2+FMA detection in
            // `CpuBackend`; lengths are validated by `QuantTensor`.
            QuantType::Q4K => Some(unsafe { dot_q4k_q8k(row, activation) }),
            QuantType::Q5K => Some(unsafe { dot_q5k_q8k(row, activation) }),
            QuantType::Q6K => Some(unsafe { dot_q6k_q8k(row, activation) }),
            _ => None,
        }
    }

    /// Expand 32 bits into 32 lanes of 0x00/0xFF, bit `j` selecting byte `j`.
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
            let codes = _mm256_add_epi8(nibbles_32(block.as_ptr().add(6)), high);
            let values = _mm256_loadu_si256(act.qs.as_ptr().cast::<__m256i>());
            acc = _mm256_fmadd_ps(
                _mm256_set1_ps(d),
                _mm256_cvtepi32_ps(mul_sum_u8(codes, values)),
                acc,
            );
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

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_dot_f32_impl(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let mut index = 0;
        while index + 8 <= a.len() {
            let va = _mm256_loadu_ps(a.as_ptr().add(index));
            let vb = _mm256_loadu_ps(b.as_ptr().add(index));
            acc = _mm256_fmadd_ps(va, vb, acc);
            index += 8;
        }
        let mut sum = hsum(acc);
        while index < a.len() {
            sum += a[index] * b[index];
            index += 1;
        }
        sum
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_mad_f32_impl(dst: &mut [f32], src: &[f32], scale: f32) {
        let vs = _mm256_set1_ps(scale);
        let mut index = 0;
        while index + 8 <= dst.len() {
            let vd = _mm256_loadu_ps(dst.as_ptr().add(index));
            let vsrc = _mm256_loadu_ps(src.as_ptr().add(index));
            let res = _mm256_fmadd_ps(vs, vsrc, vd);
            _mm256_storeu_ps(dst.as_mut_ptr().add(index), res);
            index += 8;
        }
        while index < dst.len() {
            dst[index] += scale * src[index];
            index += 1;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_add_f32_impl(dst: &mut [f32], src: &[f32]) {
        let mut index = 0;
        while index + 8 <= dst.len() {
            let vd = _mm256_loadu_ps(dst.as_ptr().add(index));
            let vsrc = _mm256_loadu_ps(src.as_ptr().add(index));
            let res = _mm256_add_ps(vd, vsrc);
            _mm256_storeu_ps(dst.as_mut_ptr().add(index), res);
            index += 8;
        }
        while index < dst.len() {
            dst[index] += src[index];
            index += 1;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_rmsnorm_impl(hidden: &mut [f32], weight: &[f32], eps: f32) {
        let sum_sq = vec_dot_f32_impl(hidden, hidden);
        let mean_sq = sum_sq / hidden.len() as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        let v_inv = _mm256_set1_ps(inv_rms);
        let mut index = 0;
        while index + 8 <= hidden.len() {
            let vh = _mm256_loadu_ps(hidden.as_ptr().add(index));
            let vw = _mm256_loadu_ps(weight.as_ptr().add(index));
            let res = _mm256_mul_ps(_mm256_mul_ps(vh, vw), v_inv);
            _mm256_storeu_ps(hidden.as_mut_ptr().add(index), res);
            index += 8;
        }
        while index < hidden.len() {
            hidden[index] *= inv_rms * weight[index];
            index += 1;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_rope_inplace_impl(x: &mut [f32], cos: &[f32], sin: &[f32]) {
        let half = x.len() / 2;
        let mut index = 0;
        while index + 8 <= half {
            let r = _mm256_loadu_ps(x.as_ptr().add(index));
            let im = _mm256_loadu_ps(x.as_ptr().add(half + index));
            let c = _mm256_loadu_ps(cos.as_ptr().add(index));
            let s = _mm256_loadu_ps(sin.as_ptr().add(index));
            let r_new = _mm256_fmsub_ps(r, c, _mm256_mul_ps(im, s));
            let im_new = _mm256_fmadd_ps(r, s, _mm256_mul_ps(im, c));
            _mm256_storeu_ps(x.as_mut_ptr().add(index), r_new);
            _mm256_storeu_ps(x.as_mut_ptr().add(half + index), im_new);
            index += 8;
        }
        while index < half {
            let r = x[index];
            let im = x[index + half];
            x[index] = r * cos[index] - im * sin[index];
            x[index + half] = r * sin[index] + im * cos[index];
            index += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp256_ps(mut x: __m256) -> __m256 {
        x = _mm256_max_ps(x, _mm256_set1_ps(-87.336544));
        x = _mm256_min_ps(x, _mm256_set1_ps(88.722839));

        let log2e = _mm256_set1_ps(1.44269504088896341);
        let ln2_hi = _mm256_set1_ps(0.6931471805599453);
        let ln2_lo = _mm256_set1_ps(2.3190468138462996e-17);

        let z = _mm256_mul_ps(x, log2e);
        let emm0 = _mm256_cvtps_epi32(z);
        let fx = _mm256_cvtepi32_ps(emm0);

        let mut x_r = _mm256_fnmadd_ps(fx, ln2_hi, x);
        x_r = _mm256_fnmadd_ps(fx, ln2_lo, x_r);

        let c5 = _mm256_set1_ps(1.0 / 120.0);
        let c4 = _mm256_set1_ps(1.0 / 24.0);
        let c3 = _mm256_set1_ps(1.0 / 6.0);
        let c2 = _mm256_set1_ps(0.5);

        let mut poly = _mm256_fmadd_ps(c5, x_r, c4);
        poly = _mm256_fmadd_ps(poly, x_r, c3);
        poly = _mm256_fmadd_ps(poly, x_r, c2);
        poly = _mm256_fmadd_ps(poly, x_r, _mm256_set1_ps(1.0));
        poly = _mm256_fmadd_ps(poly, x_r, _mm256_set1_ps(1.0));

        let emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
        let emm0 = _mm256_slli_epi32(emm0, 23);
        let pow2n = _mm256_castsi256_ps(emm0);

        _mm256_mul_ps(poly, pow2n)
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_swiglu_impl(gate: &mut [f32], up: &[f32]) {
        let one = _mm256_set1_ps(1.0);
        let zero = _mm256_setzero_ps();
        let mut index = 0;
        while index + 8 <= gate.len() {
            let g = _mm256_loadu_ps(gate.as_ptr().add(index));
            let u = _mm256_loadu_ps(up.as_ptr().add(index));
            let neg_g = _mm256_sub_ps(zero, g);
            let exp_neg_g = exp256_ps(neg_g);
            let denom = _mm256_add_ps(one, exp_neg_g);
            let silu_g = _mm256_div_ps(g, denom);
            _mm256_storeu_ps(gate.as_mut_ptr().add(index), _mm256_mul_ps(silu_g, u));
            index += 8;
        }
        while index < gate.len() {
            let g = gate[index];
            gate[index] = (g / (1.0 + (-g).exp())) * up[index];
            index += 1;
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

pub(crate) fn vec_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { x86::vec_dot_f32_impl(a, b) };
        }
    }
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

pub(crate) fn vec_mad_f32(dst: &mut [f32], src: &[f32], scale: f32) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { x86::vec_mad_f32_impl(dst, src, scale) };
            return;
        }
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += scale * s;
    }
}

pub(crate) fn vec_add_f32(dst: &mut [f32], src: &[f32]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { x86::vec_add_f32_impl(dst, src) };
            return;
        }
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

pub(crate) fn vec_rmsnorm(hidden: &mut [f32], weight: &[f32], eps: f32) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { x86::vec_rmsnorm_impl(hidden, weight, eps) };
            return;
        }
    }
    let mean_sq = hidden.iter().map(|x| x * x).sum::<f32>() / hidden.len() as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    for (value, &w) in hidden.iter_mut().zip(weight) {
        *value *= inv_rms * w;
    }
}

pub(crate) fn vec_rope_inplace(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { x86::vec_rope_inplace_impl(x, cos, sin) };
            return;
        }
    }
    let half = x.len() / 2;
    for i in 0..half {
        let real = x[i];
        let imag = x[i + half];
        x[i] = real * cos[i] - imag * sin[i];
        x[i + half] = real * sin[i] + imag * cos[i];
    }
}

pub(crate) fn vec_swiglu(gate: &mut [f32], up: &[f32]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { x86::vec_swiglu_impl(gate, up) };
            return;
        }
    }
    for (g, &u) in gate.iter_mut().zip(up) {
        let val = *g;
        *g = (val / (1.0 + (-val).exp())) * u;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon_impl(row: &[u8], ty: QuantType, input: &[f32]) -> f32 {
    use std::arch::aarch64::*;
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
