// Numeric parity harness for the 32-block integer kernels (Q8_0, Q5_0, Q4_0)
// against Q8_32 activations.
//
// Motivation: on Qwen2.5-0.5B hidden_size=896 is not a multiple of QK_K=256,
// so quantize_q4k falls back to Q5_0 for every projection except down_proj.
// A Q4_K-only integer kernel would therefore accelerate ~21% of the weights.
// These 32-block kernels cover the rest.
//
//   gcc -O2 -mavx2 -mfma -mf16c -o q8_32_parity q8_32_parity.c -lm && ./q8_32_parity

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float f16_to_f32(const uint8_t *d, size_t o) {
    return _cvtsh_ss((uint16_t)(d[o] | (d[o + 1] << 8)));
}
static uint16_t f32_to_f16(float v) { return _cvtss_sh(v, 0); }

// ---- scalar references, mirroring src/quant.rs::dot_row_scalar ----

static float dot_q8_0_scalar(const uint8_t *row, size_t nb, const float *in) {
    float sum = 0.0f;
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t *b = row + bi * 34;
        float d = f16_to_f32(b, 0);
        const float *x = in + bi * 32;
        for (size_t j = 0; j < 32; j++) sum += d * (float)(int8_t)b[2 + j] * x[j];
    }
    return sum;
}

static float dot_q5_0_scalar(const uint8_t *row, size_t nb, const float *in) {
    float sum = 0.0f;
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t *b = row + bi * 22;
        float d = f16_to_f32(b, 0);
        uint32_t qh;
        memcpy(&qh, b + 2, 4);
        const float *x = in + bi * 32;
        for (size_t j = 0; j < 16; j++) {
            uint8_t packed = b[6 + j];
            uint32_t q0 = (packed & 0x0f) | (((qh >> j) & 1) << 4);
            uint32_t q1 = (packed >> 4) | (((qh >> (j + 16)) & 1) << 4);
            sum += d * ((float)q0 - 16.0f) * x[j];
            sum += d * ((float)q1 - 16.0f) * x[j + 16];
        }
    }
    return sum;
}

static float dot_q4_0_scalar(const uint8_t *row, size_t nb, const float *in) {
    float sum = 0.0f;
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t *b = row + bi * 18;
        float d = f16_to_f32(b, 0);
        const float *x = in + bi * 32;
        for (size_t j = 0; j < 16; j++) {
            uint8_t q = b[2 + j];
            sum += d * ((float)(q & 0x0f) - 8.0f) * x[j];
            sum += d * ((float)(q >> 4) - 8.0f) * x[j + 16];
        }
    }
    return sum;
}

// ---- activation: one scale per 32 values, plus the lane sum ----

typedef struct {
    float d;
    int8_t qs[32];
    int32_t sum;  // sum(qs), for weight formats with a constant offset
} block_q8_32;

static void quantize_activation_q8_32(const float *in, size_t n, block_q8_32 *out) {
    for (size_t b = 0; b < n / 32; b++) {
        const float *c = in + b * 32;
        block_q8_32 *blk = &out[b];
        float amax = 0.0f;
        for (size_t i = 0; i < 32; i++) {
            float a = fabsf(c[i]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float inv = (d == 0.0f) ? 0.0f : 1.0f / d;
        blk->d = d;
        int32_t s = 0;
        for (size_t i = 0; i < 32; i++) {
            float r = roundf(c[i] * inv);
            if (r > 127.0f) r = 127.0f;
            if (r < -127.0f) r = -127.0f;
            blk->qs[i] = (int8_t)r;
            s += blk->qs[i];
        }
        blk->sum = s;
    }
}

static float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// Sum of 8 i32 lanes into one i32.
static int32_t hsum_i32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

// Expand 32 bits into 32 bytes of 0x00/0xFF, bit j -> byte j.
static __m256i bytes_from_bits_32(const uint8_t *x) {
    uint32_t x32;
    memcpy(&x32, x, 4);
    const __m256i shuf = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202,
                                           0x0101010101010101, 0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int)x32), shuf);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

// i8 x i8 via the sign trick: maddubs needs an unsigned left operand.
static __m256i mul_sum_i8(__m256i w, __m256i a) {
    __m256i aw = _mm256_sign_epi8(w, w);       // |w|, unsigned
    __m256i sa = _mm256_sign_epi8(a, w);       // a * sign(w)
    return _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sa), _mm256_set1_epi16(1));
}

// Unsigned codes (0..31) x i8 activation.
static __m256i mul_sum_u8(__m256i codes, __m256i a) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(codes, a), _mm256_set1_epi16(1));
}

// Nibbles of one 32-value block, ordered [lo0..lo15, hi0..hi15] to match the
// scalar loop's x[j] / x[j+16] pairing.
static __m256i nibbles_32(const uint8_t *qs) {
    __m128i packed = _mm_loadu_si128((const __m128i *)qs);
    __m128i lo = _mm_and_si128(packed, _mm_set1_epi8(0x0f));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0f));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

static float dot_q8_0_q8_32(const uint8_t *row, size_t nb, const block_q8_32 *act) {
    __m256 acc = _mm256_setzero_ps();
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t *b = row + bi * 34;
        float d = f16_to_f32(b, 0) * act[bi].d;
        __m256i w = _mm256_loadu_si256((const __m256i *)(b + 2));
        __m256i a = _mm256_loadu_si256((const __m256i *)act[bi].qs);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(mul_sum_i8(w, a)), acc);
    }
    return hsum(acc);
}

static float dot_q5_0_q8_32(const uint8_t *row, size_t nb, const block_q8_32 *act) {
    __m256 acc = _mm256_setzero_ps();
    float offset = 0.0f;
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t *b = row + bi * 22;
        float dw = f16_to_f32(b, 0);
        float d = dw * act[bi].d;
        __m256i codes = nibbles_32(b + 6);
        __m256i high = _mm256_and_si256(bytes_from_bits_32(b + 2), _mm256_set1_epi8(16));
        codes = _mm256_add_epi8(codes, high);  // 0..31, still unsigned-safe
        __m256i a = _mm256_loadu_si256((const __m256i *)act[bi].qs);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(mul_sum_u8(codes, a)), acc);
        // The -16 bias applies to every lane, so it factors out into the sum.
        offset += d * 16.0f * (float)act[bi].sum;
    }
    return hsum(acc) - offset;
}

static float dot_q4_0_q8_32(const uint8_t *row, size_t nb, const block_q8_32 *act) {
    __m256 acc = _mm256_setzero_ps();
    float offset = 0.0f;
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t *b = row + bi * 18;
        float d = f16_to_f32(b, 0) * act[bi].d;
        __m256i codes = nibbles_32(b + 2);
        __m256i a = _mm256_loadu_si256((const __m256i *)act[bi].qs);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(mul_sum_u8(codes, a)), acc);
        offset += d * 8.0f * (float)act[bi].sum;
    }
    return hsum(acc) - offset;
}

static uint32_t rs = 987654321;
static uint32_t nextr(void) { return rs = rs * 1664525u + 1013904223u; }
static float frand(void) { return (float)((nextr() >> 8) & 0xffff) / 32768.0f - 1.0f; }

typedef float (*scalar_fn)(const uint8_t *, size_t, const float *);
typedef float (*int_fn)(const uint8_t *, size_t, const block_q8_32 *);

static int check(const char *name, size_t bytes, size_t nb, float amp, scalar_fn sf, int_fn kf) {
    size_t n = nb * 32;
    uint8_t *row = malloc(nb * bytes);
    for (size_t i = 0; i < nb * bytes; i++) row[i] = (uint8_t)(nextr() >> 16);
    // Give every block a sane scale rather than a random f16 bit pattern.
    for (size_t b = 0; b < nb; b++) {
        uint16_t d = f32_to_f16(0.02f + 0.01f * fabsf(frand()));
        memcpy(row + b * bytes, &d, 2);
    }
    float *in = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) in[i] = frand() * amp;
    block_q8_32 *act = malloc(nb * sizeof(block_q8_32));
    quantize_activation_q8_32(in, n, act);

    float reference = sf(row, nb, in);
    float integer = kf(row, nb, act);

    float *deq = malloc(n * sizeof(float));
    for (size_t b = 0; b < nb; b++)
        for (size_t i = 0; i < 32; i++) deq[b * 32 + i] = act[b].d * (float)act[b].qs[i];
    float exact = sf(row, nb, deq);

    float kernel_rel = fabsf(integer - exact) / fmaxf(fabsf(exact), 1e-6f);
    float q8_rel = fabsf(reference - integer) / fmaxf(fabsf(reference), 1e-6f);
    int ok = kernel_rel < 1e-5f;
    printf("%-16s blocks=%3zu amp=%-8.3f scalar=%12.4f int8=%12.4f | kernel_rel=%.2e %s | q8_rel=%.2e\n",
           name, nb, amp, reference, integer, kernel_rel, ok ? "OK  " : "FAIL", q8_rel);
    free(row); free(in); free(act); free(deq);
    return ok;
}

int main(void) {
    printf("32-block integer kernels vs f32 scalar references\n");
    printf("kernel_rel must be ~0; q8_rel is the inherent cost of 8-bit activations\n\n");
    int ok = 1;
    // 896 columns = 28 blocks: the real q_proj/o_proj/gate/up/lm_head shape.
    ok &= check("Q8_0 c=896", 34, 28, 1.0f, dot_q8_0_scalar, dot_q8_0_q8_32);
    ok &= check("Q8_0 c=4864", 34, 152, 1.0f, dot_q8_0_scalar, dot_q8_0_q8_32);
    ok &= check("Q8_0 big amp", 34, 28, 100.0f, dot_q8_0_scalar, dot_q8_0_q8_32);
    ok &= check("Q5_0 c=896", 22, 28, 1.0f, dot_q5_0_scalar, dot_q5_0_q8_32);
    ok &= check("Q5_0 c=4864", 22, 152, 1.0f, dot_q5_0_scalar, dot_q5_0_q8_32);
    ok &= check("Q5_0 small amp", 22, 28, 0.001f, dot_q5_0_scalar, dot_q5_0_q8_32);
    ok &= check("Q4_0 c=896", 18, 28, 1.0f, dot_q4_0_scalar, dot_q4_0_q8_32);
    ok &= check("Q4_0 big amp", 18, 28, 500.0f, dot_q4_0_scalar, dot_q4_0_q8_32);
    printf("\n%s\n", ok ? "PASS: all 32-block integer kernels match" : "FAIL");
    return ok ? 0 : 1;
}
