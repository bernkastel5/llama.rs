// Numeric parity harness for the Q4_K x Q8_K integer kernel.
//
// The Rust toolchain is unavailable in this environment, so this file mirrors
// src/simd.rs::dot_q4k_q8k and src/quant.rs::dot_q4k line for line and checks
// that they agree. It verifies the arithmetic -- block layout, nibble order,
// scale/min handling, bsums indexing and saturation limits -- which is where
// this kernel can silently go wrong. It does not verify the Rust code compiles.
//
//   gcc -O2 -mavx2 -mfma -mf16c -o q8k_parity q8k_parity.c -lm && ./q8k_parity

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK_K 256
#define Q4K_BYTES 144

static float f16_to_f32(const uint8_t *data, size_t offset) {
    uint16_t bits = (uint16_t)(data[offset] | (data[offset + 1] << 8));
    return _cvtsh_ss(bits);
}

static uint16_t f32_to_f16(float value) { return _cvtss_sh(value, 0); }

// Mirrors src/quant.rs::scale_min_k4 and src/simd.rs::scale_min.
static void scale_min(size_t index, const uint8_t *scales, uint8_t *sc, uint8_t *mn) {
    if (index < 4) {
        *sc = scales[index] & 63;
        *mn = scales[index + 4] & 63;
    } else {
        *sc = (uint8_t)((scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4));
        *mn = (uint8_t)((scales[index + 4] >> 4) | ((scales[index] >> 6) << 4));
    }
}

// Mirrors src/quant.rs::dot_q4k -- the f32 reference.
static float dot_q4k_scalar(const uint8_t *row, size_t nblocks, const float *input) {
    float sum = 0.0f;
    for (size_t bi = 0; bi < nblocks; bi++) {
        const uint8_t *block = row + bi * Q4K_BYTES;
        float d = f16_to_f32(block, 0);
        float dmin = f16_to_f32(block, 2);
        const uint8_t *scales = block + 4;
        const uint8_t *qs = block + 16;
        const float *x = input + bi * QK_K;
        for (size_t group = 0; group < 8; group++) {
            uint8_t sc, mn;
            scale_min(group, scales, &sc, &mn);
            float ds = d * (float)sc;
            float dm = dmin * (float)mn;
            for (size_t j = 0; j < 32; j++) {
                uint8_t packed = qs[group / 2 * 32 + j];
                uint8_t q = (group % 2 == 0) ? (packed & 0x0f) : (packed >> 4);
                sum += (ds * (float)q - dm) * x[group * 32 + j];
            }
        }
    }
    return sum;
}

typedef struct {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
} block_q8K;

// Mirrors src/activation.rs::quantize_activation_q8k.
static void quantize_activation_q8k(const float *input, size_t n, block_q8K *out) {
    for (size_t b = 0; b < n / QK_K; b++) {
        const float *chunk = input + b * QK_K;
        block_q8K *blk = &out[b];
        memset(blk, 0, sizeof(*blk));
        float amax = 0.0f;
        for (size_t i = 0; i < QK_K; i++) {
            float a = fabsf(chunk[i]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float inv = (d == 0.0f) ? 0.0f : 1.0f / d;
        blk->d = d;
        for (size_t i = 0; i < QK_K; i++) {
            float r = roundf(chunk[i] * inv);
            if (r > 127.0f) r = 127.0f;
            if (r < -127.0f) r = -127.0f;
            blk->qs[i] = (int8_t)r;
        }
        for (size_t g = 0; g < QK_K / 16; g++) {
            int32_t total = 0;
            for (size_t j = 0; j < 16; j++) total += blk->qs[g * 16 + j];
            blk->bsums[g] = (int16_t)total;
        }
    }
}

static float hsum(__m256 value) {
    __m128 low = _mm256_castps256_ps128(value);
    __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

// Mirrors src/simd.rs::dot_q4k_q8k -- the kernel under test.
static float dot_q4k_q8k(const uint8_t *row, size_t nblocks, const block_q8K *act) {
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256 acc = _mm256_setzero_ps();
    float mins_total = 0.0f;
    for (size_t bi = 0; bi < nblocks; bi++) {
        const uint8_t *block = row + bi * Q4K_BYTES;
        float d = f16_to_f32(block, 0) * act[bi].d;
        float dmin = -f16_to_f32(block, 2) * act[bi].d;
        const uint8_t *scales = block + 4;
        const uint8_t *qs = block + 16;
        const int8_t *q8 = act[bi].qs;
        __m256i sumi = _mm256_setzero_si256();
        int32_t mins_acc = 0;
        for (size_t pair = 0; pair < 4; pair++) {
            __m256i packed = _mm256_loadu_si256((const __m256i *)(qs + pair * 32));
            __m256i lo = _mm256_and_si256(packed, low_mask);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
            uint8_t sl, ml, sh, mh;
            scale_min(pair * 2, scales, &sl, &ml);
            scale_min(pair * 2 + 1, scales, &sh, &mh);
            __m256i a_lo = _mm256_loadu_si256((const __m256i *)(q8 + pair * 64));
            __m256i a_hi = _mm256_loadu_si256((const __m256i *)(q8 + pair * 64 + 32));
            sumi = _mm256_add_epi32(
                sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(lo, a_lo), _mm256_set1_epi16(sl)));
            sumi = _mm256_add_epi32(
                sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(hi, a_hi), _mm256_set1_epi16(sh)));
            mins_acc += (int32_t)ml * (act[bi].bsums[pair * 4] + act[bi].bsums[pair * 4 + 1]) +
                        (int32_t)mh * (act[bi].bsums[pair * 4 + 2] + act[bi].bsums[pair * 4 + 3]);
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        mins_total += dmin * (float)mins_acc;
    }
    return hsum(acc) + mins_total;
}

static uint32_t rng_state = 12345;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((rng_state >> 8) & 0xffff) / 32768.0f - 1.0f;
}

// Build a Q4_K row with the scale/min structure a real quantizer produces.
static void make_row(uint8_t *row, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        uint8_t *block = row + b * Q4K_BYTES;
        uint16_t d = f32_to_f16(0.015f + 0.01f * fabsf(frand()));
        uint16_t dmin = f32_to_f16(0.004f + 0.003f * fabsf(frand()));
        memcpy(block, &d, 2);
        memcpy(block + 2, &dmin, 2);
        // Exercise both scale_min branches and the full 6-bit range.
        for (size_t i = 0; i < 12; i++) block[4 + i] = (uint8_t)(rng_state = rng_state * 1103515245u + 12345u) >> 1;
        for (size_t i = 0; i < 128; i++) block[16 + i] = (uint8_t)((rng_state = rng_state * 1103515245u + 12345u) >> 16);
    }
}

static int check(const char *name, size_t nblocks, float amp) {
    size_t n = nblocks * QK_K;
    uint8_t *row = malloc(nblocks * Q4K_BYTES);
    float *input = malloc(n * sizeof(float));
    block_q8K *act = malloc(nblocks * sizeof(block_q8K));
    make_row(row, nblocks);
    for (size_t i = 0; i < n; i++) input[i] = frand() * amp;
    quantize_activation_q8k(input, n, act);

    float reference = dot_q4k_scalar(row, nblocks, input);
    float integer = dot_q4k_q8k(row, nblocks, act);

    // Re-run the reference against the dequantized activation to separate
    // "the kernel is wrong" from "8-bit activations lose precision".
    float *dequant = malloc(n * sizeof(float));
    for (size_t b = 0; b < nblocks; b++)
        for (size_t i = 0; i < QK_K; i++)
            dequant[b * QK_K + i] = act[b].d * (float)act[b].qs[i];
    float exact = dot_q4k_scalar(row, nblocks, dequant);

    float kernel_err = fabsf(integer - exact) / fmaxf(fabsf(exact), 1e-6f);
    float quant_err = fabsf(reference - integer) / fmaxf(fabsf(reference), 1e-6f);
    int ok = kernel_err < 1e-5f;
    printf("%-22s blocks=%2zu amp=%-7.3f scalar=%12.5f int8=%12.5f | kernel_rel=%.2e %s | q8_rel=%.2e\n",
           name, nblocks, amp, reference, integer, kernel_err, ok ? "OK  " : "FAIL", quant_err);
    free(row); free(input); free(act); free(dequant);
    return ok;
}

int main(void) {
    printf("Q4_K x Q8_K integer kernel vs f32 scalar reference\n");
    printf("kernel_rel = kernel vs scalar on identical (dequantized) inputs -> must be ~0\n");
    printf("q8_rel     = cost of 8-bit activations vs full f32 -> expected ~1e-3\n\n");
    int ok = 1;
    ok &= check("single block", 1, 1.0f);
    ok &= check("hidden 896", 3, 1.0f);   // 896 is not a multiple of 256; 3 blocks stands in
    ok &= check("intermediate 4864", 19, 1.0f);
    ok &= check("large amplitude", 4, 40.0f);
    ok &= check("small amplitude", 4, 0.001f);
    ok &= check("saturation probe", 8, 1e4f);
    printf("\n%s\n", ok ? "PASS: integer kernel matches the scalar reference" : "FAIL");
    return ok ? 0 : 1;
}
