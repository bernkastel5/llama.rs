#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define QK_K 256

typedef struct {
    uint8_t scales[12];
    uint8_t qs[128];
    float d;
    float dmin;
} block_q4_K;

typedef struct {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
} block_q8_K;

void quantize_row_q8_k(const float *x, block_q8_K *y, int k) {
    int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        y[i].d = 0.01f;
        for (int j = 0; j < 256; j++) y[i].qs[j] = (int8_t)(j & 0x7F);
    }
}

static inline float dot_q4k_fast(const block_q4_K *w, const block_q8_K *a) {
    __m256 acc = _mm256_setzero_ps();
    const __m256i *wq = (const __m256i*)w->qs;
    const __m256i *aq = (const __m256i*)a->qs;
    for (int i = 0; i < 4; i++) {
        __m256i v = _mm256_maddubs_epi16(_mm256_loadu_si256(wq + i), _mm256_loadu_si256(aq + i));
        acc = _mm256_add_ps(acc, _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(v, _mm256_setzero_si256())));
    }
    float buf[8];
    _mm256_storeu_ps(buf, acc);
    return (buf[0] + buf[1] + buf[2] + buf[3]) * w->d * a->d;
}

double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    int rows = 896;
    int cols = 4864;
    int nb = cols / QK_K;

    block_q4_K *weights = (block_q4_K*)aligned_alloc(64, rows * nb * sizeof(block_q4_K));
    memset(weights, 1, rows * nb * sizeof(block_q4_K));
    for (int i = 0; i < rows * nb; i++) { weights[i].d = 0.01f; weights[i].dmin = 0.005f; }

    int batches[] = {1, 4, 16, 32};
    for (int b_idx = 0; b_idx < 4; b_idx++) {
        int B = batches[b_idx];
        float *inputs = (float*)aligned_alloc(64, B * cols * sizeof(float));
        float *outputs = (float*)aligned_alloc(64, B * rows * sizeof(float));
        block_q8_K *q8_acts = (block_q8_K*)aligned_alloc(64, B * nb * sizeof(block_q8_K));

        for (int i = 0; i < B * cols; i++) inputs[i] = 0.1f * (i % 10);

        int iters = 20;
        double t0 = get_time_sec();
        for (int it = 0; it < iters; it++) {
            for (int b = 0; b < B; b++) {
                quantize_row_q8_k(&inputs[b * cols], &q8_acts[b * nb], cols);
                for (int r = 0; r < rows; r++) {
                    float sum = 0.0f;
                    for (int bi = 0; bi < nb; bi++) {
                        sum += dot_q4k_fast(&weights[r * nb + bi], &q8_acts[b * nb + bi]);
                    }
                    outputs[b * rows + r] = sum;
                }
            }
        }
        double t_unbatched = (get_time_sec() - t0) / (iters * B);

        t0 = get_time_sec();
        for (int it = 0; it < iters; it++) {
            for (int b = 0; b < B; b++) {
                quantize_row_q8_k(&inputs[b * cols], &q8_acts[b * nb], cols);
            }
            for (int r = 0; r < rows; r++) {
                for (int b = 0; b < B; b++) {
                    float sum = 0.0f;
                    for (int bi = 0; bi < nb; bi++) {
                        sum += dot_q4k_fast(&weights[r * nb + bi], &q8_acts[b * nb + bi]);
                    }
                    outputs[b * rows + r] = sum;
                }
            }
        }
        double t_batched = (get_time_sec() - t0) / (iters * B);

        printf("Batch size %2d: unbatched = %6.3f ms/token, batched = %6.3f ms/token, speedup = %5.2fx\n",
               B, t_unbatched * 1000.0, t_batched * 1000.0, t_unbatched / t_batched);

        free(inputs);
        free(outputs);
        free(q8_acts);
    }

    free(weights);
    return 0;
}
