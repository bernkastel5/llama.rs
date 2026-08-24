# C-микробенчмарки

Воспроизводят алгоритмы `src/simd.rs`, `src/model.rs` и `src/backend.rs` на C, чтобы
измерить эффект предлагаемых оптимизаций без Rust-тулчейна. Использованы при подготовке
`docs/PERF_PLAN.md`.

Сборка: `gcc -O3 -mavx2 -mfma -mf16c -o NAME NAME.c -lm -lpthread`

| файл | что меряет | аргументы |
|---|---|---|
| `q4k.c`   | Q4_K matvec: f32-ядро репозитория vs int8 Q4_K×Q8_K | `rows cols iters` (cols кратно 256) |
| `attn.c`  | QK^T: скаляр vs AVX2 f32-KV vs AVX2 f16-KV | `ctx iters` |
| `gemm.c`  | prefill: N отдельных matvec vs батч-GEMM | `rows cols N iters` |
| `barrier.c` | `pthread_barrier` fork-join оверхед | `nthreads iters` |
| `spin.c`  | atomic spin-барьер | `nthreads iters` |
| `bw.c`    | пропускная способность, nt-load, кэш-резидентно | `MB nthreads [rep]` |
| `bw2.c`   | пропускная способность, реальный стриминг | `MB nthreads [rep]` |
| `vnni.c`  | AVX2 vs AVX-512/VNNI на Q4_K (вывод: выигрыша нет) | `rows cols iters` |
| `full.c`  | сквозной шаг декодирования Qwen2.5-0.5B: текущая vs предлагаемая | `nthreads ctx iters` |
