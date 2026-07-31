# llama.rs

CPU-only Qwen2/Qwen2.5 inference engine in Rust.

## Implemented

### Model formats

- Hugging Face SafeTensors: F32, F16 and BF16; one-file and sharded models.
- Optional load-time quantization:
  - `q4k`: uniform `Q4_K`, with `Q5_0` fallback for rows divisible by 32 but not 256;
  - `q4km`: quality-biased mixed recipe using `Q8_0` for embeddings/value projections,
    `Q6_K` for sensitive down projections and Q4/Q5 elsewhere;
  - existing quantized GGUF tensors are used directly and are never requantized.
- GGUF v2/v3 with mmap-backed weights.
- GGUF tensor types: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
  Q4_K, Q5_K and Q6_K.

`Q4_K_M` is a file-level mixed-quantization recipe, not a tensor type. Such a GGUF
contains a mixture of Q4_K, Q5/Q6 and fallback tensors; the loader preserves that mix.

### Fast CPU backend

- Runtime dispatch with safe feature detection:
  - AVX2 + FMA on x86/x86-64;
  - NEON on AArch64;
  - portable scalar fallback.
- Dedicated SIMD dot-product kernels for Q4_0/Q4_1, Q5_0/Q5_1, Q4_K, Q5_K,
  Q6_K and Q8_0. Weights stay compressed; the AVX2 path decodes nibbles/bits in
  vector registers and does not materialize a full F32 row.
- Persistent, configurable Rayon thread pool (`--threads`), rather than creating
  threads per projection.
- Qwen2 RoPE sin/cos is computed once per position and reused by all layers.
- Prefill skips the expensive vocabulary projection for every prompt token except
  the last one.
- Preallocated activations and KV cache; no per-token model-buffer allocation.

On the two-vCPU test environment used while developing this revision, Qwen2.5-0.5B
Q4_K_M improved from roughly 2.7 decode tok/s with the scalar backend to 11.7 tok/s
with AVX2. Exact results depend on CPU, memory bandwidth, context length and thread count.
Use `llama-bench` on the target machine rather than relying on this example.

### Stable architecture boundary

- `architecture::CausalModel` separates model-family logic from the engine.
- `backend::KernelBackend` separates architecture code from numerical kernels.
- `ArchitectureKind` is selected from Hugging Face `model_type` or GGUF
  `general.architecture`.
- Qwen2 is the first implementation. A new architecture can reuse `QuantTensor`,
  `KernelBackend`, `KVCache`, metrics and sampling without changing the engine API.
- `ModelLoader::load_dynamic` is the architecture-neutral entry point;
  `ModelLoader::load` remains as a Qwen2-compatible API.

## Chat

A release build is essential.

```bash
cargo run --release --bin llama-rs -- \
  --model model/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  --context 4096 \
  --threads 8 \
  --backend auto
```

SafeTensors with load-time quantization:

```bash
cargo run --release --bin llama-rs -- \
  --model model/qwen-0.5b \
  --quantize q4km \
  --threads 8
```

Options:

```text
-m, --model PATH        HF model directory, .safetensors, or .gguf
-q, --quantize MODE     none, q4k, or q4km (default: none)
-c, --context N         allocated KV-cache context (default: 4096)
-n, --max-tokens N      generation limit (default: 512)
-t, --threads N         worker threads; 0 = all visible logical CPUs
    --backend NAME      auto, scalar, avx2, or neon
```

For a GGUF file, place the matching Hugging Face `tokenizer.json` next to it.

## Reproducible benchmark

`llama-bench` reports load, prefill and decode separately. It uses token IDs directly
after startup so tokenization and stochastic sampling are not included in kernel timings.
The median round is reported.

```bash
cargo run --release --bin llama-bench -- \
  --model model/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  --prompt-tokens 128 \
  --decode-tokens 32 \
  --rounds 3 \
  --threads 8 \
  --backend auto
```

Compare SIMD against the scalar reference:

```bash
cargo run --release --bin llama-bench -- -m model/model.gguf -p 128 -n 32 --backend scalar
cargo run --release --bin llama-bench -- -m model/model.gguf -p 128 -n 32 --backend avx2
```

Machine-readable output:

```bash
cargo run --release --bin llama-bench -- -m model/model.gguf --json
```

The interactive CLI also prints prefill and decode tok/s for every answer.

## Correctness and current scope

- Qwen2/Qwen2.5 dense models are supported.
- Q/K/V bias, GQA, split-half RoPE, RMSNorm and `silu(gate) * up` are implemented.
- SIMD kernels are tested against the scalar implementation for Q4_K, Q5_0,
  synthetic Q5_K, Q6_K and Q8_0.
- One GGUF file is supported; split GGUF sets are not yet joined.
- Standard RoPE is supported; YaRN/long-context scaling is not yet implemented.
- Prefill currently uses the causal single-token architecture path while avoiding
  intermediate LM-head projections. A tiled multi-token GEMM prefill path is the next
  major performance step.
- An offline llama.cpp quantization with a good importance matrix normally preserves
  more quality than calibration-free load-time quantization.
