# llama.rs

CPU-only Qwen2/Qwen2.5 inference prototype in Rust.

## Implemented

- Hugging Face SafeTensors: F32, F16 and BF16; one-file and sharded models.
- Optional load-time quantization:
  - `q4k`: uniform `Q4_K`, with llama.cpp-compatible `Q5_0` fallback for rows
    divisible by 32 but not by 256.
  - `q4km`: quality-biased Q4_K_M-style load-time recipe using `Q8_0` for sensitive
    embeddings/value projections and `Q6_K` for down projections. Existing imatrix
    Q4_K_M GGUF files keep their original Q4_K/Q6_K mixture.
  - Other unusual row widths remain F32.
- GGUF v2/v3 parser and mmap-backed weights.
- GGUF tensor kernels: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
  Q4_K, Q5_K and Q6_K.
- Correct Qwen2 Q/K/V biases, GQA, split-half RoPE, RMSNorm and SwiGLU.
- Full prompt prefill and a preallocated KV cache.

`Q4_K_M` is a model/file-level mixed-quantization recipe in llama.cpp, not a tensor
encoding. The tensor encoding in a Q4_K_M GGUF is `Q4_K`, with selected tensors in
Q5_K/Q6_K or compatible fallback types. The GGUF loader therefore accepts this mix.

## Run

A release build is important; the debug build is intentionally very slow.

```bash
cargo run --release -- --model model/qwen-0.5b --quantize none
cargo run --release -- --model model/qwen-0.5b --quantize q4km
cargo run --release -- --model model/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Options:

```text
-m, --model PATH        HF model directory, .safetensors, or .gguf
-q, --quantize MODE     none, q4k, or q4km (default: none)
-c, --context N         allocated KV-cache context (default: 4096)
-n, --max-tokens N      generation limit (default: 512)
```

For a GGUF file, place the matching Hugging Face `tokenizer.json` next to the GGUF.
The weights and model configuration come from GGUF; using the sidecar tokenizer avoids
silently approximating Qwen2's pre-tokenization rules from incomplete GGUF tokenizer
metadata.

## Current scope

- Architecture: Qwen2/Qwen2.5 dense models only.
- One GGUF file (split GGUF sets are not yet joined).
- Standard RoPE only; YaRN/long-context rope scaling is not implemented.
- Scalar dequantization kernels with row-level Rayon parallelism. The data layouts are
  GGML-compatible, but AVX2/AVX-512/NEON fused dot kernels are the next performance step.
- Load-time Q4_K ports llama.cpp's deterministic weighted least-squares reference encoder
  and writes the exact GGML block layout. An offline llama.cpp quantization with a good
  importance matrix can still preserve more quality; for the best result, use an existing
  imatrix-generated Q4_K_M GGUF.
