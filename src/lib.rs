//! CPU-only Qwen2/Qwen2.5 inference in Rust.
//!
//! - Hugging Face SafeTensors (F32/F16/BF16, including sharded models)
//! - GGUF v2/v3 with F32/F16/BF16, Q4_0/Q4_1, Q5_0/Q5_1,
//!   Q8_0, Q4_K, Q5_K and Q6_K tensors
//! - optional load-time Q4_K and mixed Q4_K_M-style quantization
//! - runtime AVX2/NEON quantized kernels with scalar reference fallback
//! - separate architecture/backend traits and measured prefill/decode
//! - GQA, split-half RoPE, RMSNorm, SwiGLU and a preallocated KV cache

pub mod activation;
pub mod architecture;
pub mod backend;
pub mod benchmark;
pub mod cache;
pub mod config;
pub mod engine;
pub mod gguf;
pub mod loader;
pub mod model;
pub mod quant;
pub mod track;
pub(crate) mod simd;

pub use architecture::{ArchitectureKind, CausalModel, LogitsMode};
pub use benchmark::{GenerationOutput, InferenceMetrics};
pub use engine::{EngineOptions, InferenceEngine};
pub use loader::{LoadOptions, LoadQuantization, ModelLoader};
pub use model::Qwen2Model;
