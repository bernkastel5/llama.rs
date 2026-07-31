use crate::backend::KernelBackend;
use crate::cache::{InferenceBuffers, KVCache};
use crate::config::LlamaConfig;
use anyhow::{ensure, Result};
use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureKind {
    Qwen2,
}

impl ArchitectureKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Qwen2 => "qwen2",
        }
    }

    pub fn from_hf_model_type(value: &str) -> Result<Self> {
        match value {
            "qwen2" => Ok(Self::Qwen2),
            other => anyhow::bail!("unsupported Hugging Face model_type {other:?}"),
        }
    }

    pub fn from_gguf_architecture(value: &str) -> Result<Self> {
        match value {
            "qwen2" => Ok(Self::Qwen2),
            other => anyhow::bail!("unsupported GGUF architecture {other:?}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogitsMode {
    Skip,
    Compute,
}

/// Architecture boundary used by the engine. New model families implement this
/// trait while reusing the same tensor storage, CPU backend, KV cache, and sampler.
pub trait CausalModel: Send + Sync + Debug {
    fn architecture(&self) -> ArchitectureKind;
    fn config(&self) -> &LlamaConfig;
    fn backend(&self) -> &dyn KernelBackend;

    fn forward_token(
        &self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut KVCache,
        buffers: &mut InferenceBuffers,
        logits: LogitsMode,
    ) -> Result<()>;

    /// Default causal prefill. Intermediate vocabulary projections are skipped;
    /// only the final prompt token computes logits. This is a major latency win
    /// for large vocabularies and leaves room for a future architecture-specific
    /// batched GEMM implementation without changing the engine API.
    fn prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        kv_cache: &mut KVCache,
        buffers: &mut InferenceBuffers,
    ) -> Result<()> {
        ensure!(!tokens.is_empty(), "prefill requires at least one token");
        for (index, &token) in tokens.iter().enumerate() {
            let mode = if index + 1 == tokens.len() {
                LogitsMode::Compute
            } else {
                LogitsMode::Skip
            };
            self.forward_token(token, start_pos + index, kv_cache, buffers, mode)?;
        }
        Ok(())
    }
}
