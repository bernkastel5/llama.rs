use crate::config::LlamaConfig;
use anyhow::{ensure, Result};

#[derive(Debug)]
pub struct KVCache {
    pub k_cache: Vec<Vec<f32>>,
    pub v_cache: Vec<Vec<f32>>,
    pub max_seq_len: usize,
    kv_width: usize,
}

impl KVCache {
    pub fn new(config: &LlamaConfig, max_seq_len: usize) -> Result<Self> {
        ensure!(max_seq_len > 0, "max_seq_len must be positive");
        ensure!(
            max_seq_len <= config.max_position_embeddings,
            "requested context {max_seq_len} exceeds model limit {}",
            config.max_position_embeddings
        );
        let kv_width = config.kv_width();
        let layer_len = max_seq_len
            .checked_mul(kv_width)
            .ok_or_else(|| anyhow::anyhow!("KV cache size overflow"))?;
        let mut k_cache = Vec::with_capacity(config.num_hidden_layers);
        let mut v_cache = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            k_cache.push(vec![0.0; layer_len]);
            v_cache.push(vec![0.0; layer_len]);
        }
        Ok(Self {
            k_cache,
            v_cache,
            max_seq_len,
            kv_width,
        })
    }

    pub fn update(&mut self, layer: usize, pos: usize, key: &[f32], value: &[f32]) -> Result<()> {
        ensure!(layer < self.k_cache.len(), "KV layer out of bounds");
        ensure!(
            pos < self.max_seq_len,
            "position {pos} exceeds KV-cache context {}",
            self.max_seq_len
        );
        ensure!(
            key.len() == self.kv_width && value.len() == self.kv_width,
            "KV width mismatch"
        );
        let offset = pos * self.kv_width;
        self.k_cache[layer][offset..offset + self.kv_width].copy_from_slice(key);
        self.v_cache[layer][offset..offset + self.kv_width].copy_from_slice(value);
        Ok(())
    }

    pub fn reset(&mut self) {
        // Positions are overwritten before they are read, so clearing gigabytes is unnecessary.
    }
}

#[derive(Debug)]
pub struct InferenceBuffers {
    pub hidden_states: Vec<f32>,
    pub residual: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attention: Vec<f32>,
    pub projection: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
    pub scores: Vec<f32>,
    /// RoPE values for the current position, computed once and reused by every layer.
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
    pub logits: Vec<f32>,
}

impl InferenceBuffers {
    pub fn new(config: &LlamaConfig, max_seq_len: usize) -> Self {
        let hidden = config.hidden_size;
        let kv = config.kv_width();
        let intermediate = config.intermediate_size;
        let num_heads = config.num_attention_heads;
        Self {
            hidden_states: vec![0.0; hidden],
            residual: vec![0.0; hidden],
            q: vec![0.0; hidden],
            k: vec![0.0; kv],
            v: vec![0.0; kv],
            attention: vec![0.0; hidden],
            projection: vec![0.0; hidden],
            gate: vec![0.0; intermediate],
            up: vec![0.0; intermediate],
            down: vec![0.0; hidden],
            scores: vec![0.0; max_seq_len * num_heads],
            rope_cos: vec![0.0; config.head_dim() / 2],
            rope_sin: vec![0.0; config.head_dim() / 2],
            logits: vec![0.0; config.vocab_size],
        }
    }
}
