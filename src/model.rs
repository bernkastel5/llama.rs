use crate::architecture::{ArchitectureKind, CausalModel, LogitsMode};
use crate::backend::KernelBackend;
use crate::cache::{InferenceBuffers, KVCache};
use crate::config::LlamaConfig;
use crate::quant::QuantTensor;
use anyhow::{ensure, Result};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weight: QuantTensor,
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl LinearLayer {
    pub fn new(weight: QuantTensor, bias: Option<Vec<f32>>) -> Result<Self> {
        ensure!(weight.shape.len() == 2, "linear weight must be 2D");
        let out_features = weight.shape[0];
        let in_features = weight.shape[1];
        if let Some(bias) = &bias {
            ensure!(bias.len() == out_features, "linear bias length mismatch");
        }
        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    pub fn forward(
        &self,
        backend: &dyn KernelBackend,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        self.weight.matvec_with(backend, input, output)?;
        if let Some(bias) = &self.bias {
            for (value, bias) in output.iter_mut().zip(bias) {
                *value += *bias;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct RMSNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(weight: Vec<f32>, eps: f32) -> Result<Self> {
        ensure!(!weight.is_empty(), "RMSNorm weight cannot be empty");
        ensure!(
            eps.is_finite() && eps > 0.0,
            "RMSNorm epsilon must be positive"
        );
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, hidden: &mut [f32]) -> Result<()> {
        ensure!(hidden.len() == self.weight.len(), "RMSNorm shape mismatch");
        let mean_sq = hidden.iter().map(|x| x * x).sum::<f32>() / hidden.len() as f32;
        let inv_rms = 1.0 / (mean_sq + self.eps).sqrt();
        for (value, &weight) in hidden.iter_mut().zip(&self.weight) {
            *value *= inv_rms * weight;
        }
        Ok(())
    }
}

fn prepare_rope(pos: usize, head_dim: usize, rope_theta: f32, cos: &mut [f32], sin: &mut [f32]) {
    for i in 0..head_dim / 2 {
        let inv_freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32);
        let angle = pos as f32 * inv_freq;
        let (s, c) = angle.sin_cos();
        cos[i] = c;
        sin[i] = s;
    }
}

/// Qwen2 uses the non-interleaved (split-half) RoPE convention.
fn apply_rope_inplace(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = x.len() / 2;
    for i in 0..half {
        let real = x[i];
        let imag = x[i + half];
        x[i] = real * cos[i] - imag * sin[i];
        x[i + half] = real * sin[i] + imag * cos[i];
    }
}

#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub input_layernorm: RMSNorm,
    pub self_attn_q: LinearLayer,
    pub self_attn_k: LinearLayer,
    pub self_attn_v: LinearLayer,
    pub self_attn_o: LinearLayer,
    pub post_attention_layernorm: RMSNorm,
    pub mlp_gate: LinearLayer,
    pub mlp_up: LinearLayer,
    pub mlp_down: LinearLayer,
}

#[derive(Debug, Clone)]
pub struct Qwen2Model {
    pub embed_tokens: QuantTensor,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: LinearLayer,
    pub config: LlamaConfig,
    pub backend: Arc<dyn KernelBackend>,
}

/// Backward-compatible name retained for the original public API.
pub type LlamaModel = Qwen2Model;

impl Qwen2Model {
    /// Compatibility wrapper that computes vocabulary logits.
    pub fn forward(
        &self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut KVCache,
        buffers: &mut InferenceBuffers,
    ) -> Result<()> {
        self.forward_impl(token_id, pos, kv_cache, buffers, LogitsMode::Compute)
    }

    fn forward_impl(
        &self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut KVCache,
        buffers: &mut InferenceBuffers,
        logits: LogitsMode,
    ) -> Result<()> {
        let cfg = &self.config;
        let backend = self.backend.as_ref();
        ensure!(
            (token_id as usize) < cfg.vocab_size,
            "token id {token_id} is outside vocabulary"
        );
        ensure!(
            pos < kv_cache.max_seq_len,
            "position {pos} exceeds configured context {}",
            kv_cache.max_seq_len
        );

        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let kv_width = cfg.kv_width();
        let groups = num_heads / num_kv_heads;
        let attention_scale = 1.0 / (head_dim as f32).sqrt();
        prepare_rope(
            pos,
            head_dim,
            cfg.rope_theta,
            &mut buffers.rope_cos,
            &mut buffers.rope_sin,
        );

        self.embed_tokens
            .row_into(token_id as usize, &mut buffers.residual)?;

        for (layer_idx, block) in self.layers.iter().enumerate() {
            // Pre-attention norm.
            buffers.hidden_states.copy_from_slice(&buffers.residual);
            block.input_layernorm.forward(&mut buffers.hidden_states)?;

            block
                .self_attn_q
                .forward(backend, &buffers.hidden_states, &mut buffers.q)?;
            block
                .self_attn_k
                .forward(backend, &buffers.hidden_states, &mut buffers.k)?;
            block
                .self_attn_v
                .forward(backend, &buffers.hidden_states, &mut buffers.v)?;

            for head in 0..num_heads {
                let start = head * head_dim;
                apply_rope_inplace(
                    &mut buffers.q[start..start + head_dim],
                    &buffers.rope_cos,
                    &buffers.rope_sin,
                );
            }
            for head in 0..num_kv_heads {
                let start = head * head_dim;
                apply_rope_inplace(
                    &mut buffers.k[start..start + head_dim],
                    &buffers.rope_cos,
                    &buffers.rope_sin,
                );
            }

            kv_cache.update(layer_idx, pos, &buffers.k, &buffers.v)?;
            buffers.attention.fill(0.0);

            for head in 0..num_heads {
                let kv_head = head / groups;
                let q_start = head * head_dim;
                let q_head = &buffers.q[q_start..q_start + head_dim];
                let scores = &mut buffers.scores[..=pos];

                for (past_pos, score) in scores.iter_mut().enumerate() {
                    let key_start = past_pos * kv_width + kv_head * head_dim;
                    let key = &kv_cache.k_cache[layer_idx][key_start..key_start + head_dim];
                    *score =
                        q_head.iter().zip(key).map(|(q, k)| q * k).sum::<f32>() * attention_scale;
                }
                softmax_inplace(scores);

                let out_start = head * head_dim;
                let head_out = &mut buffers.attention[out_start..out_start + head_dim];
                for (past_pos, &weight) in scores.iter().enumerate() {
                    let value_start = past_pos * kv_width + kv_head * head_dim;
                    let value = &kv_cache.v_cache[layer_idx][value_start..value_start + head_dim];
                    for (out, &v) in head_out.iter_mut().zip(value) {
                        *out += weight * v;
                    }
                }
            }

            block
                .self_attn_o
                .forward(backend, &buffers.attention, &mut buffers.projection)?;
            for i in 0..hidden {
                buffers.residual[i] += buffers.projection[i];
            }

            // Post-attention norm and SwiGLU: silu(gate) * up.
            buffers.hidden_states.copy_from_slice(&buffers.residual);
            block
                .post_attention_layernorm
                .forward(&mut buffers.hidden_states)?;
            block
                .mlp_gate
                .forward(backend, &buffers.hidden_states, &mut buffers.gate)?;
            block
                .mlp_up
                .forward(backend, &buffers.hidden_states, &mut buffers.up)?;
            for i in 0..cfg.intermediate_size {
                let gate = buffers.gate[i];
                buffers.gate[i] = (gate / (1.0 + (-gate).exp())) * buffers.up[i];
            }
            block
                .mlp_down
                .forward(backend, &buffers.gate, &mut buffers.down)?;
            for i in 0..hidden {
                buffers.residual[i] += buffers.down[i];
            }
        }

        if logits == LogitsMode::Compute {
            buffers.hidden_states.copy_from_slice(&buffers.residual);
            self.norm.forward(&mut buffers.hidden_states)?;
            self.lm_head
                .forward(backend, &buffers.hidden_states, &mut buffers.logits)?;
        }
        Ok(())
    }
}

impl CausalModel for Qwen2Model {
    fn architecture(&self) -> ArchitectureKind {
        ArchitectureKind::Qwen2
    }

    fn config(&self) -> &LlamaConfig {
        &self.config
    }

    fn backend(&self) -> &dyn KernelBackend {
        self.backend.as_ref()
    }

    fn forward_token(
        &self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut KVCache,
        buffers: &mut InferenceBuffers,
        logits: LogitsMode,
    ) -> Result<()> {
        self.forward_impl(token_id, pos, kv_cache, buffers, logits)
    }
}

fn softmax_inplace(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum.is_finite() && sum > 0.0 {
        for value in values {
            *value /= sum;
        }
    } else {
        let uniform = 1.0 / values.len() as f32;
        values.fill(uniform);
    }
}
