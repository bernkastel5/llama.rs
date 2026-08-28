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
            crate::simd::vec_add_f32(output, bias);
        }
        Ok(())
    }

    pub fn forward_quant(
        &self,
        backend: &dyn KernelBackend,
        input: &[f32],
        quant_act: &crate::activation::QuantizedActivation,
        output: &mut [f32],
    ) -> Result<()> {
        self.weight
            .matvec_quantized_with(backend, input, quant_act, output)?;
        if let Some(bias) = &self.bias {
            crate::simd::vec_add_f32(output, bias);
        }
        Ok(())
    }

    pub fn forward_batch(
        &self,
        backend: &dyn KernelBackend,
        inputs: &[f32],
        outputs: &mut [f32],
        batch_size: usize,
    ) -> Result<()> {
        if batch_size == 1 {
            return self.forward(backend, inputs, outputs);
        }
        self.weight
            .matvec_batch_with(backend, inputs, outputs, batch_size)?;
        if let Some(bias) = &self.bias {
            let out_len = self.out_features;
            for b in 0..batch_size {
                let out_b = &mut outputs[b * out_len..(b + 1) * out_len];
                crate::simd::vec_add_f32(out_b, bias);
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
        crate::simd::vec_rmsnorm(hidden, &self.weight, self.eps);
        Ok(())
    }
}

fn prepare_rope(pos: usize, inv_freq: &[f32], cos: &mut [f32], sin: &mut [f32]) {
    for (i, &freq) in inv_freq.iter().enumerate() {
        let angle = pos as f32 * freq;
        let (s, c) = angle.sin_cos();
        cos[i] = c;
        sin[i] = s;
    }
}

/// Qwen2 uses the non-interleaved (split-half) RoPE convention.
fn apply_rope_inplace(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    crate::simd::vec_rope_inplace(x, cos, sin);
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
    pub inv_freq: Vec<f32>,
}

/// Backward-compatible name retained for the original public API.
pub type LlamaModel = Qwen2Model;

impl Qwen2Model {
    pub fn new(
        embed_tokens: QuantTensor,
        layers: Vec<TransformerBlock>,
        norm: RMSNorm,
        lm_head: LinearLayer,
        config: LlamaConfig,
        backend: Arc<dyn KernelBackend>,
    ) -> Self {
        let head_dim = config.head_dim();
        let half_dim = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            inv_freq.push(1.0 / config.rope_theta.powf(2.0 * i as f32 / head_dim as f32));
        }
        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
            backend,
            inv_freq,
        }
    }

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

        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let kv_width = cfg.kv_width();
        let groups = num_heads / num_kv_heads;
        let attention_scale = 1.0 / (head_dim as f32).sqrt();
        prepare_rope(
            pos,
            &self.inv_freq,
            &mut buffers.rope_cos,
            &mut buffers.rope_sin,
        );

        self.embed_tokens
            .row_into(token_id as usize, &mut buffers.residual)?;

        for (layer_idx, block) in self.layers.iter().enumerate() {
            // Pre-attention norm.
            buffers.hidden_states.copy_from_slice(&buffers.residual);
            block.input_layernorm.forward(&mut buffers.hidden_states)?;

            // Quantize hidden_states ONCE for self_attn_q, self_attn_k, self_attn_v
            buffers
                .quant_act
                .quantize_for_tensor(&block.self_attn_q.weight, &buffers.hidden_states);

            block.self_attn_q.forward_quant(
                backend,
                &buffers.hidden_states,
                &buffers.quant_act,
                &mut buffers.q,
            )?;
            block.self_attn_k.forward_quant(
                backend,
                &buffers.hidden_states,
                &buffers.quant_act,
                &mut buffers.k,
            )?;
            block.self_attn_v.forward_quant(
                backend,
                &buffers.hidden_states,
                &buffers.quant_act,
                &mut buffers.v,
            )?;

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

            let max_seq_len = kv_cache.max_seq_len;
            let q_slice = &buffers.q;
            let k_slice = &kv_cache.k_cache[layer_idx];
            let v_slice = &kv_cache.v_cache[layer_idx];
            let attn_ptr = buffers.attention.as_mut_ptr() as usize;
            let scores_ptr = buffers.scores.as_mut_ptr() as usize;

            if backend.threads() > 1 && num_heads >= 2 {
                backend.execute(&|thread_idx, num_threads| {
                    let start_head = thread_idx * num_heads / num_threads;
                    let end_head = ((thread_idx + 1) * num_heads / num_threads).min(num_heads);
                    for head in start_head..end_head {
                        let kv_head = head / groups;
                        let q_start = head * head_dim;
                        let q_head = &q_slice[q_start..q_start + head_dim];

                        let scores = unsafe {
                            std::slice::from_raw_parts_mut(
                                (scores_ptr as *mut f32).add(head * max_seq_len),
                                pos + 1,
                            )
                        };

                        for (past_pos, score) in scores.iter_mut().enumerate() {
                            let key_start = past_pos * kv_width + kv_head * head_dim;
                            let key = &k_slice[key_start..key_start + head_dim];
                            *score = crate::simd::vec_dot_f32(q_head, key) * attention_scale;
                        }
                        softmax_inplace(scores);

                        let head_out = unsafe {
                            std::slice::from_raw_parts_mut(
                                (attn_ptr as *mut f32).add(head * head_dim),
                                head_dim,
                            )
                        };
                        for (past_pos, &weight) in scores.iter().enumerate() {
                            let value_start = past_pos * kv_width + kv_head * head_dim;
                            let value = &v_slice[value_start..value_start + head_dim];
                            crate::simd::vec_mad_f32(head_out, value, weight);
                        }
                    }
                });
            } else {
                for head in 0..num_heads {
                    let kv_head = head / groups;
                    let q_start = head * head_dim;
                    let q_head = &q_slice[q_start..q_start + head_dim];
                    let scores =
                        &mut buffers.scores[head * max_seq_len..head * max_seq_len + pos + 1];

                    for (past_pos, score) in scores.iter_mut().enumerate() {
                        let key_start = past_pos * kv_width + kv_head * head_dim;
                        let key = &k_slice[key_start..key_start + head_dim];
                        *score = crate::simd::vec_dot_f32(q_head, key) * attention_scale;
                    }
                    softmax_inplace(scores);

                    let head_out = &mut buffers.attention[head * head_dim..(head + 1) * head_dim];
                    for (past_pos, &weight) in scores.iter().enumerate() {
                        let value_start = past_pos * kv_width + kv_head * head_dim;
                        let value = &v_slice[value_start..value_start + head_dim];
                        crate::simd::vec_mad_f32(head_out, value, weight);
                    }
                }
            }

            buffers
                .quant_act
                .quantize_for_tensor(&block.self_attn_o.weight, &buffers.attention);
            block.self_attn_o.forward_quant(
                backend,
                &buffers.attention,
                &buffers.quant_act,
                &mut buffers.projection,
            )?;
            crate::simd::vec_add_f32(&mut buffers.residual, &buffers.projection);

            // Post-attention norm and SwiGLU: silu(gate) * up.
            buffers.hidden_states.copy_from_slice(&buffers.residual);
            block
                .post_attention_layernorm
                .forward(&mut buffers.hidden_states)?;

            // Quantize hidden_states ONCE for mlp_gate and mlp_up
            buffers
                .quant_act
                .quantize_for_tensor(&block.mlp_gate.weight, &buffers.hidden_states);

            block.mlp_gate.forward_quant(
                backend,
                &buffers.hidden_states,
                &buffers.quant_act,
                &mut buffers.gate,
            )?;
            block.mlp_up.forward_quant(
                backend,
                &buffers.hidden_states,
                &buffers.quant_act,
                &mut buffers.up,
            )?;
            crate::simd::vec_swiglu(&mut buffers.gate, &buffers.up);

            buffers
                .quant_act
                .quantize_for_tensor(&block.mlp_down.weight, &buffers.gate);
            block.mlp_down.forward_quant(
                backend,
                &buffers.gate,
                &buffers.quant_act,
                &mut buffers.down,
            )?;
            crate::simd::vec_add_f32(&mut buffers.residual, &buffers.down);
        }

        if logits == LogitsMode::Compute {
            buffers.hidden_states.copy_from_slice(&buffers.residual);
            self.norm.forward(&mut buffers.hidden_states)?;
            buffers
                .quant_act
                .quantize_for_tensor(&self.lm_head.weight, &buffers.hidden_states);
            self.lm_head.forward_quant(
                backend,
                &buffers.hidden_states,
                &buffers.quant_act,
                &mut buffers.logits,
            )?;
        }
        Ok(())
    }

    fn prefill_chunk(
        &self,
        chunk: &[u32],
        start_pos: usize,
        kv_cache: &mut KVCache,
        buffers: &mut InferenceBuffers,
        is_last_chunk: bool,
    ) -> Result<()> {
        let b = chunk.len();
        if b == 0 {
            return Ok(());
        }
        if b == 1 {
            let mode = if is_last_chunk {
                LogitsMode::Compute
            } else {
                LogitsMode::Skip
            };
            return self.forward_impl(chunk[0], start_pos, kv_cache, buffers, mode);
        }

        let cfg = &self.config;
        let backend = self.backend.as_ref();
        let hidden_dim = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let kv_width = cfg.kv_width();
        let intermediate_size = cfg.intermediate_size;
        let groups = num_heads / num_kv_heads;
        let attention_scale = 1.0 / (head_dim as f32).sqrt();

        let mut batch_residual = vec![0.0f32; b * hidden_dim];
        let mut batch_hidden_states = vec![0.0f32; b * hidden_dim];
        let mut batch_q = vec![0.0f32; b * hidden_dim];
        let mut batch_k = vec![0.0f32; b * kv_width];
        let mut batch_v = vec![0.0f32; b * kv_width];
        let mut batch_attention = vec![0.0f32; b * hidden_dim];
        let mut batch_proj = vec![0.0f32; b * hidden_dim];
        let mut batch_gate = vec![0.0f32; b * intermediate_size];
        let mut batch_up = vec![0.0f32; b * intermediate_size];
        let mut batch_down = vec![0.0f32; b * hidden_dim];

        let mut rope_cos = vec![0.0f32; head_dim / 2];
        let mut rope_sin = vec![0.0f32; head_dim / 2];
        let mut scores = vec![0.0f32; kv_cache.max_seq_len];

        for (i, &token_id) in chunk.iter().enumerate() {
            ensure!(
                (token_id as usize) < cfg.vocab_size,
                "token id {token_id} is outside vocabulary"
            );
            let dst = &mut batch_residual[i * hidden_dim..(i + 1) * hidden_dim];
            self.embed_tokens.row_into(token_id as usize, dst)?;
        }

        for (layer_idx, block) in self.layers.iter().enumerate() {
            batch_hidden_states.copy_from_slice(&batch_residual);
            for i in 0..b {
                let h = &mut batch_hidden_states[i * hidden_dim..(i + 1) * hidden_dim];
                block.input_layernorm.forward(h)?;
            }

            block
                .self_attn_q
                .forward_batch(backend, &batch_hidden_states, &mut batch_q, b)?;
            block
                .self_attn_k
                .forward_batch(backend, &batch_hidden_states, &mut batch_k, b)?;
            block
                .self_attn_v
                .forward_batch(backend, &batch_hidden_states, &mut batch_v, b)?;

            for i in 0..b {
                let pos = start_pos + i;
                ensure!(
                    pos < kv_cache.max_seq_len,
                    "position {pos} exceeds configured context {}",
                    kv_cache.max_seq_len
                );
                prepare_rope(pos, &self.inv_freq, &mut rope_cos, &mut rope_sin);

                let q_i = &mut batch_q[i * hidden_dim..(i + 1) * hidden_dim];
                for head in 0..num_heads {
                    let start = head * head_dim;
                    apply_rope_inplace(
                        &mut q_i[start..start + head_dim],
                        &rope_cos,
                        &rope_sin,
                    );
                }

                let k_i = &mut batch_k[i * kv_width..(i + 1) * kv_width];
                for head in 0..num_kv_heads {
                    let start = head * head_dim;
                    apply_rope_inplace(
                        &mut k_i[start..start + head_dim],
                        &rope_cos,
                        &rope_sin,
                    );
                }

                let v_i = &batch_v[i * kv_width..(i + 1) * kv_width];
                kv_cache.update(layer_idx, pos, k_i, v_i)?;
            }

            batch_attention.fill(0.0);
            for i in 0..b {
                let pos = start_pos + i;
                let q_i = &batch_q[i * hidden_dim..(i + 1) * hidden_dim];
                let attn_i = &mut batch_attention[i * hidden_dim..(i + 1) * hidden_dim];

                for head in 0..num_heads {
                    let kv_head = head / groups;
                    let q_start = head * head_dim;
                    let q_head = &q_i[q_start..q_start + head_dim];
                    let scores_slice = &mut scores[..=pos];

                    for (past_pos, score) in scores_slice.iter_mut().enumerate() {
                        let key_start = past_pos * kv_width + kv_head * head_dim;
                        let key = &kv_cache.k_cache[layer_idx][key_start..key_start + head_dim];
                        *score = crate::simd::vec_dot_f32(q_head, key) * attention_scale;
                    }
                    softmax_inplace(scores_slice);

                    let out_start = head * head_dim;
                    let head_out = &mut attn_i[out_start..out_start + head_dim];
                    for (past_pos, &weight) in scores_slice.iter().enumerate() {
                        let value_start = past_pos * kv_width + kv_head * head_dim;
                        let value =
                            &kv_cache.v_cache[layer_idx][value_start..value_start + head_dim];
                        crate::simd::vec_mad_f32(head_out, value, weight);
                    }
                }
            }

            block
                .self_attn_o
                .forward_batch(backend, &batch_attention, &mut batch_proj, b)?;
            for i in 0..b {
                let res = &mut batch_residual[i * hidden_dim..(i + 1) * hidden_dim];
                let proj = &batch_proj[i * hidden_dim..(i + 1) * hidden_dim];
                crate::simd::vec_add_f32(res, proj);
            }

            batch_hidden_states.copy_from_slice(&batch_residual);
            for i in 0..b {
                let h = &mut batch_hidden_states[i * hidden_dim..(i + 1) * hidden_dim];
                block.post_attention_layernorm.forward(h)?;
            }

            block
                .mlp_gate
                .forward_batch(backend, &batch_hidden_states, &mut batch_gate, b)?;
            block
                .mlp_up
                .forward_batch(backend, &batch_hidden_states, &mut batch_up, b)?;
            for i in 0..b {
                let gate = &mut batch_gate[i * intermediate_size..(i + 1) * intermediate_size];
                let up = &batch_up[i * intermediate_size..(i + 1) * intermediate_size];
                crate::simd::vec_swiglu(gate, up);
            }

            block
                .mlp_down
                .forward_batch(backend, &batch_gate, &mut batch_down, b)?;
            for i in 0..b {
                let res = &mut batch_residual[i * hidden_dim..(i + 1) * hidden_dim];
                let down = &batch_down[i * hidden_dim..(i + 1) * hidden_dim];
                crate::simd::vec_add_f32(res, down);
            }
        }

        let last_idx = b - 1;
        let last_res = &batch_residual[last_idx * hidden_dim..b * hidden_dim];
        buffers.residual.copy_from_slice(last_res);

        if is_last_chunk {
            buffers.hidden_states.copy_from_slice(last_res);
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

    fn prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        kv_cache: &mut KVCache,
        buffers: &mut InferenceBuffers,
    ) -> Result<()> {
        ensure!(!tokens.is_empty(), "prefill requires at least one token");
        const CHUNK_SIZE: usize = 32;
        let total = tokens.len();
        let mut current_pos = start_pos;
        for chunk in tokens.chunks(CHUNK_SIZE) {
            let is_last = current_pos + chunk.len() == start_pos + total;
            self.prefill_chunk(chunk, current_pos, kv_cache, buffers, is_last)?;
            current_pos += chunk.len();
        }
        Ok(())
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
        let inv_sum = 1.0 / sum;
        for value in values {
            *value *= inv_sum;
        }
    } else {
        let uniform = 1.0 / values.len() as f32;
        values.fill(uniform);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendConfig, CpuBackend};

    #[test]
    fn linear_layer_forward_batch_matches_single_forward() {
        let backend =
            Arc::new(CpuBackend::new(&BackendConfig::reference()).unwrap()) as Arc<dyn KernelBackend>;
        let in_features = 64;
        let out_features = 32;
        let batch_size = 3;

        let weight_data: Vec<f32> = (0..out_features * in_features)
            .map(|i| (i as f32 * 0.03).sin())
            .collect();
        let bias_data: Vec<f32> = (0..out_features).map(|i| i as f32 * 0.1).collect();
        let tensor =
            QuantTensor::quantize_q8_0(&weight_data, out_features, in_features).unwrap();
        let linear = LinearLayer::new(tensor, Some(bias_data)).unwrap();

        let inputs: Vec<f32> = (0..batch_size * in_features)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();

        let mut batched_out = vec![0.0f32; batch_size * out_features];
        linear
            .forward_batch(backend.as_ref(), &inputs, &mut batched_out, batch_size)
            .unwrap();

        for b in 0..batch_size {
            let in_b = &inputs[b * in_features..(b + 1) * in_features];
            let mut single_out = vec![0.0f32; out_features];
            linear
                .forward(backend.as_ref(), in_b, &mut single_out)
                .unwrap();
            let b_slice = &batched_out[b * out_features..(b + 1) * out_features];
            for (s, batch_v) in single_out.iter().zip(b_slice) {
                assert!((s - batch_v).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn qwen2_prefill_matches_token_by_token_forward() {
        let backend =
            Arc::new(CpuBackend::new(&BackendConfig::reference()).unwrap()) as Arc<dyn KernelBackend>;

        let hidden = 64;
        let heads = 4;
        let kv_heads = 2;
        let kv_w = 32;
        let intermediate = 128;
        let vocab = 50;
        let max_ctx = 64;

        let config = LlamaConfig {
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_attention_heads: heads,
            num_key_value_heads: kv_heads,
            num_hidden_layers: 1,
            vocab_size: vocab,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: max_ctx,
            head_dim: None,
            tie_word_embeddings: false,
        };

        let embed_data: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32 * 0.02).sin()).collect();
        let embed = QuantTensor::quantize_q8_0(&embed_data, vocab, hidden).unwrap();

        let norm1 = RMSNorm::new(vec![1.0; hidden], 1e-5).unwrap();
        let q_data: Vec<f32> = (0..hidden * hidden).map(|i| (i as f32 * 0.01).cos()).collect();
        let k_data: Vec<f32> = (0..kv_w * hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        let v_data: Vec<f32> = (0..kv_w * hidden).map(|i| (i as f32 * 0.02).cos()).collect();
        let o_data: Vec<f32> = (0..hidden * hidden).map(|i| (i as f32 * 0.02).sin()).collect();

        let q_proj = LinearLayer::new(QuantTensor::quantize_q8_0(&q_data, hidden, hidden).unwrap(), None).unwrap();
        let k_proj = LinearLayer::new(QuantTensor::quantize_q8_0(&k_data, kv_w, hidden).unwrap(), None).unwrap();
        let v_proj = LinearLayer::new(QuantTensor::quantize_q8_0(&v_data, kv_w, hidden).unwrap(), None).unwrap();
        let o_proj = LinearLayer::new(QuantTensor::quantize_q8_0(&o_data, hidden, hidden).unwrap(), None).unwrap();

        let norm2 = RMSNorm::new(vec![1.0; hidden], 1e-5).unwrap();
        let gate_data: Vec<f32> = (0..intermediate * hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        let up_data: Vec<f32> = (0..intermediate * hidden).map(|i| (i as f32 * 0.01).cos()).collect();
        let down_data: Vec<f32> = (0..hidden * intermediate).map(|i| (i as f32 * 0.02).sin()).collect();

        let gate_proj = LinearLayer::new(QuantTensor::quantize_q8_0(&gate_data, intermediate, hidden).unwrap(), None).unwrap();
        let up_proj = LinearLayer::new(QuantTensor::quantize_q8_0(&up_data, intermediate, hidden).unwrap(), None).unwrap();
        let down_proj = LinearLayer::new(QuantTensor::quantize_q8_0(&down_data, hidden, intermediate).unwrap(), None).unwrap();

        let block = TransformerBlock {
            input_layernorm: norm1,
            self_attn_q: q_proj,
            self_attn_k: k_proj,
            self_attn_v: v_proj,
            self_attn_o: o_proj,
            post_attention_layernorm: norm2,
            mlp_gate: gate_proj,
            mlp_up: up_proj,
            mlp_down: down_proj,
        };

        let norm = RMSNorm::new(vec![1.0; hidden], 1e-5).unwrap();
        let lm_head_data: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32 * 0.03).cos()).collect();
        let lm_head = LinearLayer::new(QuantTensor::quantize_q8_0(&lm_head_data, vocab, hidden).unwrap(), None).unwrap();

        let model = Qwen2Model::new(
            embed,
            vec![block],
            norm,
            lm_head,
            config.clone(),
            backend,
        );

        let tokens = vec![3u32, 12, 45, 7];

        // 1. Sequential token-by-token
        let mut kv_seq = KVCache::new(&config, max_ctx).unwrap();
        let mut buf_seq = InferenceBuffers::new(&config, max_ctx);
        for (i, &t) in tokens.iter().enumerate() {
            let mode = if i + 1 == tokens.len() {
                LogitsMode::Compute
            } else {
                LogitsMode::Skip
            };
            model
                .forward_token(t, i, &mut kv_seq, &mut buf_seq, mode)
                .unwrap();
        }

        // 2. Batched prefill
        let mut kv_batch = KVCache::new(&config, max_ctx).unwrap();
        let mut buf_batch = InferenceBuffers::new(&config, max_ctx);
        model
            .prefill(&tokens, 0, &mut kv_batch, &mut buf_batch)
            .unwrap();

        // Compare KV cache
        let k_seq = &kv_seq.k_cache[0][..tokens.len() * kv_w];
        let k_batch = &kv_batch.k_cache[0][..tokens.len() * kv_w];
        for (a, b) in k_seq.iter().zip(k_batch) {
            assert!((a - b).abs() < 1e-4, "KV K mismatch: {a} vs {b}");
        }

        let v_seq = &kv_seq.v_cache[0][..tokens.len() * kv_w];
        let v_batch = &kv_batch.v_cache[0][..tokens.len() * kv_w];
        for (a, b) in v_seq.iter().zip(v_batch) {
            assert!((a - b).abs() < 1e-4, "KV V mismatch: {a} vs {b}");
        }

        // Compare final logits
        for (i, (s, b)) in buf_seq.logits.iter().zip(&buf_batch.logits).enumerate() {
            assert!((s - b).abs() < 1e-3, "Logits[{i}] mismatch: {s} vs {b}");
        }
    }
}
