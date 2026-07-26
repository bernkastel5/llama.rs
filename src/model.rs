use crate::tensor::{QuantTensor, QuantType};
use half::f16;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weight: QuantTensor,
    pub bias: Option<Vec<f32>>,
}

impl LinearLayer {
    pub fn new(weight: QuantTensor, bias: Option<Vec<f32>>) -> Self {
        Self { weight, bias }
    }

    /// Matrix-vector multiply with INT8 weights (dequant on the fly).
    /// TODO: Plug SIMD kernel here.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        let in_features = self.weight.shape[1];
        let out_features = self.weight.shape[0];

        assert_eq!(input.len(), in_features);
        assert_eq!(output.len(), out_features);

        // Simple dequant + matvec
        output.par_iter_mut().enumerate().for_each(|(i, out)| {
            let mut sum = 0.0f32;
            let scale = self.weight.scales[0];
            let zp = self.weight.zero_points.as_ref().map_or(0.0, |z| z[0]);

            for j in 0..in_features {
                let q = self.weight.data[i * in_features + j] as f32;
                let w = (q - zp) * scale;
                sum += w * input[j];
            }
            if let Some(b) = &self.bias {
                sum += b[i];
            }
            *out = sum;
        });
    }
}

#[derive(Debug, Clone)]
pub struct RMSNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, hidden: &mut [f32]) {
        let n = hidden.len();
        let mean_sq: f32 = hidden.iter().map(|x| x * x).sum::<f32>() / n as f32;
        let rms = (mean_sq + self.eps).sqrt();
        for (h, &w) in hidden.iter_mut().zip(self.weight.iter()) {
            *h = (*h / rms) * w;
        }
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
pub struct LlamaModel {
    pub embed_tokens: Vec<f32>, // vocab_size x hidden_size (dequantized for simplicity)
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub lm_head: LinearLayer,
    pub config: crate::config::LlamaConfig,
}