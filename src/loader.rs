use crate::config::LlamaConfig;
use crate::model::{LinearLayer, LlamaModel, RMSNorm, TransformerBlock};
use crate::tensor::{quantize_fp16_to_int8, QuantTensor};
use anyhow::Result;
use half::f16;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::fs::File;

pub struct ModelLoader;

impl ModelLoader {
    pub fn load(model_dir: &str) -> Result<LlamaModel> {
        let config_path = format!("{}/config.json", model_dir);
        let config_json = std::fs::read_to_string(config_path)?;
        let config = LlamaConfig::from_json(&config_json)?;

        let safetensors_path = format!("{}/model.safetensors", model_dir);
        let file = File::open(safetensors_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let tensors = SafeTensors::deserialize(&mmap)?;

        // For demo, we create dummy quantized weights.
        // In real impl, iterate tensors, detect dtype, quantize.
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;

        // Dummy embed and lm_head
        let embed_tokens = vec![0.01f32; vocab_size * hidden_size];
        let lm_head_weight = QuantTensor::new(
            vec![128u8; hidden_size * vocab_size],
            vec![0.01],
            Some(vec![0.0]),
            vec![vocab_size, hidden_size],
            crate::tensor::QuantType::Int8,
        );
        let lm_head = LinearLayer::new(lm_head_weight, None);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            let ln_w = vec![1.0f32; hidden_size];
            let input_norm = RMSNorm::new(ln_w.clone(), config.rms_norm_eps);
            let post_norm = RMSNorm::new(ln_w, config.rms_norm_eps);

            let q_weight = quantize_fp16_to_int8(&vec![f16::from_f32(0.01); hidden_size * hidden_size]);
            let k_weight = quantize_fp16_to_int8(&vec![f16::from_f32(0.01); hidden_size * hidden_size]);
            let v_weight = quantize_fp16_to_int8(&vec![f16::from_f32(0.01); hidden_size * hidden_size]);
            let o_weight = quantize_fp16_to_int8(&vec![f16::from_f32(0.01); hidden_size * hidden_size]);

            let gate_w = quantize_fp16_to_int8(&vec![f16::from_f32(0.01); hidden_size * config.intermediate_size]);
            let up_w = quantize_fp16_to_int8(&vec![f16::from_f32(0.01); hidden_size * config.intermediate_size]);
            let down_w = quantize_fp16_to_int8(&vec![f16::from_f32(0.01); config.intermediate_size * hidden_size]);

            let block = TransformerBlock {
                input_layernorm: input_norm,
                self_attn_q: LinearLayer::new(q_weight, None),
                self_attn_k: LinearLayer::new(k_weight, None),
                self_attn_v: LinearLayer::new(v_weight, None),
                self_attn_o: LinearLayer::new(o_weight, None),
                post_attention_layernorm: post_norm,
                mlp_gate: LinearLayer::new(gate_w, None),
                mlp_up: LinearLayer::new(up_w, None),
                mlp_down: LinearLayer::new(down_w, None),
            };
            layers.push(block);
        }

        Ok(LlamaModel {
            embed_tokens,
            layers,
            norm: RMSNorm::new(vec![1.0; hidden_size], config.rms_norm_eps),
            lm_head,
            config,
        })
    }
}