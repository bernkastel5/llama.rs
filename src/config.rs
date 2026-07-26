use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
}

fn default_max_position_embeddings() -> usize {
    2048
}

impl LlamaConfig {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let config: LlamaConfig = serde_json::from_str(json)?;
        Ok(config)
    }
}