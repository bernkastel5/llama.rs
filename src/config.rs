use anyhow::{bail, Result};
use serde::Deserialize;

fn default_max_position_embeddings() -> usize {
    32_768
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_rope_theta() -> f32 {
    10_000.0
}

/// The subset of the Hugging Face Qwen2/Qwen2.5 configuration needed by inference.
/// Unknown fields in config.json are intentionally ignored by serde.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// Some newer Qwen configs make this explicit. If absent, hidden_size / heads is used.
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl LlamaConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)?;
        config.validate()?;
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn kv_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.num_hidden_layers == 0
            || self.vocab_size == 0
        {
            bail!("invalid Qwen2 config: dimensions must be non-zero");
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) && self.head_dim.is_none() {
            bail!("hidden_size must be divisible by num_attention_heads");
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            bail!("num_attention_heads must be divisible by num_key_value_heads");
        }
        if self.num_attention_heads * self.head_dim() != self.hidden_size {
            bail!("num_attention_heads * head_dim must equal hidden_size");
        }
        if !self.head_dim().is_multiple_of(2) {
            bail!("Qwen2 RoPE requires an even head_dim");
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            bail!("rms_norm_eps must be finite and positive");
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            bail!("rope_theta must be finite and positive");
        }
        Ok(())
    }
}
