use crate::cache::{InferenceBuffers, KVCache};
use crate::config::LlamaConfig;
use crate::model::LlamaModel;
use anyhow::Result;
use tokenizers::Tokenizer;

pub struct InferenceEngine {
    pub model: LlamaModel,
    pub tokenizer: Tokenizer,
    pub kv_cache: KVCache,
    pub buffers: InferenceBuffers,
}

impl InferenceEngine {
    pub fn new(model_dir: &str) -> Result<Self> {
        let model = crate::loader::ModelLoader::load(model_dir)?;
        let tokenizer_path = format!("{}/tokenizer.json", model_dir);
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

        let config = &model.config;
        let head_dim = config.hidden_size / config.num_attention_heads;
        let kv_cache = KVCache::new(
            config.num_hidden_layers,
            config.max_position_embeddings,
            config.num_key_value_heads,
            head_dim,
        );
        let buffers = InferenceBuffers::new(config.hidden_size, config.vocab_size);

        Ok(Self { model, tokenizer, kv_cache, buffers })
    }

    pub fn forward_step(&mut self, token_id: u32, _pos: usize) -> u32 {
        // Placeholder forward: return a dummy next token
        // Full implementation would use model + cache + buffers
        (token_id + 1) % self.model.config.vocab_size as u32
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let encoding = self.tokenizer.encode(prompt, false).map_err(|e| anyhow::anyhow!(e))?;
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

        for i in 0..max_tokens {
            let next = self.forward_step(tokens.last().copied().unwrap_or(0), tokens.len());
            tokens.push(next);
            if next == 2 { break; } // EOS placeholder
        }

        let decoded = self.tokenizer.decode(&tokens, true).map_err(|e| anyhow::anyhow!(e))?;
        Ok(decoded)
    }
}