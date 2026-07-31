use crate::architecture::{CausalModel, LogitsMode};
use crate::benchmark::{GenerationOutput, InferenceMetrics};
use crate::cache::{InferenceBuffers, KVCache};
use crate::loader::{LoadOptions, ModelLoader};
use anyhow::{bail, ensure, Context, Result};
use rand::Rng;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub load: LoadOptions,
    pub context_length: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            load: LoadOptions::default(),
            context_length: 4096,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}

pub struct InferenceEngine {
    pub model: Box<dyn CausalModel>,
    pub tokenizer: Tokenizer,
    pub kv_cache: KVCache,
    pub buffers: InferenceBuffers,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub max_repeat_window: usize,
    pub eos_token_ids: Vec<u32>,
    recent_tokens: Vec<u32>,
    candidates: Vec<(usize, f32)>,
    seen_tokens: Vec<bool>,
}

impl InferenceEngine {
    pub fn new(model_path: &str) -> Result<Self> {
        Self::from_path(model_path, EngineOptions::default())
    }

    pub fn from_path(path: impl AsRef<Path>, options: EngineOptions) -> Result<Self> {
        validate_sampling(&options)?;
        let path = path.as_ref();
        let model = ModelLoader::load_dynamic(path, &options.load)?;
        let tokenizer_path = tokenizer_path(path);
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load {}: {e}", tokenizer_path.display()))?;
        let context_length = options
            .context_length
            .min(model.config().max_position_embeddings);
        ensure!(context_length > 0, "context length must be positive");
        let kv_cache = KVCache::new(model.config(), context_length)?;
        let buffers = InferenceBuffers::new(model.config(), context_length);
        let eos_token_ids = find_eos_tokens(path, &tokenizer)?;
        let vocab_size = model.config().vocab_size;

        Ok(Self {
            model,
            tokenizer,
            kv_cache,
            buffers,
            temperature: options.temperature,
            top_p: options.top_p,
            top_k: options.top_k,
            repetition_penalty: options.repetition_penalty,
            max_repeat_window: options.repeat_last_n,
            eos_token_ids,
            recent_tokens: Vec::with_capacity(options.repeat_last_n + 1),
            candidates: Vec::with_capacity(vocab_size),
            seen_tokens: vec![false; vocab_size],
        })
    }

    pub fn reset(&mut self) {
        self.kv_cache.reset();
        self.recent_tokens.clear();
    }

    /// Runs the model on one token at the specified position, then samples its successor.
    pub fn forward_step(&mut self, token_id: u32, pos: usize) -> Result<u32> {
        self.model.forward_token(
            token_id,
            pos,
            &mut self.kv_cache,
            &mut self.buffers,
            LogitsMode::Compute,
        )?;
        let next = self.sample_current_logits();
        self.remember(next);
        Ok(next)
    }

    /// Prefills every prompt token, then performs cached decode.
    /// The returned string contains only newly generated tokens.
    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize) -> Result<String> {
        Ok(self.generate_with_metrics(prompt, max_new_tokens)?.text)
    }

    pub fn generate_with_metrics(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<GenerationOutput> {
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode error: {e}"))?;
        let prompt_tokens = encoding.get_ids();
        ensure!(
            !prompt_tokens.is_empty(),
            "the prompt encoded to zero tokens"
        );
        ensure!(
            prompt_tokens.len() < self.kv_cache.max_seq_len,
            "prompt has {} tokens, but context is {}",
            prompt_tokens.len(),
            self.kv_cache.max_seq_len
        );

        self.reset();
        for &token in prompt_tokens
            .iter()
            .rev()
            .take(self.max_repeat_window)
            .rev()
        {
            self.remember(token);
        }

        let prefill_started = Instant::now();
        self.model
            .prefill(prompt_tokens, 0, &mut self.kv_cache, &mut self.buffers)?;
        let prefill = prefill_started.elapsed();

        let available = self.kv_cache.max_seq_len - prompt_tokens.len();
        let count = max_new_tokens.min(available);
        let mut generated = Vec::with_capacity(count);
        let decode_started = Instant::now();
        for _ in 0..count {
            let token = self.sample_current_logits();
            generated.push(token);
            self.remember(token);
            if self.eos_token_ids.contains(&token) {
                break;
            }
            let pos = prompt_tokens.len() + generated.len() - 1;
            self.model.forward_token(
                token,
                pos,
                &mut self.kv_cache,
                &mut self.buffers,
                LogitsMode::Compute,
            )?;
        }
        let decode = decode_started.elapsed();
        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode error: {e}"))?;
        Ok(GenerationOutput {
            text,
            metrics: InferenceMetrics {
                prompt_tokens: prompt_tokens.len(),
                generated_tokens: generated.len(),
                prefill,
                decode,
            },
            token_ids: generated,
        })
    }

    /// Low-level benchmark API. It avoids tokenization and sampling overhead.
    pub fn benchmark_tokens(
        &mut self,
        prompt_tokens: &[u32],
        decode_tokens: usize,
    ) -> Result<InferenceMetrics> {
        ensure!(!prompt_tokens.is_empty(), "benchmark prompt is empty");
        ensure!(
            prompt_tokens.len() + decode_tokens <= self.kv_cache.max_seq_len,
            "benchmark token counts exceed context"
        );
        self.reset();
        let started = Instant::now();
        self.model
            .prefill(prompt_tokens, 0, &mut self.kv_cache, &mut self.buffers)?;
        let prefill = started.elapsed();

        let started = Instant::now();
        for step in 0..decode_tokens {
            let token = self.argmax_current_logits();
            self.model.forward_token(
                token,
                prompt_tokens.len() + step,
                &mut self.kv_cache,
                &mut self.buffers,
                LogitsMode::Compute,
            )?;
        }
        let decode = started.elapsed();
        Ok(InferenceMetrics {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: decode_tokens,
            prefill,
            decode,
        })
    }

    pub fn backend_summary(&self) -> String {
        format!(
            "{} / {} threads",
            self.model.backend().name(),
            self.model.backend().threads()
        )
    }

    fn argmax_current_logits(&self) -> u32 {
        self.buffers
            .logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index as u32)
            .unwrap_or(0)
    }

    fn remember(&mut self, token: u32) {
        if self.max_repeat_window == 0 {
            return;
        }
        self.recent_tokens.push(token);
        if self.recent_tokens.len() > self.max_repeat_window {
            self.recent_tokens.remove(0);
        }
    }

    fn sample_current_logits(&mut self) -> u32 {
        for &token in &self.recent_tokens {
            if let Some(value) = self.seen_tokens.get_mut(token as usize) {
                *value = true;
            }
        }

        self.candidates.clear();
        let temperature = self.temperature;
        for (token, &raw) in self.buffers.logits.iter().enumerate() {
            let mut value = raw;
            if self.repetition_penalty != 1.0 && self.seen_tokens[token] {
                value = if value > 0.0 {
                    value / self.repetition_penalty
                } else {
                    value * self.repetition_penalty
                };
            }
            if temperature > 0.0 {
                value /= temperature;
            }
            if !value.is_finite() {
                value = f32::NEG_INFINITY;
            }
            self.candidates.push((token, value));
        }
        for &token in &self.recent_tokens {
            if let Some(value) = self.seen_tokens.get_mut(token as usize) {
                *value = false;
            }
        }

        self.candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        if temperature <= 0.0 {
            return self.candidates[0].0 as u32;
        }
        if self.top_k > 0 && self.top_k < self.candidates.len() {
            self.candidates.truncate(self.top_k);
        }

        let max = self.candidates[0].1;
        let mut sum = 0.0f32;
        for (_, value) in &mut self.candidates {
            *value = (*value - max).exp();
            sum += *value;
        }
        if !sum.is_finite() || sum <= 0.0 {
            return self.candidates[0].0 as u32;
        }
        for (_, probability) in &mut self.candidates {
            *probability /= sum;
        }

        let mut cumulative = 0.0f32;
        let mut cutoff = self.candidates.len();
        for (index, &(_, probability)) in self.candidates.iter().enumerate() {
            cumulative += probability;
            if cumulative >= self.top_p {
                cutoff = index + 1;
                break;
            }
        }
        let kept = &self.candidates[..cutoff];
        let kept_sum: f32 = kept.iter().map(|(_, p)| p).sum();
        let mut random = rand::thread_rng().gen::<f32>() * kept_sum;
        for &(token, probability) in kept {
            random -= probability;
            if random <= 0.0 {
                return token as u32;
            }
        }
        kept.last().unwrap().0 as u32
    }
}

fn validate_sampling(options: &EngineOptions) -> Result<()> {
    if !options.temperature.is_finite() || options.temperature < 0.0 {
        bail!("temperature must be finite and non-negative");
    }
    if !options.top_p.is_finite() || !(0.0 < options.top_p && options.top_p <= 1.0) {
        bail!("top_p must be in (0, 1]");
    }
    if !options.repetition_penalty.is_finite() || options.repetition_penalty <= 0.0 {
        bail!("repetition_penalty must be finite and positive");
    }
    Ok(())
}

fn model_dir(path: &Path) -> &Path {
    if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    }
}

fn tokenizer_path(path: &Path) -> PathBuf {
    model_dir(path).join("tokenizer.json")
}

fn find_eos_tokens(path: &Path, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
    let mut result = Vec::new();
    let generation_config = model_dir(path).join("generation_config.json");
    if generation_config.exists() {
        let value: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&generation_config)
                .with_context(|| format!("failed to read {}", generation_config.display()))?,
        )?;
        if let Some(eos) = value.get("eos_token_id") {
            match eos {
                serde_json::Value::Number(n) => {
                    if let Some(id) = n.as_u64().and_then(|v| u32::try_from(v).ok()) {
                        result.push(id);
                    }
                }
                serde_json::Value::Array(values) => {
                    for value in values {
                        if let Some(id) = value.as_u64().and_then(|v| u32::try_from(v).ok()) {
                            result.push(id);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    for token in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(token) {
            result.push(id);
        }
    }
    result.sort_unstable();
    result.dedup();
    Ok(result)
}
