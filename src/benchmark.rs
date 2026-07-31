use std::time::Duration;

#[derive(Debug, Clone, Default)]
pub struct InferenceMetrics {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill: Duration,
    pub decode: Duration,
}

impl InferenceMetrics {
    pub fn prefill_tokens_per_second(&self) -> f64 {
        rate(self.prompt_tokens, self.prefill)
    }

    pub fn decode_tokens_per_second(&self) -> f64 {
        rate(self.generated_tokens, self.decode)
    }
}

#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub metrics: InferenceMetrics,
}

fn rate(tokens: usize, duration: Duration) -> f64 {
    if duration.is_zero() {
        0.0
    } else {
        tokens as f64 / duration.as_secs_f64()
    }
}
