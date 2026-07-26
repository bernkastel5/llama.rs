#[derive(Debug, Clone)]
pub struct KVCache {
    pub k_cache: Vec<Vec<f32>>, // [layer][seq_len * head_dim]
    pub v_cache: Vec<Vec<f32>>,
    pub max_seq_len: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl KVCache {
    pub fn new(num_layers: usize, max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let size = max_seq_len * num_kv_heads * head_dim;
        let k_cache = vec![vec![0.0f32; size]; num_layers];
        let v_cache = vec![vec![0.0f32; size]; num_layers];
        Self { k_cache, v_cache, max_seq_len, num_layers, num_kv_heads, head_dim }
    }

    pub fn update(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let offset = pos * self.num_kv_heads * self.head_dim;
        let len = k.len();
        self.k_cache[layer][offset..offset + len].copy_from_slice(k);
        self.v_cache[layer][offset..offset + len].copy_from_slice(v);
    }
}

#[derive(Debug, Clone)]
pub struct InferenceBuffers {
    pub residual: Vec<f32>,
    pub hidden_states: Vec<f32>,
    pub logits: Vec<f32>,
    pub scratch: Vec<f32>,
}

impl InferenceBuffers {
    pub fn new(hidden_size: usize, vocab_size: usize) -> Self {
        Self {
            residual: vec![0.0; hidden_size],
            hidden_states: vec![0.0; hidden_size],
            logits: vec![0.0; vocab_size],
            scratch: vec![0.0; hidden_size * 4], // generous scratch
        }
    }
}