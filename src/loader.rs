use crate::config::LlamaConfig;
use crate::gguf::GgufFile;
use crate::model::{LinearLayer, LlamaModel, RMSNorm, TransformerBlock};
use crate::quant::{QuantTensor, QuantType};
use anyhow::{bail, ensure, Context, Result};
use memmap2::Mmap;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoadQuantization {
    #[default]
    None,
    /// Uniform load-time GGML Q4_K, with Q5_0 fallback for incompatible rows.
    Q4K,
    /// A quality-biased Q4_K_M-style load-time recipe: Q8_0 for embeddings/value
    /// projections, Q6_K for sensitive down projections, and Q5_0 fallback elsewhere.
    Q4KM,
}

#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    pub quantization: LoadQuantization,
}

pub struct ModelLoader;

impl ModelLoader {
    /// `path` can be a Hugging Face model directory, a `.safetensors` file,
    /// or a single `.gguf` file.
    pub fn load(path: impl AsRef<Path>, options: &LoadOptions) -> Result<LlamaModel> {
        let path = path.as_ref();
        if path.is_file() && path.extension().and_then(|v| v.to_str()) == Some("gguf") {
            return GgufLoader::load(path, options);
        }
        if path.is_file() && path.extension().and_then(|v| v.to_str()) == Some("safetensors") {
            let dir = path.parent().unwrap_or_else(|| Path::new("."));
            return SafetensorsLoader::load(dir, options);
        }
        if path.is_dir() {
            let mut ggufs = std::fs::read_dir(path)?
                .filter_map(|entry| entry.ok().map(|e| e.path()))
                .filter(|p| p.extension().and_then(|v| v.to_str()) == Some("gguf"))
                .collect::<Vec<_>>();
            ggufs.sort();
            let has_safetensors = path.join("model.safetensors").exists()
                || path.join("model.safetensors.index.json").exists();
            if has_safetensors {
                return SafetensorsLoader::load(path, options);
            }
            if ggufs.len() == 1 {
                return GgufLoader::load(&ggufs[0], options);
            }
            if ggufs.len() > 1 {
                bail!(
                    "{} contains multiple GGUF files; pass the desired file explicitly",
                    path.display()
                );
            }
        }
        bail!("cannot detect model format at {}", path.display())
    }
}

pub struct SafetensorsLoader;

impl SafetensorsLoader {
    pub fn load(model_dir: impl AsRef<Path>, options: &LoadOptions) -> Result<LlamaModel> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let config_json = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let config = LlamaConfig::from_json(&config_json)?;
        let store = SafetensorStore::open(model_dir)?;

        let embed_tokens = load_st_matrix(
            &store,
            "model.embed_tokens.weight",
            config.vocab_size,
            config.hidden_size,
            options.quantization,
        )?;
        let norm = RMSNorm::new(
            load_st_vector(&store, "model.norm.weight", config.hidden_size)?,
            config.rms_norm_eps,
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let kv_width = config.kv_width();
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let input_layernorm = RMSNorm::new(
                load_st_vector(
                    &store,
                    &format!("{prefix}.input_layernorm.weight"),
                    config.hidden_size,
                )?,
                config.rms_norm_eps,
            )?;
            let post_attention_layernorm = RMSNorm::new(
                load_st_vector(
                    &store,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    config.hidden_size,
                )?,
                config.rms_norm_eps,
            )?;

            let self_attn_q = load_st_linear(
                &store,
                &format!("{prefix}.self_attn.q_proj.weight"),
                Some(&format!("{prefix}.self_attn.q_proj.bias")),
                config.hidden_size,
                config.hidden_size,
                options.quantization,
            )?;
            let self_attn_k = load_st_linear(
                &store,
                &format!("{prefix}.self_attn.k_proj.weight"),
                Some(&format!("{prefix}.self_attn.k_proj.bias")),
                kv_width,
                config.hidden_size,
                options.quantization,
            )?;
            let self_attn_v = load_st_linear(
                &store,
                &format!("{prefix}.self_attn.v_proj.weight"),
                Some(&format!("{prefix}.self_attn.v_proj.bias")),
                kv_width,
                config.hidden_size,
                options.quantization,
            )?;
            let self_attn_o = load_st_linear(
                &store,
                &format!("{prefix}.self_attn.o_proj.weight"),
                None,
                config.hidden_size,
                config.hidden_size,
                options.quantization,
            )?;
            let mlp_gate = load_st_linear(
                &store,
                &format!("{prefix}.mlp.gate_proj.weight"),
                None,
                config.intermediate_size,
                config.hidden_size,
                options.quantization,
            )?;
            let mlp_up = load_st_linear(
                &store,
                &format!("{prefix}.mlp.up_proj.weight"),
                None,
                config.intermediate_size,
                config.hidden_size,
                options.quantization,
            )?;
            let mlp_down = load_st_linear(
                &store,
                &format!("{prefix}.mlp.down_proj.weight"),
                None,
                config.hidden_size,
                config.intermediate_size,
                options.quantization,
            )?;

            layers.push(TransformerBlock {
                input_layernorm,
                self_attn_q,
                self_attn_k,
                self_attn_v,
                self_attn_o,
                post_attention_layernorm,
                mlp_gate,
                mlp_up,
                mlp_down,
            });
        }

        let lm_weight = if store.contains("lm_head.weight") {
            load_st_matrix(
                &store,
                "lm_head.weight",
                config.vocab_size,
                config.hidden_size,
                options.quantization,
            )?
        } else {
            // Qwen2 commonly ties the language-model head to token embeddings.
            embed_tokens.clone()
        };
        let lm_head = LinearLayer::new(lm_weight, None)?;

        Ok(LlamaModel {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
        })
    }
}

pub struct GgufLoader;

impl GgufLoader {
    pub fn load(path: impl AsRef<Path>, options: &LoadOptions) -> Result<LlamaModel> {
        let path = path.as_ref();
        let gguf = GgufFile::open(path)?;
        let architecture = gguf.metadata_str("general.architecture")?;
        ensure!(
            architecture == "qwen2",
            "GGUF architecture {architecture:?} is unsupported; this MVP supports qwen2/Qwen2.5"
        );

        let token_info = gguf.tensor_info("token_embd.weight")?;
        ensure!(
            token_info.dimensions.len() == 2,
            "token_embd.weight must be 2D"
        );
        let config = LlamaConfig {
            hidden_size: as_usize(
                gguf.metadata_u64("qwen2.embedding_length")?,
                "embedding_length",
            )?,
            intermediate_size: as_usize(
                gguf.metadata_u64("qwen2.feed_forward_length")?,
                "feed_forward_length",
            )?,
            num_attention_heads: as_usize(
                gguf.metadata_u64("qwen2.attention.head_count")?,
                "head_count",
            )?,
            num_key_value_heads: as_usize(
                gguf.metadata_u64("qwen2.attention.head_count_kv")?,
                "head_count_kv",
            )?,
            num_hidden_layers: as_usize(gguf.metadata_u64("qwen2.block_count")?, "block_count")?,
            vocab_size: token_info.dimensions[1],
            rms_norm_eps: gguf.metadata_f32("qwen2.attention.layer_norm_rms_epsilon")?,
            rope_theta: gguf
                .metadata
                .get("qwen2.rope.freq_base")
                .and_then(|v| v.as_f32())
                .unwrap_or(10_000.0),
            max_position_embeddings: as_usize(
                gguf.metadata_u64("qwen2.context_length")?,
                "context_length",
            )?,
            head_dim: None,
            tie_word_embeddings: !gguf.tensors.contains_key("output.weight"),
        };
        config.validate()?;
        ensure!(
            token_info.dimensions[0] == config.hidden_size,
            "GGUF embedding width/config mismatch"
        );

        let embed_tokens = load_gguf_matrix(
            &gguf,
            "token_embd.weight",
            config.vocab_size,
            config.hidden_size,
            options.quantization,
        )?;
        let norm = RMSNorm::new(
            load_gguf_vector(&gguf, "output_norm.weight", config.hidden_size)?,
            config.rms_norm_eps,
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let kv_width = config.kv_width();
        for i in 0..config.num_hidden_layers {
            let prefix = format!("blk.{i}");
            layers.push(TransformerBlock {
                input_layernorm: RMSNorm::new(
                    load_gguf_vector(
                        &gguf,
                        &format!("{prefix}.attn_norm.weight"),
                        config.hidden_size,
                    )?,
                    config.rms_norm_eps,
                )?,
                self_attn_q: load_gguf_linear(
                    &gguf,
                    &format!("{prefix}.attn_q.weight"),
                    Some(&format!("{prefix}.attn_q.bias")),
                    config.hidden_size,
                    config.hidden_size,
                    options.quantization,
                )?,
                self_attn_k: load_gguf_linear(
                    &gguf,
                    &format!("{prefix}.attn_k.weight"),
                    Some(&format!("{prefix}.attn_k.bias")),
                    kv_width,
                    config.hidden_size,
                    options.quantization,
                )?,
                self_attn_v: load_gguf_linear(
                    &gguf,
                    &format!("{prefix}.attn_v.weight"),
                    Some(&format!("{prefix}.attn_v.bias")),
                    kv_width,
                    config.hidden_size,
                    options.quantization,
                )?,
                self_attn_o: load_gguf_linear(
                    &gguf,
                    &format!("{prefix}.attn_output.weight"),
                    None,
                    config.hidden_size,
                    config.hidden_size,
                    options.quantization,
                )?,
                post_attention_layernorm: RMSNorm::new(
                    load_gguf_vector(
                        &gguf,
                        &format!("{prefix}.ffn_norm.weight"),
                        config.hidden_size,
                    )?,
                    config.rms_norm_eps,
                )?,
                mlp_gate: load_gguf_linear(
                    &gguf,
                    &format!("{prefix}.ffn_gate.weight"),
                    None,
                    config.intermediate_size,
                    config.hidden_size,
                    options.quantization,
                )?,
                mlp_up: load_gguf_linear(
                    &gguf,
                    &format!("{prefix}.ffn_up.weight"),
                    None,
                    config.intermediate_size,
                    config.hidden_size,
                    options.quantization,
                )?,
                mlp_down: load_gguf_linear(
                    &gguf,
                    &format!("{prefix}.ffn_down.weight"),
                    None,
                    config.hidden_size,
                    config.intermediate_size,
                    options.quantization,
                )?,
            });
        }

        let lm_weight = if gguf.tensors.contains_key("output.weight") {
            load_gguf_matrix(
                &gguf,
                "output.weight",
                config.vocab_size,
                config.hidden_size,
                options.quantization,
            )?
        } else {
            embed_tokens.clone()
        };
        let lm_head = LinearLayer::new(lm_weight, None)?;
        Ok(LlamaModel {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
        })
    }
}

fn as_usize(value: u64, name: &str) -> Result<usize> {
    usize::try_from(value).with_context(|| format!("{name} does not fit usize"))
}

fn maybe_quantize(
    tensor: QuantTensor,
    quantization: LoadQuantization,
    name: &str,
) -> Result<QuantTensor> {
    if quantization == LoadQuantization::None
        || tensor.shape.len() != 2
        || !matches!(
            tensor.quant_type,
            QuantType::None | QuantType::F16 | QuantType::BF16
        )
    {
        return Ok(tensor);
    }

    let rows = tensor.rows();
    let cols = tensor.cols();
    let values = tensor.to_vec_f32();
    match quantization {
        LoadQuantization::None => unreachable!(),
        LoadQuantization::Q4K => QuantTensor::quantize_q4k(&values, rows, cols),
        LoadQuantization::Q4KM => {
            if name.contains("embed_tokens")
                || name == "token_embd.weight"
                || name == "lm_head.weight"
                || name == "output.weight"
                || name.contains("attn.v_proj.weight")
                || name.contains("attn_v.weight")
            {
                QuantTensor::quantize_q8_0(&values, rows, cols)
            } else if name.contains("mlp.down_proj.weight") || name.contains("ffn_down.weight") {
                // Down projections are exceptionally sensitive on small Qwen2 models.
                // Without an importance matrix, keep them at Q6_K. Existing GGUF
                // Q4_K_M tensors are preserved verbatim and may use an imatrix-backed
                // Q4_K/Q6_K mixture instead.
                QuantTensor::quantize_q6k(&values, rows, cols)
            } else {
                QuantTensor::quantize_q4k(&values, rows, cols)
            }
        }
    }
}

fn check_matrix(tensor: &QuantTensor, name: &str, rows: usize, cols: usize) -> Result<()> {
    ensure!(
        tensor.shape == [rows, cols],
        "tensor {name} has shape {:?}, expected [{rows}, {cols}]",
        tensor.shape
    );
    Ok(())
}

fn load_st_matrix(
    store: &SafetensorStore,
    name: &str,
    rows: usize,
    cols: usize,
    quantization: LoadQuantization,
) -> Result<QuantTensor> {
    let tensor = store.tensor(name)?;
    check_matrix(&tensor, name, rows, cols)?;
    maybe_quantize(tensor, quantization, name)
}

fn load_st_vector(store: &SafetensorStore, name: &str, len: usize) -> Result<Vec<f32>> {
    let tensor = store.tensor(name)?;
    ensure!(
        tensor.shape == [len],
        "tensor {name} has shape {:?}, expected [{len}]",
        tensor.shape
    );
    Ok(tensor.to_vec_f32())
}

fn load_st_linear(
    store: &SafetensorStore,
    weight_name: &str,
    bias_name: Option<&str>,
    rows: usize,
    cols: usize,
    quantization: LoadQuantization,
) -> Result<LinearLayer> {
    let weight = load_st_matrix(store, weight_name, rows, cols, quantization)?;
    let bias = match bias_name {
        Some(name) if store.contains(name) => Some(load_st_vector(store, name, rows)?),
        Some(_) => Some(vec![0.0; rows]),
        None => None,
    };
    LinearLayer::new(weight, bias)
}

fn load_gguf_matrix(
    gguf: &GgufFile,
    name: &str,
    rows: usize,
    cols: usize,
    quantization: LoadQuantization,
) -> Result<QuantTensor> {
    let tensor = gguf.tensor(name)?;
    check_matrix(&tensor, name, rows, cols)?;
    maybe_quantize(tensor, quantization, name)
}

fn load_gguf_vector(gguf: &GgufFile, name: &str, len: usize) -> Result<Vec<f32>> {
    let tensor = gguf.tensor(name)?;
    ensure!(
        tensor.shape == [len],
        "tensor {name} has shape {:?}, expected [{len}]",
        tensor.shape
    );
    Ok(tensor.to_vec_f32())
}

fn load_gguf_linear(
    gguf: &GgufFile,
    weight_name: &str,
    bias_name: Option<&str>,
    rows: usize,
    cols: usize,
    quantization: LoadQuantization,
) -> Result<LinearLayer> {
    let weight = load_gguf_matrix(gguf, weight_name, rows, cols, quantization)?;
    let bias = match bias_name {
        Some(name) if gguf.tensors.contains_key(name) => Some(load_gguf_vector(gguf, name, rows)?),
        Some(_) => Some(vec![0.0; rows]),
        None => None,
    };
    LinearLayer::new(weight, bias)
}

#[derive(Deserialize)]
struct SafetensorIndex {
    weight_map: HashMap<String, String>,
}

struct SafetensorStore {
    files: HashMap<PathBuf, Arc<Mmap>>,
    tensor_files: HashMap<String, PathBuf>,
}

impl SafetensorStore {
    fn open(model_dir: &Path) -> Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let mut tensor_files = HashMap::new();
        let mut paths = Vec::new();

        if index_path.exists() {
            let index: SafetensorIndex = serde_json::from_str(
                &std::fs::read_to_string(&index_path)
                    .with_context(|| format!("failed to read {}", index_path.display()))?,
            )?;
            for (name, file) in index.weight_map {
                let path = model_dir.join(file);
                tensor_files.insert(name, path.clone());
                paths.push(path);
            }
            paths.sort();
            paths.dedup();
        } else {
            paths = std::fs::read_dir(model_dir)?
                .filter_map(|entry| entry.ok().map(|e| e.path()))
                .filter(|p| p.extension().and_then(|v| v.to_str()) == Some("safetensors"))
                .collect();
            paths.sort();
            ensure!(
                !paths.is_empty(),
                "no .safetensors files in {}",
                model_dir.display()
            );
        }

        let mut files = HashMap::new();
        for path in paths {
            let file =
                File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
            let mmap = Arc::new(unsafe { Mmap::map(&file)? });
            if !index_path.exists() {
                let tensors = SafeTensors::deserialize(&mmap[..])
                    .with_context(|| format!("invalid safetensors file {}", path.display()))?;
                for (name, _) in tensors.iter() {
                    ensure!(
                        tensor_files.insert(name.to_owned(), path.clone()).is_none(),
                        "duplicate safetensors tensor {name}"
                    );
                }
            }
            files.insert(path, mmap);
        }

        Ok(Self {
            files,
            tensor_files,
        })
    }

    fn contains(&self, name: &str) -> bool {
        self.tensor_files.contains_key(name)
    }

    fn tensor(&self, name: &str) -> Result<QuantTensor> {
        let path = self
            .tensor_files
            .get(name)
            .with_context(|| format!("missing safetensors tensor {name}"))?;
        let mmap = self
            .files
            .get(path)
            .context("internal safetensors shard lookup error")?
            .clone();
        let (start, len, shape, ty) = {
            let tensors = SafeTensors::deserialize(&mmap[..])?;
            let view = tensors.tensor(name)?;
            tensor_view_metadata(&mmap, &view)?
        };
        QuantTensor::from_mmap(mmap, start, len, shape, ty)
            .with_context(|| format!("invalid safetensors tensor {name}"))
    }
}

fn tensor_view_metadata(
    mmap: &Mmap,
    view: &TensorView<'_>,
) -> Result<(usize, usize, Vec<usize>, QuantType)> {
    let ty = match view.dtype() {
        Dtype::F32 => QuantType::None,
        Dtype::F16 => QuantType::F16,
        Dtype::BF16 => QuantType::BF16,
        other => bail!("unsupported safetensors dtype {other:?}; use F32/F16/BF16"),
    };
    let base = mmap.as_ptr() as usize;
    let ptr = view.data().as_ptr() as usize;
    ensure!(ptr >= base, "safetensors data pointer precedes mmap");
    let start = ptr - base;
    let len = view.data().len();
    ensure!(
        start <= mmap.len() && len <= mmap.len() - start,
        "safetensors tensor range is outside mmap"
    );
    Ok((start, len, view.shape().to_vec(), ty))
}
