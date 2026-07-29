use anyhow::Result;
use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config, Model as Qwen2Model};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    println!("=== llama.rs (Candle + Qwen2) ===");

    let model_dir = "model/qwen-0.5b";
    let device = Device::Cpu;

    let config: Config = serde_json::from_str(
        &std::fs::read_to_string(format!("{}/config.json", model_dir))?
    )?;

    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_dir))
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

    let safetensors_path = format!("{}/model.safetensors", model_dir);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[safetensors_path], DType::F32, &device)?
    };

    let model = Qwen2Model::new(&config, vb)?;

    println!("Model loaded successfully!");

    let prompt = "Hello, world!";
    println!("Prompt: {}", prompt);

    // Простая генерация с помощью TextGeneration (если доступно)
    // или ручной цикл

    println!("Done.");
    Ok(())
}
