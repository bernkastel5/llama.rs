use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config, Model as Qwen2Model};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    println!("=== llama.rs (Candle + Qwen2) ===");

    let model_dir = "model/qwen-0.5b";
    let device = Device::Cpu;
    let max_new_tokens = 40;

    let config: Config = serde_json::from_str(
        &std::fs::read_to_string(format!("{}/config.json", model_dir))?
    )?;

    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_dir))
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

    let safetensors_path = format!("{}/model.safetensors", model_dir);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[safetensors_path.clone()], DType::F32, &device)?
    };

    let mut model = Qwen2Model::new(&config, vb)?;

    // Загружаем веса эмбеддингов с явной формой
    let vb_embed = unsafe {
        VarBuilder::from_mmaped_safetensors(&[safetensors_path], DType::F32, &device)?
    };

    // Явно указываем форму [vocab_size, hidden_size]
    let embed_tokens = vb_embed.get((151936, 896), "model.embed_tokens.weight")?;

    println!("Model loaded successfully!");

    let prompt = "Hello, world!";
    println!("Prompt: {}", prompt);

    let encoding = tokenizer.encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Encode error: {}", e))?;
    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

    println!("Generating...");

    for step in 0..max_new_tokens {
        let last_token = *tokens.last().unwrap();
        let input = Tensor::new(&[last_token], &device)?.unsqueeze(0)?;
        let position_ids = Tensor::new(&[tokens.len() as u32 - 1], &device)?.unsqueeze(0)?;

        let hidden_states = model.forward(&input, step, Some(&position_ids))?;

        let last_hidden = hidden_states.squeeze(0)?;
        let last_token_hidden = last_hidden.get(0)?;

        let logits = last_token_hidden
            .unsqueeze(0)?
            .matmul(&embed_tokens.t()?)?;
        let logits = logits.squeeze(0)?;

        let next_token = logits.argmax(0)?.to_scalar::<u32>()?;

        tokens.push(next_token);

        if next_token == 151645 {
            break;
        }
    }

    let generated_text = tokenizer.decode(&tokens, true)
        .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;

    println!("Generated: {}", generated_text);
    Ok(())
}
