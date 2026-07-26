use anyhow::Result;

fn main() -> Result<()> {
    println!("=== llama.rs Lightweight CPU Inference Engine ===");

    let model_dir = "./model"; // directory with config.json, model.safetensors, tokenizer.json

    // In practice ensure files exist. For skeleton:
    println!("Loading model from {}...", model_dir);

    let mut engine = llama_rs::engine::InferenceEngine::new(model_dir)?;

    println!("Model loaded. Running generation...");

    let output = engine.generate("Hello, world!", 20)?;
    println!("Generated: {}", output);

    println!("Inference complete.");
    Ok(())
}