use anyhow::{bail, Context, Result};
use llama_rs::{EngineOptions, InferenceEngine, LoadQuantization};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    model: PathBuf,
    quantization: LoadQuantization,
    context: usize,
    max_tokens: usize,
}

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1);
    let mut result = Args {
        model: PathBuf::from("model/qwen-0.5b"),
        quantization: LoadQuantization::None,
        context: 4096,
        max_tokens: 512,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" | "-m" => {
                result.model = PathBuf::from(args.next().context("--model requires a path")?)
            }
            "--quantize" | "-q" => {
                result.quantization = match args
                    .next()
                    .context("--quantize requires none, q4k, or q4km")?
                    .to_ascii_lowercase()
                    .as_str()
                {
                    "none" => LoadQuantization::None,
                    "q4k" | "q4_k" => LoadQuantization::Q4K,
                    "q4km" | "q4_k_m" => LoadQuantization::Q4KM,
                    value => bail!("unknown quantization {value:?}; use none, q4k, or q4km"),
                }
            }
            "--context" | "-c" => {
                result.context = args
                    .next()
                    .context("--context requires a number")?
                    .parse()?
            }
            "--max-tokens" | "-n" => {
                result.max_tokens = args
                    .next()
                    .context("--max-tokens requires a number")?
                    .parse()?
            }
            "--help" | "-h" => {
                println!(
                    "llama-rs --model PATH [--quantize none|q4k|q4km] [--context N] [--max-tokens N]"
                );
                println!(
                    "PATH is a Hugging Face model directory, .safetensors file, or .gguf file."
                );
                std::process::exit(0);
            }
            value => bail!("unknown argument {value:?}; try --help"),
        }
    }
    Ok(result)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let mut options = EngineOptions::default();
    options.load.quantization = args.quantization;
    options.context_length = args.context;

    println!("=== llama.rs — CPU-only Qwen2 ===");
    println!(
        "Loading {} ({:?})...",
        args.model.display(),
        args.quantization
    );
    let started = Instant::now();
    let mut engine = InferenceEngine::from_path(&args.model, options)?;
    println!(
        "Loaded {} layers, hidden {}, context {} in {:.2?}\n",
        engine.model.config.num_hidden_layers,
        engine.model.config.hidden_size,
        engine.kv_cache.max_seq_len,
        started.elapsed()
    );

    let system_prompt = "You are a helpful, concise, and direct AI assistant.";
    loop {
        print!("User: ");
        io::stdout().flush()?;
        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            break;
        }

        let prompt = format!(
            "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
        );
        let started = Instant::now();
        let response = engine.generate(&prompt, args.max_tokens)?;
        println!("\nAssistant: {}", response.trim());
        println!("[{:.2?}]\n", started.elapsed());
    }
    Ok(())
}
