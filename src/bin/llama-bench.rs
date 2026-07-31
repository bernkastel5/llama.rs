use anyhow::{bail, Context, Result};
use llama_rs::backend::{BackendConfig, KernelPreference};
use llama_rs::{EngineOptions, InferenceEngine, LoadOptions, LoadQuantization};
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    model: PathBuf,
    quantization: LoadQuantization,
    context: usize,
    prompt_tokens: usize,
    decode_tokens: usize,
    rounds: usize,
    threads: usize,
    kernel: KernelPreference,
    json: bool,
}

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1);
    let mut result = Args {
        model: PathBuf::from("model/qwen-0.5b"),
        quantization: LoadQuantization::None,
        context: 512,
        prompt_tokens: 128,
        decode_tokens: 32,
        rounds: 3,
        threads: 0,
        kernel: KernelPreference::Auto,
        json: false,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" | "-m" => result.model = args.next().context("--model requires PATH")?.into(),
            "--quantize" | "-q" => {
                result.quantization =
                    match args.next().context("--quantize requires MODE")?.as_str() {
                        "none" => LoadQuantization::None,
                        "q4k" | "q4_k" => LoadQuantization::Q4K,
                        "q4km" | "q4_k_m" => LoadQuantization::Q4KM,
                        value => bail!("unknown quantization {value:?}"),
                    }
            }
            "--context" | "-c" => {
                result.context = args.next().context("missing context")?.parse()?
            }
            "--prompt-tokens" | "-p" => {
                result.prompt_tokens = args.next().context("missing prompt token count")?.parse()?
            }
            "--decode-tokens" | "-n" => {
                result.decode_tokens = args.next().context("missing decode token count")?.parse()?
            }
            "--rounds" | "-r" => {
                result.rounds = args.next().context("missing round count")?.parse()?
            }
            "--threads" | "-t" => {
                result.threads = args.next().context("missing thread count")?.parse()?
            }
            "--backend" => {
                result.kernel = KernelPreference::parse(&args.next().context("missing backend")?)?
            }
            "--json" => result.json = true,
            "--help" | "-h" => {
                println!("llama-bench -m PATH [-p 128] [-n 32] [-r 3] [-t N] [--backend auto|scalar|avx2|neon] [--json]");
                std::process::exit(0);
            }
            value => bail!("unknown argument {value:?}"),
        }
    }
    if result.rounds == 0 || result.prompt_tokens == 0 {
        bail!("rounds and prompt-tokens must be positive");
    }
    if result.prompt_tokens + result.decode_tokens > result.context {
        result.context = result.prompt_tokens + result.decode_tokens;
    }
    Ok(result)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let options = EngineOptions {
        load: LoadOptions {
            quantization: args.quantization,
            backend: BackendConfig {
                threads: args.threads,
                kernel: args.kernel,
                ..BackendConfig::default()
            },
        },
        context_length: args.context,
        ..EngineOptions::default()
    };

    let load_started = Instant::now();
    let mut engine = InferenceEngine::from_path(&args.model, options)?;
    let load = load_started.elapsed();

    let seed = engine
        .tokenizer
        .encode("The quick brown fox jumps over the lazy dog. ", false)
        .map_err(|error| anyhow::anyhow!("tokenizer error: {error}"))?;
    let seed = seed.get_ids();
    if seed.is_empty() {
        bail!("benchmark seed encoded to zero tokens");
    }
    let prompt: Vec<u32> = seed
        .iter()
        .copied()
        .cycle()
        .take(args.prompt_tokens)
        .collect();

    // Warm instruction/data caches without including it in reported rounds.
    let warm_prompt = &prompt[..prompt.len().min(8)];
    let _ = engine.benchmark_tokens(warm_prompt, usize::from(args.decode_tokens > 0))?;

    let mut results = Vec::with_capacity(args.rounds);
    for _ in 0..args.rounds {
        results.push(engine.benchmark_tokens(&prompt, args.decode_tokens)?);
    }
    results.sort_by(|a, b| {
        let left = a.prefill + a.decode;
        let right = b.prefill + b.decode;
        left.cmp(&right)
    });
    let result = &results[results.len() / 2];

    if args.json {
        println!(
            "{{\"architecture\":\"{}\",\"backend\":\"{}\",\"threads\":{},\"load_ms\":{:.3},\"prompt_tokens\":{},\"prefill_ms\":{:.3},\"prefill_tps\":{:.3},\"decode_tokens\":{},\"decode_ms\":{:.3},\"decode_tps\":{:.3}}}",
            engine.model.architecture().name(),
            engine.model.backend().name(),
            engine.model.backend().threads(),
            load.as_secs_f64() * 1000.0,
            result.prompt_tokens,
            result.prefill.as_secs_f64() * 1000.0,
            result.prefill_tokens_per_second(),
            result.generated_tokens,
            result.decode.as_secs_f64() * 1000.0,
            result.decode_tokens_per_second(),
        );
    } else {
        println!("llama.rs benchmark");
        println!("  model:        {}", args.model.display());
        println!("  architecture: {}", engine.model.architecture().name());
        println!("  backend:      {}", engine.backend_summary());
        println!("  load:         {:.3} s", load.as_secs_f64());
        println!(
            "  prefill:      {:>6} tokens, {:>8.2} tok/s, {:>8.3} s",
            result.prompt_tokens,
            result.prefill_tokens_per_second(),
            result.prefill.as_secs_f64(),
        );
        println!(
            "  decode:       {:>6} tokens, {:>8.2} tok/s, {:>8.3} s",
            result.generated_tokens,
            result.decode_tokens_per_second(),
            result.decode.as_secs_f64(),
        );
    }
    Ok(())
}
