//! Performance control chart: run the fixed benchmark, record it, show the trend.
//!
//! Usage:
//!   llama-track -m model/qwen-0.5b [-q none|q4k|q4km] [-p 128] [-n 32]
//!               [-r 5] [-t N] [--backend auto|scalar|avx2|neon]
//!               [--note "step 1"] [--history bench/history.csv]

use anyhow::{bail, Context, Result};
use llama_rs::backend::{BackendConfig, KernelBackend, KernelPreference};
use llama_rs::track::{append_record, format_trend, read_decode_history, TrackRecord};
use llama_rs::{CausalModel, EngineOptions, InferenceEngine, LoadOptions, LoadQuantization};
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    model: PathBuf,
    quantization: LoadQuantization,
    quant_label: String,
    context: usize,
    prompt_tokens: usize,
    decode_tokens: usize,
    rounds: usize,
    threads: usize,
    kernel: KernelPreference,
    note: String,
    history: PathBuf,
}

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1);
    let mut result = Args {
        model: PathBuf::from("model/qwen-0.5b"),
        quantization: LoadQuantization::None,
        quant_label: "none".to_string(),
        context: 512,
        prompt_tokens: 128,
        decode_tokens: 32,
        rounds: 5,
        threads: 0,
        kernel: KernelPreference::Auto,
        note: String::new(),
        history: PathBuf::from("bench/history.csv"),
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" | "-m" => result.model = args.next().context("--model requires PATH")?.into(),
            "--quantize" | "-q" => {
                let value = args.next().context("--quantize requires MODE")?;
                result.quantization = match value.as_str() {
                    "none" => LoadQuantization::None,
                    "q4k" | "q4_k" => LoadQuantization::Q4K,
                    "q4km" | "q4_k_m" => LoadQuantization::Q4KM,
                    other => bail!("unknown quantization {other:?}; use none, q4k, or q4km"),
                };
                result.quant_label = value;
            }
            "--context" | "-c" => result.context = args.next().context("missing context")?.parse()?,
            "--prompt-tokens" | "-p" => {
                result.prompt_tokens = args.next().context("missing prompt token count")?.parse()?
            }
            "--decode-tokens" | "-n" => {
                result.decode_tokens = args.next().context("missing decode token count")?.parse()?
            }
            "--rounds" | "-r" => result.rounds = args.next().context("missing rounds")?.parse()?,
            "--threads" | "-t" => result.threads = args.next().context("missing threads")?.parse()?,
            "--backend" => {
                result.kernel = KernelPreference::parse(&args.next().context("missing backend")?)?
            }
            "--note" => result.note = args.next().context("--note requires TEXT")?,
            "--history" => result.history = args.next().context("--history requires PATH")?.into(),
            "--help" | "-h" => {
                println!(
                    "llama-track -m PATH [-q none|q4k|q4km] [-c 512] [-p 128] [-n 32] [-r 5] [-t N]\n            [--backend auto|scalar|avx2|neon] [--note TEXT] [--history PATH]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}"),
        }
    }
    if result.rounds == 0 || result.prompt_tokens == 0 {
        bail!("rounds and prompt-tokens must be positive");
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
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

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

    // Warm caches outside the measured rounds.
    let warm = &prompt[..prompt.len().min(8)];
    let _ = engine.benchmark_tokens(warm, usize::from(args.decode_tokens > 0))?;

    let mut results = Vec::with_capacity(args.rounds);
    for _ in 0..args.rounds {
        results.push(engine.benchmark_tokens(&prompt, args.decode_tokens)?);
    }
    // Median resists a single scheduling hiccup on a busy laptop.
    results.sort_by(|a, b| (a.prefill + a.decode).cmp(&(b.prefill + b.decode)));
    let median = &results[results.len() / 2];

    let mut record = TrackRecord::from_metrics(median, load_ms);
    record.note = args.note;
    record.backend = engine.model.backend().name().to_string();
    record.threads = engine.model.backend().threads();
    record.quant = args.quant_label;

    append_record(&args.history, &record)?;
    let history = read_decode_history(&args.history);

    println!("llama.rs control chart");
    println!("  model:    {}", args.model.display());
    println!("  backend:  {} x{}", record.backend, record.threads);
    println!("  quant:    {}", record.quant);
    println!(
        "  prefill:  {:>6} tokens, {:>8.2} tok/s",
        record.prompt_tokens, record.prefill_tps
    );
    print!("{}", format_trend(&history, &record, &args.history));
    Ok(())
}
