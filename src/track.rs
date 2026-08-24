//! Performance control chart.
//!
//! `llama-bench` already measures one build correctly: fixed token counts,
//! greedy decoding, a warmup round and a median over rounds. What it cannot do
//! is answer "did this change help?", because it keeps no history. This module
//! appends every run to a CSV and reports the delta against the previous run
//! and against the baseline.
//!
//! Timing an interactive reply instead has three systematic faults: the reply
//! length varies with sampling, prefill and decode collapse into one number
//! even though they have different bottlenecks, and nothing is retained, so
//! small regressions accumulate unnoticed.

use crate::benchmark::InferenceMetrics;
use anyhow::{Context, Result};
use std::fmt::Write as _;
use std::fs::{create_dir_all, OpenOptions};
use std::io::Write as _;
use std::path::Path;

pub const HISTORY_HEADER: &str = "utc,commit,dirty,note,backend,threads,quant,prompt_tokens,prefill_tps,decode_tokens,decode_tps,decode_ms_per_token,load_ms";

/// One row of the control chart.
#[derive(Debug, Clone)]
pub struct TrackRecord {
    pub utc: String,
    pub commit: String,
    pub dirty: bool,
    pub note: String,
    pub backend: String,
    pub threads: usize,
    pub quant: String,
    pub prompt_tokens: usize,
    pub prefill_tps: f64,
    pub decode_tokens: usize,
    pub decode_tps: f64,
    pub decode_ms_per_token: f64,
    pub load_ms: f64,
}

impl TrackRecord {
    pub fn from_metrics(metrics: &InferenceMetrics, load_ms: f64) -> Self {
        Self {
            utc: utc_timestamp(),
            commit: git_commit(),
            dirty: git_dirty(),
            note: String::new(),
            backend: String::new(),
            threads: 0,
            quant: String::new(),
            prompt_tokens: metrics.prompt_tokens,
            prefill_tps: metrics.prefill_tokens_per_second(),
            decode_tokens: metrics.generated_tokens,
            decode_tps: metrics.decode_tokens_per_second(),
            decode_ms_per_token: ms_per_token(metrics),
            load_ms,
        }
    }

    pub fn to_csv_line(&self) -> String {
        let mut line = String::new();
        let _ = write!(
            line,
            "{},{},{},{},{},{},{},{},{:.3},{},{:.3},{:.3},{:.3}",
            self.utc,
            self.commit,
            if self.dirty { "dirty" } else { "clean" },
            escape(&self.note),
            escape(&self.backend),
            self.threads,
            escape(&self.quant),
            self.prompt_tokens,
            self.prefill_tps,
            self.decode_tokens,
            self.decode_tps,
            self.decode_ms_per_token,
            self.load_ms,
        );
        line
    }
}

/// Milliseconds per decoded token: the quantity a reader actually feels.
///
/// Tokens per second is nonlinear, so equal percentages hide unequal savings —
/// 10 to 11 tok/s frees 9 ms per token while 50 to 55 frees 1.8 ms. Milliseconds
/// add up linearly and compare directly against the per-step budgets in the plan.
pub fn ms_per_token(metrics: &InferenceMetrics) -> f64 {
    if metrics.generated_tokens == 0 {
        0.0
    } else {
        metrics.decode.as_secs_f64() * 1000.0 / metrics.generated_tokens as f64
    }
}

/// Decode throughputs already on record, oldest first.
pub fn read_decode_history(path: &Path) -> Vec<f64> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .skip(1)
        .filter_map(|line| line.split(',').nth(10))
        .filter_map(|field| field.trim().parse::<f64>().ok())
        .filter(|value| *value > 0.0)
        .collect()
}

pub fn append_record(path: &Path, record: &TrackRecord) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent)
                .with_context(|| format!("creating {} failed", parent.display()))?;
        }
    }
    let fresh = !path.exists();
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {} failed", path.display()))?;
    if fresh {
        writeln!(file, "{HISTORY_HEADER}")?;
    }
    writeln!(file, "{}", record.to_csv_line())?;
    Ok(())
}

/// Render the trend. `history` must already include the current run.
pub fn format_trend(history: &[f64], current: &TrackRecord, path: &Path) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        "\n  decode: {:.2} tok/s  ({:.3} мс/токен)\n",
        current.decode_tps, current.decode_ms_per_token
    );
    if history.len() >= 2 {
        let previous = history[history.len() - 2];
        if previous > 0.0 {
            let _ = write!(
                out,
                "  к прошлому запуску: {:+.1}%\n",
                (current.decode_tps / previous - 1.0) * 100.0
            );
        }
    }
    if let Some(&baseline) = history.first() {
        if baseline > 0.0 && history.len() >= 2 {
            let _ = write!(
                out,
                "  к baseline:         {:+.1}%  (было {:.2} tok/s)\n",
                (current.decode_tps / baseline - 1.0) * 100.0,
                baseline
            );
        }
    }
    let _ = write!(
        out,
        "  история: {} ({} записей)\n",
        path.display(),
        history.len()
    );
    out
}

fn escape(value: &str) -> String {
    value.replace([',', '\n', '\r'], ";")
}

fn utc_timestamp() -> String {
    // Civil-time conversion from the Unix epoch, so no date dependency is
    // pulled in for one formatted string.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        tod / 3600,
        (tod % 3600) / 60,
        tod % 60
    )
}

/// Howard Hinnant's `civil_from_days`, valid across the range we care about.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn git_output(args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn git_commit() -> String {
    git_output(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "none".to_string())
}

fn git_dirty() -> bool {
    git_output(&["status", "--porcelain"]).is_some_and(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn record(tps: f64) -> TrackRecord {
        TrackRecord {
            utc: "2026-08-24T10:00:00Z".into(),
            commit: "abc1234".into(),
            dirty: false,
            note: "n".into(),
            backend: "avx2+fma".into(),
            threads: 2,
            quant: "none".into(),
            prompt_tokens: 128,
            prefill_tps: 100.0,
            decode_tokens: 32,
            decode_tps: tps,
            decode_ms_per_token: 1000.0 / tps,
            load_ms: 10.0,
        }
    }

    #[test]
    fn ms_per_token_matches_decode_duration() {
        let metrics = InferenceMetrics {
            prompt_tokens: 128,
            generated_tokens: 32,
            prefill: Duration::from_millis(900),
            decode: Duration::from_millis(3200),
        };
        assert!((ms_per_token(&metrics) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn ms_per_token_handles_zero_tokens() {
        assert_eq!(ms_per_token(&InferenceMetrics::default()), 0.0);
    }

    #[test]
    fn csv_line_survives_commas_in_note() {
        let mut r = record(11.7);
        r.note = "step 1, activations".into();
        let line = r.to_csv_line();
        assert_eq!(line.split(',').count(), 13, "line={line}");
    }

    #[test]
    fn history_roundtrips_through_the_csv() {
        let dir = std::env::temp_dir().join(format!("llama-track-{}", std::process::id()));
        let path = dir.join("history.csv");
        let _ = std::fs::remove_file(&path);

        append_record(&path, &record(11.70)).unwrap();
        append_record(&path, &record(28.50)).unwrap();
        let history = read_decode_history(&path);
        assert_eq!(history.len(), 2);
        assert!((history[0] - 11.70).abs() < 1e-9);
        assert!((history[1] - 28.50).abs() < 1e-9);

        let trend = format_trend(&history, &record(28.50), &path);
        assert!(trend.contains("+143.6%"), "trend={trend}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn first_run_reports_no_delta() {
        let history = vec![11.7];
        let trend = format_trend(&history, &record(11.7), Path::new("h.csv"));
        assert!(!trend.contains("baseline"), "trend={trend}");
        assert!(!trend.contains("прошлому"), "trend={trend}");
    }

    #[test]
    fn timestamp_is_well_formed() {
        let stamp = utc_timestamp();
        assert_eq!(stamp.len(), 20, "stamp={stamp}");
        assert!(stamp.ends_with('Z'));
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_000), (2022, 1, 8));
    }
}
