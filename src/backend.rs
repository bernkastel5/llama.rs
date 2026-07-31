use crate::quant::{dot_row_for_kernel, QuantTensor};
use anyhow::{bail, ensure, Result};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::fmt;
use std::sync::Arc;

/// Runtime-selectable CPU kernel family. `Auto` never enables instructions that
/// were not detected on the current CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KernelPreference {
    #[default]
    Auto,
    Scalar,
    Avx2,
    Neon,
}

impl KernelPreference {
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "scalar" => Ok(Self::Scalar),
            "avx2" => Ok(Self::Avx2),
            "neon" => Ok(Self::Neon),
            other => bail!("unknown kernel backend {other:?}; use auto, scalar, avx2, or neon"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    Scalar,
    Avx2,
    Neon,
}

impl KernelKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2+fma",
            Self::Neon => "neon",
        }
    }
}

#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Zero means all logical CPUs visible to the process.
    pub threads: usize,
    pub kernel: KernelPreference,
    /// Matrices with fewer multiply-adds run on the caller thread.
    pub parallel_threshold: usize,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            threads: 0,
            kernel: KernelPreference::Auto,
            parallel_threshold: 128 * 1024,
        }
    }
}

/// Stable boundary between model architectures and numerical kernels.
/// Architecture code does not know whether a matrix uses scalar, AVX2, or NEON code.
pub trait KernelBackend: Send + Sync + fmt::Debug {
    fn name(&self) -> &'static str;
    fn threads(&self) -> usize;
    fn matvec(&self, weight: &QuantTensor, input: &[f32], output: &mut [f32]) -> Result<()>;
}

#[derive(Clone)]
pub struct CpuBackend {
    inner: Arc<CpuBackendInner>,
}

struct CpuBackendInner {
    kind: KernelKind,
    threads: usize,
    parallel_threshold: usize,
    pool: ThreadPool,
}

impl fmt::Debug for CpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuBackend")
            .field("kernel", &self.inner.kind.name())
            .field("threads", &self.inner.threads)
            .field("parallel_threshold", &self.inner.parallel_threshold)
            .finish()
    }
}

impl CpuBackend {
    pub fn new(config: &BackendConfig) -> Result<Self> {
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let threads = if config.threads == 0 {
            available
        } else {
            config.threads
        };
        ensure!(threads > 0, "backend thread count must be positive");
        let kind = select_kernel(config.kernel)?;
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("llama-rs-{index}"))
            .build()?;
        Ok(Self {
            inner: Arc::new(CpuBackendInner {
                kind,
                threads,
                parallel_threshold: config.parallel_threshold,
                pool,
            }),
        })
    }

    pub fn kernel_kind(&self) -> KernelKind {
        self.inner.kind
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new(&BackendConfig::default()).expect("default CPU backend must be constructible")
    }
}

impl KernelBackend for CpuBackend {
    fn name(&self) -> &'static str {
        self.inner.kind.name()
    }

    fn threads(&self) -> usize {
        self.inner.threads
    }

    fn matvec(&self, weight: &QuantTensor, input: &[f32], output: &mut [f32]) -> Result<()> {
        weight.validate_matvec(input, output)?;
        let work = weight.rows().saturating_mul(weight.cols());
        let kind = self.inner.kind;
        if self.inner.threads > 1
            && weight.rows() >= self.inner.threads * 2
            && work >= self.inner.parallel_threshold
        {
            self.inner.pool.install(|| {
                output.par_iter_mut().enumerate().for_each(|(row, out)| {
                    *out = dot_row_for_kernel(kind, weight.row_data(row), weight.quant_type, input)
                });
            });
        } else {
            for (row, out) in output.iter_mut().enumerate() {
                *out = dot_row_for_kernel(kind, weight.row_data(row), weight.quant_type, input);
            }
        }
        Ok(())
    }
}

fn select_kernel(preference: KernelPreference) -> Result<KernelKind> {
    let avx2 = avx2_available();
    let neon = neon_available();
    match preference {
        KernelPreference::Auto => Ok(if avx2 {
            KernelKind::Avx2
        } else if neon {
            KernelKind::Neon
        } else {
            KernelKind::Scalar
        }),
        KernelPreference::Scalar => Ok(KernelKind::Scalar),
        KernelPreference::Avx2 if avx2 => Ok(KernelKind::Avx2),
        KernelPreference::Avx2 => bail!("AVX2+FMA was requested but is unavailable on this CPU"),
        KernelPreference::Neon if neon => Ok(KernelKind::Neon),
        KernelPreference::Neon => bail!("NEON was requested but is unavailable on this CPU"),
    }
}

fn avx2_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

fn neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}
