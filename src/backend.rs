use crate::activation::{
    activation_layout, quantize_activation_q8_32, quantize_activation_q8k, ActivationLayout,
    BlockQ8K, BlockQ8_32, QK_K,
};
use crate::quant::{dot_row_for_kernel, QuantTensor};
use crate::simd::{dot_row_avx2_q8_32, dot_row_avx2_q8k};
use anyhow::{bail, ensure, Result};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::cell::RefCell;
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
    /// Quantize activations to 8 bits so matvec can use integer kernels.
    ///
    /// This trades accuracy for speed: the integer kernels reproduce their
    /// scalar reference to ~1e-7, but rounding the activation itself costs
    /// roughly 1e-3 relative error. That is the same trade llama.cpp makes and
    /// it is invisible in generated text, so inference enables it.
    ///
    /// Set to `false` for a reference-accuracy matvec (see
    /// [`BackendConfig::reference`]).
    pub integer_activations: bool,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            threads: 0,
            kernel: KernelPreference::Auto,
            parallel_threshold: 128 * 1024,
            integer_activations: true,
        }
    }
}

impl BackendConfig {
    /// Configuration that keeps every multiply in f32.
    ///
    /// Used where a matvec must match an exact dot product rather than merely
    /// be close to one: numerical tests, tooling and accuracy comparisons.
    pub fn reference() -> Self {
        Self {
            integer_activations: false,
            ..Self::default()
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
    integer_activations: bool,
    pool: ThreadPool,
}

impl fmt::Debug for CpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuBackend")
            .field("kernel", &self.inner.kind.name())
            .field("threads", &self.inner.threads)
            .field("parallel_threshold", &self.inner.parallel_threshold)
            .field("integer_activations", &self.inner.integer_activations)
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
                integer_activations: config.integer_activations,
                pool,
            }),
        })
    }

    pub fn kernel_kind(&self) -> KernelKind {
        self.inner.kind
    }

    /// Which integer activation layout applies, if any.
    ///
    /// Requires the AVX2 kernel set and a column count that divides evenly into
    /// the layout's block size, so no partial block is ever left unscaled.
    fn integer_layout(&self, kind: KernelKind, weight: &QuantTensor) -> ActivationLayout {
        if !self.inner.integer_activations || kind != KernelKind::Avx2 {
            return ActivationLayout::None;
        }
        match activation_layout(weight.quant_type) {
            ActivationLayout::Q8K if weight.cols().is_multiple_of(QK_K) => ActivationLayout::Q8K,
            ActivationLayout::Q8_32 if weight.cols().is_multiple_of(32) => ActivationLayout::Q8_32,
            _ => ActivationLayout::None,
        }
    }

    /// Shared row loop, so the serial and parallel policies live in one place
    /// instead of being duplicated per kernel.
    fn run_rows<F>(&self, parallel: bool, output: &mut [f32], compute: F)
    where
        F: Fn(usize) -> f32 + Sync + Send,
    {
        if parallel {
            self.inner.pool.install(|| {
                output
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(row, out)| *out = compute(row));
            });
        } else {
            for (row, out) in output.iter_mut().enumerate() {
                *out = compute(row);
            }
        }
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
        let parallel = self.inner.threads > 1
            && weight.rows() >= self.inner.threads * 2
            && work >= self.inner.parallel_threshold;

        // Quantizing the activation is worth it only when it is amortized over
        // many rows, so it happens once per matrix here rather than per row.
        match self.integer_layout(kind, weight) {
            ActivationLayout::Q8K => {
                let ty = weight.quant_type;
                let mut scratch = take_q8k_scratch();
                quantize_activation_q8k(input, &mut scratch);
                let activation: &[BlockQ8K] = &scratch;
                self.run_rows(parallel, output, |row| {
                    let data = weight.row_data(row);
                    dot_row_avx2_q8k(data, ty, activation)
                        .unwrap_or_else(|| dot_row_for_kernel(kind, data, ty, input))
                });
                restore_q8k_scratch(scratch);
                return Ok(());
            }
            ActivationLayout::Q8_32 => {
                let ty = weight.quant_type;
                let mut scratch = take_q8_32_scratch();
                quantize_activation_q8_32(input, &mut scratch);
                let activation: &[BlockQ8_32] = &scratch;
                self.run_rows(parallel, output, |row| {
                    let data = weight.row_data(row);
                    dot_row_avx2_q8_32(data, ty, activation)
                        .unwrap_or_else(|| dot_row_for_kernel(kind, data, ty, input))
                });
                restore_q8_32_scratch(scratch);
                return Ok(());
            }
            ActivationLayout::None => {}
        }

        if parallel {
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

thread_local! {
    /// Reused across matvec calls so the hot path performs no allocation.
    static Q8K_SCRATCH: RefCell<Vec<BlockQ8K>> = const { RefCell::new(Vec::new()) };
    static Q8_32_SCRATCH: RefCell<Vec<BlockQ8_32>> = const { RefCell::new(Vec::new()) };
}

/// Moving the buffer out and back keeps the borrow short and makes reentrant
/// calls allocate rather than panic. `mem::take` hands over the existing
/// allocation; `split_off(0)` would have allocated on every call.
fn take_q8k_scratch() -> Vec<BlockQ8K> {
    Q8K_SCRATCH.with(|cell| {
        cell.try_borrow_mut()
            .map(|mut slot| std::mem::take(&mut *slot))
            .unwrap_or_default()
    })
}

fn take_q8_32_scratch() -> Vec<BlockQ8_32> {
    Q8_32_SCRATCH.with(|cell| {
        cell.try_borrow_mut()
            .map(|mut slot| std::mem::take(&mut *slot))
            .unwrap_or_default()
    })
}

fn restore_q8k_scratch(buffer: Vec<BlockQ8K>) {
    Q8K_SCRATCH.with(|cell| {
        if let Ok(mut slot) = cell.try_borrow_mut() {
            // Keep whichever allocation is larger so the steady state stops
            // growing after the widest matrix has been seen once.
            if buffer.capacity() >= slot.capacity() {
                *slot = buffer;
            }
        }
    });
}

fn restore_q8_32_scratch(buffer: Vec<BlockQ8_32>) {
    Q8_32_SCRATCH.with(|cell| {
        if let Ok(mut slot) = cell.try_borrow_mut() {
            if buffer.capacity() >= slot.capacity() {
                *slot = buffer;
            }
        }
    });
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
