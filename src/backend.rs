use crate::activation::{
    activation_layout, quantize_activation_q8_32, quantize_activation_q8k, ActivationLayout,
    BlockQ8K, BlockQ8_32, QK_K,
};
use crate::quant::{dot_row_for_kernel, QuantTensor};
use crate::simd::{dot_row_avx2_q8_32, dot_row_avx2_q8k};
use anyhow::{bail, ensure, Result};
use std::cell::RefCell;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

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

type RunnerFn = unsafe fn(*const (), usize, usize);

struct PoolShared {
    generation: AtomicUsize,
    done: AtomicUsize,
    shutdown: AtomicBool,
    parked: AtomicUsize,
    lock: Mutex<()>,
    cvar: Condvar,
    task_ptr: AtomicPtr<()>,
    task_runner: AtomicPtr<()>,
}

/// Low-latency persistent thread pool with atomic spin-barrier.
///
/// Avoids the ~50 µs per-matvec fork-join overhead of generic task pools.
/// Worker threads stay alive for the backend lifetime, spin on an atomic
/// generation counter during tight token generation loops, and fall back to
/// sleeping when idle.
pub struct PersistentPool {
    threads: usize,
    shared: Arc<PoolShared>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    execution_lock: Mutex<()>,
}

fn worker_loop(thread_idx: usize, num_threads: usize, shared: Arc<PoolShared>) {
    let mut last_gen = 0;
    loop {
        let mut spins = 0;
        loop {
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
            let current_gen = shared.generation.load(Ordering::Acquire);
            if current_gen != last_gen {
                last_gen = current_gen;
                break;
            }
            if spins < 2000 {
                std::hint::spin_loop();
                spins += 1;
            } else if spins < 10000 {
                thread::yield_now();
                spins += 1;
            } else {
                let mut guard = shared.lock.lock().unwrap();
                while shared.generation.load(Ordering::Acquire) == last_gen
                    && !shared.shutdown.load(Ordering::Acquire)
                {
                    shared.parked.fetch_add(1, Ordering::Release);
                    guard = shared.cvar.wait(guard).unwrap();
                    shared.parked.fetch_sub(1, Ordering::Release);
                }
            }
        }

        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }

        let task_ptr = shared.task_ptr.load(Ordering::Acquire);
        let runner_ptr = shared.task_runner.load(Ordering::Acquire);
        if !task_ptr.is_null() && !runner_ptr.is_null() {
            let runner: RunnerFn = unsafe { std::mem::transmute(runner_ptr) };
            unsafe {
                runner(task_ptr as *const (), thread_idx, num_threads);
            }
        }

        shared.done.fetch_add(1, Ordering::Release);
    }
}

impl PersistentPool {
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(PoolShared {
            generation: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            parked: AtomicUsize::new(0),
            lock: Mutex::new(()),
            cvar: Condvar::new(),
            task_ptr: AtomicPtr::new(std::ptr::null_mut()),
            task_runner: AtomicPtr::new(std::ptr::null_mut()),
        });

        let mut workers = Vec::with_capacity(threads.saturating_sub(1));
        for thread_idx in 1..threads {
            let shared_clone = Arc::clone(&shared);
            let handle = thread::Builder::new()
                .name(format!("llama-rs-worker-{thread_idx}"))
                .spawn(move || {
                    worker_loop(thread_idx, threads, shared_clone);
                })
                .expect("failed to spawn worker thread");
            workers.push(handle);
        }

        Self {
            threads,
            shared,
            workers: Mutex::new(workers),
            execution_lock: Mutex::new(()),
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn execute<F>(&self, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        if self.threads <= 1 {
            f(0, 1);
            return;
        }

        let _exec_guard = self.execution_lock.lock().unwrap();

        unsafe fn runner_trampoline<F: Fn(usize, usize) + Sync>(
            data: *const (),
            thread_idx: usize,
            num_threads: usize,
        ) {
            let closure = &*(data as *const F);
            closure(thread_idx, num_threads);
        }

        let task_data = &f as *const F as *mut ();
        let runner_fn: RunnerFn = runner_trampoline::<F>;
        let runner_ptr = runner_fn as usize as *mut ();

        self.shared.task_ptr.store(task_data, Ordering::Release);
        self.shared.task_runner.store(runner_ptr, Ordering::Release);
        self.shared.done.store(0, Ordering::Release);

        self.shared.generation.fetch_add(1, Ordering::Release);

        if self.shared.parked.load(Ordering::Acquire) > 0 {
            let _guard = self.shared.lock.lock().unwrap();
            self.shared.cvar.notify_all();
        }

        // Main thread executes partition 0
        f(0, self.threads);

        // Spin-wait for all worker threads
        let target = self.threads - 1;
        let mut spins = 0;
        while self.shared.done.load(Ordering::Acquire) < target {
            if spins < 10000 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                thread::yield_now();
            }
        }

        self.shared.task_ptr.store(std::ptr::null_mut(), Ordering::Release);
        self.shared.task_runner.store(std::ptr::null_mut(), Ordering::Release);
    }
}

impl Drop for PersistentPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.generation.fetch_add(1, Ordering::Release);
        {
            let _guard = self.shared.lock.lock().unwrap();
            self.shared.cvar.notify_all();
        }
        if let Ok(mut workers) = self.workers.lock() {
            for handle in workers.drain(..) {
                let _ = handle.join();
            }
        }
    }
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
    pool: PersistentPool,
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
        let pool = PersistentPool::new(threads);
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

    /// Shared row loop using the persistent thread pool with contiguous chunk partitioning.
    fn run_rows<F>(&self, parallel: bool, output: &mut [f32], compute: F)
    where
        F: Fn(usize) -> f32 + Sync + Send,
    {
        if parallel {
            let num_rows = output.len();
            let out_ptr = output.as_mut_ptr() as usize;
            self.inner.pool.execute(|thread_idx, num_threads| {
                let start = thread_idx * num_rows / num_threads;
                let end = ((thread_idx + 1) * num_rows / num_threads).min(num_rows);
                if start < end {
                    let out = unsafe {
                        std::slice::from_raw_parts_mut(
                            (out_ptr as *mut f32).add(start),
                            end - start,
                        )
                    };
                    for (offset, dst) in out.iter_mut().enumerate() {
                        let row = start + offset;
                        *dst = compute(row);
                    }
                }
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

        self.run_rows(parallel, output, |row| {
            dot_row_for_kernel(kind, weight.row_data(row), weight.quant_type, input)
        });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_pool_executes_parallel_work() {
        let pool = PersistentPool::new(4);
        let mut values = vec![0u32; 1000];
        let val_ptr = values.as_mut_ptr() as usize;

        pool.execute(|thread_idx, num_threads| {
            let start = thread_idx * 1000 / num_threads;
            let end = ((thread_idx + 1) * 1000 / num_threads).min(1000);
            let slice = unsafe {
                std::slice::from_raw_parts_mut((val_ptr as *mut u32).add(start), end - start)
            };
            for (i, v) in slice.iter_mut().enumerate() {
                *v = (start + i) as u32;
            }
        });

        for (i, &v) in values.iter().enumerate() {
            assert_eq!(v, i as u32);
        }
    }

    #[test]
    fn persistent_pool_single_thread_fallback() {
        let pool = PersistentPool::new(1);
        let mut executed = false;
        pool.execute(|tid, nt| {
            assert_eq!(tid, 0);
            assert_eq!(nt, 1);
            executed = true;
        });
        assert!(executed);
    }

    #[test]
    fn persistent_pool_multiple_sequential_rounds() {
        let pool = PersistentPool::new(2);
        let mut counter = 0u64;
        let cnt_ptr = &mut counter as *mut u64 as usize;

        for _ in 0..1000 {
            pool.execute(|tid, _nt| {
                if tid == 0 {
                    unsafe { *(cnt_ptr as *mut u64) += 1 };
                }
            });
        }
        assert_eq!(counter, 1000);
    }

    #[test]
    fn persistent_pool_parks_and_wakes() {
        let pool = PersistentPool::new(2);
        // Let workers spin down and park
        thread::sleep(std::time::Duration::from_millis(20));

        let mut done = false;
        pool.execute(|_tid, _nt| {
            done = true;
        });
        assert!(done);
    }

    #[test]
    fn cpu_backend_multithreaded_matches_single_threaded() {
        let config_serial = BackendConfig {
            threads: 1,
            integer_activations: false,
            ..BackendConfig::default()
        };
        let config_parallel = BackendConfig {
            threads: 4,
            parallel_threshold: 0, // force parallel
            integer_activations: false,
            ..BackendConfig::default()
        };

        let backend_serial = CpuBackend::new(&config_serial).unwrap();
        let backend_parallel = CpuBackend::new(&config_parallel).unwrap();

        let rows = 64;
        let cols = 256;
        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.05).sin()).collect();
        let f32_data: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.03).cos())
            .collect();
        let tensor = QuantTensor::quantize_q4k(&f32_data, &[rows, cols]).unwrap();

        let mut out_serial = vec![0.0f32; rows];
        let mut out_parallel = vec![0.0f32; rows];

        backend_serial.matvec(&tensor, &input, &mut out_serial).unwrap();
        backend_parallel.matvec(&tensor, &input, &mut out_parallel).unwrap();

        for (s, p) in out_serial.iter().zip(&out_parallel) {
            assert!((s - p).abs() < 1e-5, "serial {s} vs parallel {p}");
        }
    }
}
