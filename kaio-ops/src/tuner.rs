//! Auto-tuner: benchmark kernel variants, cache results, dispatch.
//!
//! Grid search over fixed parameter sets. No heuristics, no adaptive
//! logic, no dynamic tuning. Users explicitly call `tune_*()` to
//! benchmark; `*_auto()` dispatches from cache or falls back to default.
//!
//! Cache stored as JSON at `~/.cache/kaio/tune_cache.json`
//! (override with `KAIO_TUNE_CACHE` env var).

use std::path::PathBuf;
use std::time::Instant;

use half::f16;
use kaio::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    attention, attention_causal, attention_flash, attention_flash_causal, matmul, matmul_tc,
    matmul_tc_async,
};

// Expose naive kernel for tuning comparison
use crate::matmul_naive;

// ---------------------------------------------------------------------------
// Variant enums — compile-time exhaustiveness
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatmulVariant {
    Naive16x16,
    Optimized64x64,
}

impl MatmulVariant {
    fn as_str(self) -> &'static str {
        match self {
            Self::Naive16x16 => "naive_16x16",
            Self::Optimized64x64 => "optimized_64x64",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "naive_16x16" => Some(Self::Naive16x16),
            "optimized_64x64" => Some(Self::Optimized64x64),
            _ => None,
        }
    }

    fn all() -> &'static [Self] {
        &[Self::Naive16x16, Self::Optimized64x64]
    }
}

/// Tensor-core matmul variants dispatched by `matmul_auto_tc`
/// (Sprint 6.5). Both variants have identical eligibility gates
/// (SM 8.0+ AND `M%16 = N%8 = K%16 = 0`), so pre-dispatch checks
/// live at the tuner level rather than per-variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatmulTcVariant {
    TensorCore,
    TensorCoreAsync,
}

impl MatmulTcVariant {
    fn as_str(self) -> &'static str {
        match self {
            Self::TensorCore => "tensor_core",
            Self::TensorCoreAsync => "tensor_core_async",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "tensor_core" => Some(Self::TensorCore),
            "tensor_core_async" => Some(Self::TensorCoreAsync),
            _ => None,
        }
    }

    fn all() -> &'static [Self] {
        &[Self::TensorCore, Self::TensorCoreAsync]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttentionVariant {
    Standard,
    Flash,
}

impl AttentionVariant {
    fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Flash => "flash",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "standard" => Some(Self::Standard),
            "flash" => Some(Self::Flash),
            _ => None,
        }
    }

    fn eligible(d_k: u32) -> Vec<Self> {
        let mut v = vec![Self::Standard];
        if d_k <= 256 {
            v.push(Self::Flash);
        }
        v
    }
}

// ---------------------------------------------------------------------------
// Cache types
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct TuneCache {
    version: u32,
    results: Vec<TuneResult>,
}

#[derive(Serialize, Deserialize, Clone)]
struct TuneResult {
    kernel: String,
    variant: String,
    sm_target: String,
    dims: Vec<u32>,
    median_ms: f64,
}

impl TuneCache {
    fn empty() -> Self {
        Self {
            version: 1,
            results: Vec::new(),
        }
    }

    fn lookup(&self, kernel: &str, sm_target: &str, dims: &[u32]) -> Option<&TuneResult> {
        self.results
            .iter()
            .find(|r| r.kernel == kernel && r.sm_target == sm_target && r.dims == dims)
    }

    fn upsert(&mut self, result: TuneResult) {
        // Overwrite existing matching key, collapse duplicates
        self.results.retain(|r| {
            !(r.kernel == result.kernel && r.sm_target == result.sm_target && r.dims == result.dims)
        });
        self.results.push(result);
    }
}

fn cache_path() -> PathBuf {
    if let Ok(p) = std::env::var("KAIO_TUNE_CACHE") {
        PathBuf::from(p)
    } else {
        dirs_fallback().join("tune_cache.json")
    }
}

fn dirs_fallback() -> PathBuf {
    // ~/.cache/kaio/ on Unix, %LOCALAPPDATA%/kaio/ on Windows
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cache").join("kaio")
    } else if let Ok(local) = std::env::var("LOCALAPPDATA") {
        PathBuf::from(local).join("kaio")
    } else {
        PathBuf::from(".kaio_cache")
    }
}

fn load_cache() -> TuneCache {
    let path = cache_path();
    let data = match std::fs::read_to_string(&path) {
        Ok(d) => d,
        Err(_) => return TuneCache::empty(),
    };
    match serde_json::from_str::<TuneCache>(&data) {
        Ok(c) if c.version == 1 => c,
        _ => TuneCache::empty(), // corrupted or version mismatch
    }
}

fn save_cache(cache: &TuneCache) -> Result<()> {
    let path = cache_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| KaioError::InvalidConfig(format!("failed to create cache dir: {e}")))?;
    }
    let json = serde_json::to_string_pretty(cache)
        .map_err(|e| KaioError::InvalidConfig(format!("failed to serialize cache: {e}")))?;
    std::fs::write(&path, json)
        .map_err(|e| KaioError::InvalidConfig(format!("failed to write cache: {e}")))?;
    Ok(())
}

fn sm_target(device: &KaioDevice) -> Result<String> {
    let info = device.info()?;
    Ok(format!(
        "sm_{}{}",
        info.compute_capability.0, info.compute_capability.1
    ))
}

// ---------------------------------------------------------------------------
// Benchmarking helpers
// ---------------------------------------------------------------------------

const WARMUP: usize = 3;
const ITERS: usize = 10;

fn launch_matmul(
    device: &KaioDevice,
    variant: MatmulVariant,
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    match variant {
        MatmulVariant::Naive16x16 => matmul_naive(device, a, b, c, m, n, k),
        MatmulVariant::Optimized64x64 => matmul(device, a, b, c, m, n, k),
    }
}

fn bench_matmul_variant(
    device: &KaioDevice,
    variant: MatmulVariant,
    m: u32,
    n: u32,
    k: u32,
) -> Result<f64> {
    let a = device.alloc_zeros::<f32>((m as usize) * (k as usize))?;
    let b = device.alloc_zeros::<f32>((k as usize) * (n as usize))?;
    let mut c = device.alloc_zeros::<f32>((m as usize) * (n as usize))?;

    for _ in 0..WARMUP {
        launch_matmul(device, variant, &a, &b, &mut c, m, n, k)?;
    }
    device.stream().synchronize()?;

    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        device.stream().synchronize()?;
        let start = Instant::now();
        launch_matmul(device, variant, &a, &b, &mut c, m, n, k)?;
        device.stream().synchronize()?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(times[ITERS / 2])
}

fn launch_attention(
    device: &KaioDevice,
    variant: AttentionVariant,
    causal: bool,
    q: &GpuBuffer<f32>,
    k_buf: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    out: &mut GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
) -> Result<()> {
    match (variant, causal) {
        (AttentionVariant::Standard, false) => attention(device, q, k_buf, v, out, seq_len, d_k),
        (AttentionVariant::Standard, true) => {
            attention_causal(device, q, k_buf, v, out, seq_len, d_k)
        }
        (AttentionVariant::Flash, false) => attention_flash(device, q, k_buf, v, out, seq_len, d_k),
        (AttentionVariant::Flash, true) => {
            attention_flash_causal(device, q, k_buf, v, out, seq_len, d_k)
        }
    }
}

fn bench_attention_variant(
    device: &KaioDevice,
    variant: AttentionVariant,
    causal: bool,
    seq_len: u32,
    d_k: u32,
) -> Result<f64> {
    let sd = (seq_len as usize) * (d_k as usize);
    let q = device.alloc_zeros::<f32>(sd)?;
    let k_buf = device.alloc_zeros::<f32>(sd)?;
    let v = device.alloc_zeros::<f32>(sd)?;
    let mut out = device.alloc_zeros::<f32>(sd)?;

    for _ in 0..WARMUP {
        launch_attention(
            device, variant, causal, &q, &k_buf, &v, &mut out, seq_len, d_k,
        )?;
    }
    device.stream().synchronize()?;

    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        device.stream().synchronize()?;
        let start = Instant::now();
        launch_attention(
            device, variant, causal, &q, &k_buf, &v, &mut out, seq_len, d_k,
        )?;
        device.stream().synchronize()?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(times[ITERS / 2])
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Benchmark matmul variants at the given dimensions.
/// Returns the fastest variant name and caches the result.
pub fn tune_matmul(device: &KaioDevice, m: u32, n: u32, k: u32) -> Result<String> {
    let sm = sm_target(device)?;
    let dims = vec![m, n, k];

    let mut best_variant = MatmulVariant::Optimized64x64;
    let mut best_time = f64::MAX;

    for &variant in MatmulVariant::all() {
        let time = bench_matmul_variant(device, variant, m, n, k)?;
        eprintln!(
            "  tune matmul {}: {:.3} ms ({m}×{n}×{k})",
            variant.as_str(),
            time
        );
        if time < best_time {
            best_time = time;
            best_variant = variant;
        }
    }

    let mut cache = load_cache();
    cache.upsert(TuneResult {
        kernel: "matmul".to_string(),
        variant: best_variant.as_str().to_string(),
        sm_target: sm,
        dims,
        median_ms: best_time,
    });
    save_cache(&cache)?;

    Ok(best_variant.as_str().to_string())
}

/// Benchmark attention variants at the given dimensions.
/// Returns the fastest variant name and caches the result.
/// Skips flash variant if d_k > 256 (ineligible).
pub fn tune_attention(device: &KaioDevice, seq_len: u32, d_k: u32) -> Result<String> {
    tune_attention_inner(device, false, seq_len, d_k)
}

/// Benchmark causal attention variants.
pub fn tune_attention_causal(device: &KaioDevice, seq_len: u32, d_k: u32) -> Result<String> {
    tune_attention_inner(device, true, seq_len, d_k)
}

fn tune_attention_inner(
    device: &KaioDevice,
    causal: bool,
    seq_len: u32,
    d_k: u32,
) -> Result<String> {
    let sm = sm_target(device)?;
    let kernel_name = if causal {
        "attention_causal"
    } else {
        "attention"
    };
    let dims = vec![seq_len, d_k];

    let eligible = AttentionVariant::eligible(d_k);
    let mut best_variant = AttentionVariant::Standard;
    let mut best_time = f64::MAX;

    for &variant in &eligible {
        let time = bench_attention_variant(device, variant, causal, seq_len, d_k)?;
        eprintln!(
            "  tune {} {}: {:.3} ms ({seq_len}×{d_k})",
            kernel_name,
            variant.as_str(),
            time
        );
        if time < best_time {
            best_time = time;
            best_variant = variant;
        }
    }

    let mut cache = load_cache();
    cache.upsert(TuneResult {
        kernel: kernel_name.to_string(),
        variant: best_variant.as_str().to_string(),
        sm_target: sm,
        dims,
        median_ms: best_time,
    });
    save_cache(&cache)?;

    Ok(best_variant.as_str().to_string())
}

/// Run matmul using the best tuned variant. Falls back to optimized
/// if no cached result exists. Pure dispatch — no side effects.
pub fn matmul_auto(
    device: &KaioDevice,
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    let variant = resolve_matmul_variant(device, m, n, k);
    match variant {
        MatmulVariant::Naive16x16 => matmul_naive(device, a, b, c, m, n, k),
        MatmulVariant::Optimized64x64 => matmul(device, a, b, c, m, n, k),
    }
}

/// Run attention using the best tuned variant. Falls back to standard.
pub fn attention_auto(
    device: &KaioDevice,
    q: &GpuBuffer<f32>,
    k_buf: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    out: &mut GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
) -> Result<()> {
    let variant = resolve_attention_variant(device, "attention", seq_len, d_k);
    match variant {
        AttentionVariant::Standard => attention(device, q, k_buf, v, out, seq_len, d_k),
        AttentionVariant::Flash => attention_flash(device, q, k_buf, v, out, seq_len, d_k),
    }
}

/// Run causal attention using the best tuned variant. Falls back to standard.
pub fn attention_auto_causal(
    device: &KaioDevice,
    q: &GpuBuffer<f32>,
    k_buf: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    out: &mut GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
) -> Result<()> {
    let variant = resolve_attention_variant(device, "attention_causal", seq_len, d_k);
    match variant {
        AttentionVariant::Standard => attention_causal(device, q, k_buf, v, out, seq_len, d_k),
        AttentionVariant::Flash => attention_flash_causal(device, q, k_buf, v, out, seq_len, d_k),
    }
}

// ---------------------------------------------------------------------------
// Internal dispatch resolution
// ---------------------------------------------------------------------------

fn resolve_matmul_variant(device: &KaioDevice, m: u32, n: u32, k: u32) -> MatmulVariant {
    let sm = match sm_target(device) {
        Ok(s) => s,
        Err(_) => return MatmulVariant::Optimized64x64,
    };
    let cache = load_cache();
    match cache.lookup("matmul", &sm, &[m, n, k]) {
        Some(r) => MatmulVariant::from_str(&r.variant).unwrap_or(MatmulVariant::Optimized64x64),
        None => MatmulVariant::Optimized64x64,
    }
}

fn resolve_attention_variant(
    device: &KaioDevice,
    kernel: &str,
    seq_len: u32,
    d_k: u32,
) -> AttentionVariant {
    let sm = match sm_target(device) {
        Ok(s) => s,
        Err(_) => return AttentionVariant::Standard,
    };
    let cache = load_cache();
    match cache.lookup(kernel, &sm, &[seq_len, d_k]) {
        Some(r) => {
            let v = AttentionVariant::from_str(&r.variant).unwrap_or(AttentionVariant::Standard);
            // Safety: don't dispatch flash if d_k > 256 even if cached
            if v == AttentionVariant::Flash && d_k > 256 {
                AttentionVariant::Standard
            } else {
                v
            }
        }
        None => AttentionVariant::Standard,
    }
}

// ---------------------------------------------------------------------------
// Tensor-core matmul tuner (Sprint 6.5)
// ---------------------------------------------------------------------------

/// K-dimension constraint, shared with `matmul_tc_kernel::validate_dims_tc`.
/// Sprint 6.7 Gate C: M and N are unconstrained (edge-tile predication
/// inside the kernel handles ragged dims). K stays %16=0 because the
/// mma.sync.m16n8k16 instance shape has a fixed K-tile of 16.
const TC_K_STEP: u32 = 16;

/// Pre-dispatch eligibility gate for the TC tuner.
fn check_tc_eligibility(device: &KaioDevice, m: u32, n: u32, k: u32) -> Result<()> {
    let info = device.info()?;
    let (major, minor) = info.compute_capability;
    if major < 8 {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_auto_tc requires SM 8.0+ (Ampere). \
             GPU compute capability is {major}.{minor}. \
             For pre-Ampere hardware, either convert inputs to f32 and \
             use matmul_auto, or upgrade the device."
        )));
    }
    if m == 0 || n == 0 || k == 0 {
        return Err(KaioError::InvalidConfig(
            "matmul_auto_tc dimensions must be non-zero".to_string(),
        ));
    }
    if !k.is_multiple_of(TC_K_STEP) {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_auto_tc requires K%{TC_K_STEP}=0 (got K={k}). The mma.sync.m16n8k16 \
             instance shape has a fixed K-tile of 16; the kernel does not edge-pad K. \
             Pad K to the next multiple of 16, or use the f32 matmul path (matmul_auto)."
        )));
    }
    Ok(())
}

fn launch_matmul_tc(
    device: &KaioDevice,
    variant: MatmulTcVariant,
    a: &GpuBuffer<f16>,
    b: &GpuBuffer<f16>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    match variant {
        MatmulTcVariant::TensorCore => matmul_tc(device, a, b, c, m, n, k),
        MatmulTcVariant::TensorCoreAsync => matmul_tc_async(device, a, b, c, m, n, k),
    }
}

fn bench_matmul_tc_variant(
    device: &KaioDevice,
    variant: MatmulTcVariant,
    m: u32,
    n: u32,
    k: u32,
) -> Result<f64> {
    let mk = (m as usize) * (k as usize);
    let kn = (k as usize) * (n as usize);
    let mn = (m as usize) * (n as usize);
    let a = device.alloc_zeros::<f16>(mk)?;
    let b = device.alloc_zeros::<f16>(kn)?;
    let mut c = device.alloc_zeros::<f32>(mn)?;

    for _ in 0..WARMUP {
        launch_matmul_tc(device, variant, &a, &b, &mut c, m, n, k)?;
    }
    device.stream().synchronize()?;

    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        device.stream().synchronize()?;
        let start = Instant::now();
        launch_matmul_tc(device, variant, &a, &b, &mut c, m, n, k)?;
        device.stream().synchronize()?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(times[ITERS / 2])
}

/// Benchmark the two tensor-core matmul variants at the given
/// dimensions and cache the faster one.
///
/// Requires SM 8.0+ (Ampere) and `K % 16 == 0` (the mma.sync K-tile
/// is structural; the kernel does not edge-pad K). M and N may be
/// any positive value — edge-tile predication in the kernel handles
/// non-multiple-of-64 cases.
///
/// Cached result key: `(kernel="matmul_tc", sm_target, [m, n, k])`.
/// The cache file is shared with [`tune_matmul`]; the `kernel` field
/// disambiguates the two entry points.
pub fn tune_matmul_tc(device: &KaioDevice, m: u32, n: u32, k: u32) -> Result<String> {
    check_tc_eligibility(device, m, n, k)?;

    let sm = sm_target(device)?;
    let dims = vec![m, n, k];

    let mut best_variant = MatmulTcVariant::TensorCoreAsync;
    let mut best_time = f64::MAX;

    for &variant in MatmulTcVariant::all() {
        let time = bench_matmul_tc_variant(device, variant, m, n, k)?;
        eprintln!(
            "  tune matmul_tc {}: {:.3} ms ({m}×{n}×{k})",
            variant.as_str(),
            time
        );
        if time < best_time {
            best_time = time;
            best_variant = variant;
        }
    }

    let mut cache = load_cache();
    cache.upsert(TuneResult {
        kernel: "matmul_tc".to_string(),
        variant: best_variant.as_str().to_string(),
        sm_target: sm,
        dims,
        median_ms: best_time,
    });
    save_cache(&cache)?;

    Ok(best_variant.as_str().to_string())
}

/// Run the tensor-core matmul using the best tuned variant, or a
/// size-heuristic default if the cache has no entry for these
/// dimensions.
///
/// # Contract
///
/// - **Input type:** `f16 × f16 → f32` with fp32 accumulation. f32
///   callers should use [`matmul_auto`] instead.
/// - **Hardware:** NVIDIA Ampere or newer (SM 8.0+). Pre-Ampere
///   devices return [`KaioError::InvalidConfig`] naming the f32
///   fallback.
/// - **Shape:** M and N may be any positive value (edge-tile
///   predication handles ragged dims). `K % 16 == 0` is required —
///   the mma.sync.m16n8k16 K-tile is structural and the kernel does
///   not edge-pad K. Pad K to the next multiple of 16 if needed.
/// - **Cache-miss dispatch policy:** if `max(M, N, K) >= 3072` the
///   fallback is the cp.async double-buffered variant (async wins at
///   4096² by ~12.4% after 6.7b's shared-memory improvements).
///   Otherwise the sync variant is dispatched (it wins at 256-2048
///   by small margins on the measured curve). A prior
///   `tune_matmul_tc` call overrides this with the per-shape
///   measured winner.
/// - **Performance:** Sprint 6.7 multi-warp restructure plus Sprint
///   6.7b bank-conflict-padded Tile B (col stride 32 → 36 bytes) and
///   fragment-loader `(group_id, tig)` hoist measure **82.3%** of
///   cuBLAS sgemm at 4096² for the sync variant and **92.5%** for
///   the cp.async double-buffered variant, on RTX 4090 sm_89. See
///   `docs/performance.md` for the full table (256–4096) and the
///   project-local-baseline disclaimer (KAIO uses fp16 inputs with
///   fp32 accumulation; cuBLAS sgemm is f32).
/// - **API stability:** pre-1.0 (overall crate). The signature is
///   intentionally identical in shape to [`matmul_auto`] so any
///   future extensions remain additive.
///
/// # Errors
///
/// Returns [`KaioError::InvalidConfig`] on pre-Ampere hardware,
/// zero-sized dims, or `K % 16 != 0` — with a message naming the
/// actionable fallback (pad K, or use the f32 [`matmul_auto`] path
/// if f16 precision is not required).
pub fn matmul_auto_tc(
    device: &KaioDevice,
    a: &GpuBuffer<f16>,
    b: &GpuBuffer<f16>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    check_tc_eligibility(device, m, n, k)?;

    let variant = resolve_matmul_tc_variant(device, m, n, k);
    launch_matmul_tc(device, variant, a, b, c, m, n, k)
}

/// Size threshold where the cp.async double-buffered variant overtakes
/// the sync variant on the measured curve. Below this, sync wins by
/// small margins (256-2048 at RTX 4090 sm_89); at/above, async wins by
/// ~12.4% at 4096² under the 6.7b bank-conflict-padded layout (was
/// ~6.5% under 6.7). One line of arithmetic is cheaper than a
/// consistently-wrong small-shape default.
const ASYNC_FALLBACK_MAX_DIM_THRESHOLD: u32 = 3072;

fn cache_miss_default(m: u32, n: u32, k: u32) -> MatmulTcVariant {
    if m.max(n).max(k) >= ASYNC_FALLBACK_MAX_DIM_THRESHOLD {
        MatmulTcVariant::TensorCoreAsync
    } else {
        MatmulTcVariant::TensorCore
    }
}

fn resolve_matmul_tc_variant(device: &KaioDevice, m: u32, n: u32, k: u32) -> MatmulTcVariant {
    let sm = match sm_target(device) {
        Ok(s) => s,
        // D6 post-review (2026-04-12): size-heuristic fallback.
        // Sprint 6.7 bench shows sync wins 256-2048, async wins 4096².
        // A flat TensorCoreAsync default was wrong for 4 of 5 measured
        // sizes. The heuristic matches all 5 with one comparison.
        Err(_) => return cache_miss_default(m, n, k),
    };
    let cache = load_cache();
    match cache.lookup("matmul_tc", &sm, &[m, n, k]) {
        Some(r) => {
            MatmulTcVariant::from_str(&r.variant).unwrap_or_else(|| cache_miss_default(m, n, k))
        }
        // Cache miss: use the size heuristic (see above).
        None => cache_miss_default(m, n, k),
    }
}

#[cfg(test)]
mod tc_tuner_tests {
    use super::*;

    #[test]
    fn cache_miss_default_matches_sprint_6_7_bench_curve() {
        // Sync wins the 256-2048 range on the measured 6.7 curve; async
        // wins at 4096. The threshold (3072) must land in the gap.
        assert_eq!(
            cache_miss_default(256, 256, 256),
            MatmulTcVariant::TensorCore
        );
        assert_eq!(
            cache_miss_default(512, 512, 512),
            MatmulTcVariant::TensorCore
        );
        assert_eq!(
            cache_miss_default(1024, 1024, 1024),
            MatmulTcVariant::TensorCore
        );
        assert_eq!(
            cache_miss_default(2048, 2048, 2048),
            MatmulTcVariant::TensorCore
        );
        assert_eq!(
            cache_miss_default(4096, 4096, 4096),
            MatmulTcVariant::TensorCoreAsync
        );
        // Only one dim at or above threshold is enough — tall/skinny
        // matmuls with a large K still benefit from async's pipelining.
        assert_eq!(
            cache_miss_default(128, 128, 4096),
            MatmulTcVariant::TensorCoreAsync
        );
        // Exact threshold is inclusive.
        assert_eq!(
            cache_miss_default(3072, 64, 64),
            MatmulTcVariant::TensorCoreAsync
        );
    }

    #[test]
    fn matmul_tc_variant_as_str_from_str_roundtrip() {
        for &v in MatmulTcVariant::all() {
            let s = v.as_str();
            let back = MatmulTcVariant::from_str(s).expect("round-trip");
            assert_eq!(v, back);
        }
        assert!(MatmulTcVariant::from_str("unknown_variant").is_none());
    }

    #[test]
    fn tune_result_json_roundtrip_matmul_tc_variant() {
        // Uses the same TuneResult format as scalar matmul — the only
        // thing varying is the variant string. Ensures the TC variant
        // names survive a JSON serialize/deserialize cycle.
        let result = TuneResult {
            kernel: "matmul_tc".to_string(),
            variant: MatmulTcVariant::TensorCoreAsync.as_str().to_string(),
            sm_target: "sm_89".to_string(),
            dims: vec![64, 64, 64],
            median_ms: 0.27,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: TuneResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kernel, "matmul_tc");
        assert_eq!(back.variant, "tensor_core_async");
        assert_eq!(back.dims, vec![64, 64, 64]);
        assert_eq!(
            MatmulTcVariant::from_str(&back.variant),
            Some(MatmulTcVariant::TensorCoreAsync)
        );
    }

    #[test]
    fn cache_matmul_and_matmul_tc_entries_coexist() {
        // Sprint 6.5 D6 / review feedback: scalar matmul and TC matmul
        // entries share the cache file, disambiguated by the `kernel`
        // field. This test writes one of each for the same
        // (sm_target, dims) tuple and verifies:
        //   1. Both persist (neither overwrites the other).
        //   2. `lookup(kernel="matmul", ...)` returns the scalar entry.
        //   3. `lookup(kernel="matmul_tc", ...)` returns the TC entry.
        let mut cache = TuneCache::empty();
        cache.upsert(TuneResult {
            kernel: "matmul".to_string(),
            variant: "optimized_64x64".to_string(),
            sm_target: "sm_89".to_string(),
            dims: vec![64, 64, 64],
            median_ms: 0.42,
        });
        cache.upsert(TuneResult {
            kernel: "matmul_tc".to_string(),
            variant: "tensor_core".to_string(),
            sm_target: "sm_89".to_string(),
            dims: vec![64, 64, 64],
            median_ms: 0.25,
        });

        assert_eq!(
            cache.results.len(),
            2,
            "both entries should persist; got {}",
            cache.results.len()
        );

        let scalar = cache
            .lookup("matmul", "sm_89", &[64, 64, 64])
            .expect("scalar matmul entry should be findable");
        assert_eq!(scalar.variant, "optimized_64x64");

        let tc = cache
            .lookup("matmul_tc", "sm_89", &[64, 64, 64])
            .expect("TC matmul entry should be findable");
        assert_eq!(tc.variant, "tensor_core");
    }
}
