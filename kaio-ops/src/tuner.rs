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

use kaio::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{attention, attention_causal, attention_flash, attention_flash_causal, matmul};

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
