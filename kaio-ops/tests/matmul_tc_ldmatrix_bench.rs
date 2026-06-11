#![allow(clippy::too_many_arguments)]

//! Benchmark + regression gate: `matmul_tc_ldmatrix` (the Sprint 9.3
//! warp-collective fragment-A path, built and parked per D6) vs
//! `matmul_tc` (the production `ld.shared` default).
//!
//! ## D6 record (Sprint 9.3, RTX 4090 sm_89)
//!
//! Three release runs measured the 4096³ interleaved median at
//! 102.75% / 104.49% / 102.82% — at the ±3% structural noise floor,
//! flat at smaller shapes, non-regression everywhere. Maintainer call:
//! the default stays on the proven `ld.shared` path; the ldmatrix
//! rewire ships **built-and-parked** (a deferred win, not a null
//! result — at the tile's 32-byte row stride the bank-conflict pattern
//! is unchanged by ldmatrix, so the XOR-swizzle follow-up is the lever
//! that makes the collective load pay; the default flip is one line).
//!
//! Note on absolutes: this bench runs in short bursts without a
//! sustained clock ramp, so its TF columns sit below the published
//! worst-of-10 table in `docs/performance.md`. The interleaved per-iter
//! ratios are thermal-invariant, so the verdict holds; the absolute
//! columns are not comparable to the published numbers.
//!
//! The two kernels are byte-identical except the A-fragment load
//! instructions (locked by `build_matmul_tc_module_loader_kinds_differ_
//! only_in_a_path` structurally and by the `ldmatrix_fragment_contract`
//! GPU gate at the fragment level), so per-iteration interleaved ratios
//! measure exactly the loader difference — the SC-2 methodology from
//! Sprint 9.1 (see `matmul_tc_bf16_bench.rs` for the full rationale:
//! independent cross-run timings are dominated by thermal drift;
//! same-iter alternating-order ratios cancel it).
//!
//! ## Gates (one-sided — uplift is never a failure)
//!
//! Applied **per shape** across 256³..4096³, per the Sprint 9.3 ship
//! gate "the sync path does not get slower at any measured shape":
//!
//! - **Median ratio ≥ 97%** — structural non-regression (the SC-2 ±3%
//!   median band, one-sided).
//! - **Worst ratio ≥ 85%** — catastrophic-tail canary (SC-2 ±15%,
//!   one-sided).
//!
//! ## Correctness pre-gate
//!
//! Before any timing, both kernels run on identical inputs and the
//! outputs are compared **bit-exactly** (same data through the same mma
//! sequence — only the load path differs, so any divergence is a
//! fragment-mapping bug, not noise). A failure here aborts the bench
//! with no timing numbers: a mapping bug must never produce "fast"
//! numbers for wrong output.
//!
//! The 4096³ median delta is also reported against the Sprint 9.3
//! stretch target (≥ 3 percentage points vs cuBLAS ratio — a different
//! denominator, reported for context, never asserted).
//!
//! Run with:
//! ```sh
//! cargo test -p kaio-ops --test matmul_tc_ldmatrix_bench -- --ignored --nocapture
//! ```

use std::time::Instant;

use half::f16;
use kaio::prelude::*;
use kaio_ops::{matmul_tc, matmul_tc_ldmatrix};

// --- Deterministic random data (LCG, matches matmul_tc_bench.rs pattern) ---

fn deterministic_data_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
        })
        .collect()
}

fn deterministic_data_f16(len: usize, seed: u64) -> Vec<f16> {
    deterministic_data_f32(len, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn tflops(m: usize, n: usize, k: usize, seconds: f64) -> f64 {
    2.0 * (m as f64) * (n as f64) * (k as f64) / seconds / 1e12
}

type MatmulFn = fn(
    &KaioDevice,
    &GpuBuffer<f16>,
    &GpuBuffer<f16>,
    &mut GpuBuffer<f32>,
    u32,
    u32,
    u32,
) -> kaio::prelude::Result<()>;

/// One bench "run": 5 warm-ups + 20 timed iterations, median seconds.
/// Same methodology as every other matmul bench in this directory.
fn bench_run(
    device: &KaioDevice,
    f: MatmulFn,
    a: &GpuBuffer<f16>,
    b: &GpuBuffer<f16>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> f64 {
    for _ in 0..5 {
        f(device, a, b, c, m, n, k).unwrap();
    }
    device.stream().synchronize().unwrap();
    let mut times = Vec::with_capacity(20);
    for _ in 0..20 {
        device.stream().synchronize().unwrap();
        let start = Instant::now();
        f(device, a, b, c, m, n, k).unwrap();
        device.stream().synchronize().unwrap();
        times.push(start.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[10]
}

/// Bit-exact output comparison of the two loader variants on identical
/// inputs. Panics with diagnostics on the first divergence.
fn assert_bit_equal_outputs(device: &KaioDevice, m: u32, n: u32, k: u32) {
    let a_h = deterministic_data_f16((m * k) as usize, 42);
    let b_h = deterministic_data_f16((k * n) as usize, 137);
    let a = device.alloc_from(&a_h).unwrap();
    let b = device.alloc_from(&b_h).unwrap();

    let mut c_new = device.alloc_zeros::<f32>((m * n) as usize).unwrap();
    let mut c_old = device.alloc_zeros::<f32>((m * n) as usize).unwrap();
    matmul_tc_ldmatrix(device, &a, &b, &mut c_new, m, n, k).unwrap();
    matmul_tc(device, &a, &b, &mut c_old, m, n, k).unwrap();

    let new_h = c_new.to_host(device).unwrap();
    let old_h = c_old.to_host(device).unwrap();
    let mut mismatches = 0usize;
    let mut first: Option<(usize, f32, f32)> = None;
    for (i, (nv, ov)) in new_h.iter().zip(old_h.iter()).enumerate() {
        if nv.to_bits() != ov.to_bits() {
            mismatches += 1;
            if first.is_none() {
                first = Some((i, *ov, *nv));
            }
        }
    }
    assert!(
        mismatches == 0,
        "ldmatrix A/B CORRECTNESS PRE-GATE FAILED at {m}x{n}x{k}:\n\
         {mismatches} of {} outputs differ bit-wise between the ldmatrix and\n\
         ld.shared loader builds on identical inputs (first at [{}]: \
         ld.shared {} vs ldmatrix {}).\n\
         \n\
         Identical data through the identical mma sequence cannot diverge —\n\
         this is a fragment-mapping bug in the ldmatrix loader, not noise.\n\
         NO TIMING NUMBERS were produced; fix the mapping first (see the\n\
         ldmatrix_fragment_contract gate in kaio/tests).",
        new_h.len(),
        first.map(|f| f.0).unwrap_or(0),
        first.map(|f| f.1).unwrap_or(0.0),
        first.map(|f| f.2).unwrap_or(0.0),
    );
}

const SIZES: [usize; 5] = [256, 512, 1024, 2048, 4096];
const RUNS_PER_SHAPE: usize = 10;
/// One-sided SC-2 median band: new/old median ratio below this fails.
const MEDIAN_FLOOR_PCT: f64 = 97.0;
/// One-sided catastrophic-tail canary.
const WORST_FLOOR_PCT: f64 = 85.0;
/// D6 noise floor: a 4096³ median uplift must exceed this to count as
/// "real" for the v0.5.0 default-on decision (the SC-2 ±3% band is
/// established structural noise).
const D6_REAL_UPLIFT_PCT: f64 = 103.0;

#[test]
#[ignore = "requires NVIDIA GPU (SM 8.0+); run with --ignored --nocapture"]
fn benchmark_matmul_tc_ldmatrix_ab() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    eprintln!();
    eprintln!(
        "GPU: {} (compute {}.{})",
        info.name, info.compute_capability.0, info.compute_capability.1
    );
    eprintln!(
        "matmul_tc fragment-A loader A/B: ldmatrix (new) vs ld.shared (old) — \
         {RUNS_PER_SHAPE} interleaved alternating-order runs per shape, \
         5 warm-ups + 20 timed iters per run, median per run"
    );

    // --- Correctness pre-gate: BEFORE any timing (Sprint 9.3 D5).
    // One aligned shape and one ragged-edge shape (edge-tile predication
    // interacts with the warp quadrant the loaders feed).
    eprintln!();
    eprintln!("Correctness pre-gate (bit-exact ldmatrix vs ld.shared)...");
    assert_bit_equal_outputs(&device, 256, 256, 256);
    assert_bit_equal_outputs(&device, 1023, 1023, 1024);
    eprintln!("Correctness pre-gate PASSED (256³ + 1023x1023x1024 bit-exact).");

    // --- Interleaved A/B timings per shape.
    struct ShapeResult {
        size: usize,
        median_ratio_pct: f64,
        worst_ratio_pct: f64,
        new_median_tf: f64,
        old_median_tf: f64,
    }
    let mut results: Vec<ShapeResult> = Vec::with_capacity(SIZES.len());

    for &size in &SIZES {
        let (m, n, k) = (size, size, size);
        let a_h = deterministic_data_f16(m * k, 42);
        let b_h = deterministic_data_f16(k * n, 137);
        let a = device.alloc_from(&a_h).unwrap();
        let b = device.alloc_from(&b_h).unwrap();
        let mut c_new = device.alloc_zeros::<f32>(m * n).unwrap();
        let mut c_old = device.alloc_zeros::<f32>(m * n).unwrap();

        // (new_seconds, old_seconds) per outer iter, alternating order.
        let mut samples: Vec<(f64, f64)> = Vec::with_capacity(RUNS_PER_SHAPE);
        for i in 0..RUNS_PER_SHAPE {
            let new_first = i.is_multiple_of(2);
            let (new_t, old_t) = if new_first {
                let nt = bench_run(
                    &device,
                    matmul_tc_ldmatrix,
                    &a,
                    &b,
                    &mut c_new,
                    m as u32,
                    n as u32,
                    k as u32,
                );
                let ot = bench_run(
                    &device, matmul_tc, &a, &b, &mut c_old, m as u32, n as u32, k as u32,
                );
                (nt, ot)
            } else {
                let ot = bench_run(
                    &device, matmul_tc, &a, &b, &mut c_old, m as u32, n as u32, k as u32,
                );
                let nt = bench_run(
                    &device,
                    matmul_tc_ldmatrix,
                    &a,
                    &b,
                    &mut c_new,
                    m as u32,
                    n as u32,
                    k as u32,
                );
                (nt, ot)
            };
            samples.push((new_t, old_t));
        }

        // Per-iter new/old TFLOPS ratios (>100% = ldmatrix faster).
        let ratios: Vec<f64> = samples
            .iter()
            .map(|&(new_t, old_t)| tflops(m, n, k, new_t) / tflops(m, n, k, old_t) * 100.0)
            .collect();

        let mut sorted = ratios.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ratio_pct = sorted[RUNS_PER_SHAPE / 2];
        // One-sided tail: the WORST ratio is the smallest (regression side).
        let worst_ratio_pct = sorted[0];

        let median_of = |f: fn(&(f64, f64)) -> f64, samples: &[(f64, f64)]| -> f64 {
            let mut v: Vec<f64> = samples.iter().map(f).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let new_median_tf = tflops(m, n, k, median_of(|s| s.0, &samples));
        let old_median_tf = tflops(m, n, k, median_of(|s| s.1, &samples));

        eprintln!();
        eprintln!("{size}³ per-iter ratios (ldmatrix/ld.shared):");
        for (i, (&(new_t, old_t), &r)) in samples.iter().zip(ratios.iter()).enumerate() {
            let order = if i % 2 == 0 { "new→old" } else { "old→new" };
            eprintln!(
                "  {:>4} {:>8} {:>9.3}ms {:>9.3}ms {:>9.2}%",
                i,
                order,
                new_t * 1000.0,
                old_t * 1000.0,
                r,
            );
        }
        eprintln!(
            "  {size}³: median {median_ratio_pct:.2}% | worst {worst_ratio_pct:.2}% | \
             ldmatrix {new_median_tf:.2} TF vs ld.shared {old_median_tf:.2} TF"
        );

        results.push(ShapeResult {
            size,
            median_ratio_pct,
            worst_ratio_pct,
            new_median_tf,
            old_median_tf,
        });
    }

    // --- Summary table.
    eprintln!();
    eprintln!("Summary (ldmatrix vs ld.shared, per-iter interleaved ratios):");
    eprintln!(
        "  {:>6} {:>12} {:>12} {:>12} {:>12}",
        "shape", "median", "worst", "new TF", "old TF"
    );
    for r in &results {
        eprintln!(
            "  {:>5}³ {:>11.2}% {:>11.2}% {:>12.2} {:>12.2}",
            r.size, r.median_ratio_pct, r.worst_ratio_pct, r.new_median_tf, r.old_median_tf
        );
    }

    // --- D6 verdict line (decision input, not a gate). The Sprint 9.3
    // call landed on default = ld.shared (see the header D6 record);
    // this line keeps reporting where the parked path stands so the
    // XOR-swizzle follow-up can re-read the verdict without re-deriving
    // the bands: > +5% real, +3–5% provably-real-but-marginal, inside
    // ±3% indistinguishable from the structural noise floor.
    let r4096 = results.last().unwrap();
    let verdict = if r4096.median_ratio_pct > 105.0 {
        "REAL UPLIFT (revisit the default flip — see matmul_tc docs)"
    } else if r4096.median_ratio_pct > D6_REAL_UPLIFT_PCT {
        "ABOVE NOISE FLOOR, MARGINAL (parked default stands; swizzle is the lever)"
    } else if r4096.median_ratio_pct >= MEDIAN_FLOOR_PCT {
        "WITHIN NOISE (parked default stands — matches the Sprint 9.3 D6 record)"
    } else {
        "REGRESSION (gate below will fail)"
    };
    eprintln!();
    eprintln!(
        "D6 status at 4096³: ldmatrix/ld.shared median {:.2}% → {verdict}",
        r4096.median_ratio_pct
    );

    // Debug builds: per-iter variance makes the bounds meaningless —
    // same policy as the SC-2 gate (see matmul_tc_bf16_bench.rs).
    if cfg!(debug_assertions) {
        eprintln!(
            "DEBUG BUILD detected: non-regression assertions skipped. For the\n\
             canonical verdict run:\n\
                 cargo test --release -p kaio-ops --test matmul_tc_ldmatrix_bench -- --ignored --nocapture"
        );
        return;
    }

    // --- Non-regression gates, per shape, one-sided.
    for r in &results {
        assert!(
            r.median_ratio_pct >= MEDIAN_FLOOR_PCT,
            "Sprint 9.3 LDMATRIX NON-REGRESSION GATE FAILED (median) at {}³:\n\
             the parked ldmatrix path is {:.2}% of the ld.shared default at the\n\
             median per-iter ratio — below the one-sided structural floor of\n\
             {MEDIAN_FLOOR_PCT:.0}%. At Sprint 9.3 close the two paths were\n\
             within noise of each other at every shape, so a real gap here means\n\
             the parked path has rotted (an IR/emit change regressed the ldmatrix\n\
             build). Investigate before the swizzle follow-up builds on it.",
            r.size,
            r.median_ratio_pct,
        );
        assert!(
            r.worst_ratio_pct >= WORST_FLOOR_PCT,
            "Sprint 9.3 LDMATRIX TAIL GATE FAILED at {}³: worst per-iter ratio \
             {:.2}% fell below the one-sided tail floor of {WORST_FLOOR_PCT:.0}%. \
             A single outlier against tightly-clustered peers is OS noise (re-run); \
             sustained skew is a kernel issue.",
            r.size,
            r.worst_ratio_pct,
        );
    }

    eprintln!(
        "LDMATRIX A/B GATES PASSED: median ≥ {MEDIAN_FLOOR_PCT:.0}% and worst ≥ \
         {WORST_FLOOR_PCT:.0}% at every measured shape."
    );
}
