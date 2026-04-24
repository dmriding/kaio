//! KAIO repo-internal tooling.
//!
//! Single-command access to the showcase examples and the matmul
//! tensor-core benchmark, without `cd`-ing into subdirectories. Invoked
//! via the `[alias] xtask = ...` in `.cargo/config.toml`:
//!
//! ```sh
//! cargo xtask showcase          # runs all showcase examples in sequence
//! cargo xtask showcase silu     # runs just fused_silu_gate
//! cargo xtask showcase --list   # prints the available showcases
//! cargo xtask bench             # runs all registered benchmarks
//! cargo xtask all               # showcase then bench
//! cargo xtask --help            # this message
//! ```
//!
//! Each subcommand shells out via `std::process::Command` with inherited
//! stdio so the called cargo invocations stream output live to the user's
//! terminal. Continue-on-error: one example failing does not abort the
//! rest; the summary at the end is non-zero if any step failed.
//!
//! All examples are standalone Cargo projects under `examples/<name>/` with
//! their own `Cargo.toml` (see `examples/README.md`). xtask preserves that
//! isolation — it never links against the examples, only invokes `cargo
//! run --release --manifest-path examples/<name>/Cargo.toml`.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Instant;

/// Ordered list of showcase examples. `(short_name, dir_name, description)`.
/// Short name is what users pass to `cargo xtask showcase <name>`.
const SHOWCASES: &[(&str, &str, &str)] = &[
    (
        "silu",
        "fused_silu_gate",
        "Fused SiLU gate (LLaMA feedforward primitive)",
    ),
    (
        "gelu",
        "gelu_comparison",
        "Exact vs fast GELU (BERT/GPT activations)",
    ),
    (
        "rms",
        "rms_norm",
        "Single-block RMSNorm (LLaMA normalization)",
    ),
    (
        "layernorm",
        "layer_norm",
        "Single-block LayerNorm (classic transformer normalization)",
    ),
    (
        "softmax",
        "softmax",
        "Single-block softmax (attention normalization with max-sub)",
    ),
    (
        "int8",
        "int8_dequant",
        "Symmetric INT8 dequantization (quantized-weight unpack)",
    ),
    (
        "int8matmul",
        "int8_matmul",
        "Symmetric INT8 dequantize-matmul (tensor-core W8A8)",
    ),
    (
        "int4matmul",
        "int4_matmul",
        "Symmetric INT4 GPTQ-style dequantize-matmul (tensor-core W4A16)",
    ),
    (
        "qkvattn",
        "quantized_attention",
        "End-to-end quantized attention (qkv_project_int4 → attention_tc)",
    ),
];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(|s| s.as_str()).unwrap_or("--help");
    let rest: Vec<&str> = args.iter().skip(1).map(String::as_str).collect();

    match command {
        "showcase" => run_showcase(&rest),
        "bench" => run_bench(&rest),
        "all" => run_all(&rest),
        "--help" | "-h" | "help" => {
            print_help();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("xtask: unknown command: {other}");
            eprintln!();
            print_help();
            ExitCode::from(2)
        }
    }
}

fn print_help() {
    println!(
        r#"KAIO repo tooling.

Usage:
  cargo xtask showcase [<name>|--list]   Run one or all showcase examples.
  cargo xtask bench [<name>|--list]      Run one or all benchmarks. Use --list to enumerate.
  cargo xtask all                        Run showcase + bench in sequence.
  cargo xtask --help                     Show this message.

Showcase names:
  silu       Fused SiLU gate (LLaMA feedforward primitive)
  gelu       Exact vs fast GELU (BERT/GPT activations)
  rms        Single-block RMSNorm (LLaMA normalization)
  layernorm  Single-block LayerNorm (classic transformer normalization)
  softmax    Single-block softmax (attention normalization with max-sub)
  int8       Symmetric INT8 dequantization (quantized-weight unpack)
  int8matmul Symmetric INT8 dequantize-matmul (tensor-core W8A8)
  int4matmul Symmetric INT4 GPTQ-style dequantize-matmul (tensor-core W4A16)
  qkvattn    End-to-end quantized attention (qkv_project_int4 → attention_tc)

Prerequisites:
  - Rust 1.94+ (pinned via rust-toolchain.toml)
  - NVIDIA GPU with driver installed (no CUDA toolkit needed for runtime)

Examples run in --release mode. First run compiles each example from
scratch (~15-30s per example); subsequent runs use cached builds.
"#
    );
}

/// `cargo xtask showcase [<name>|--list]` entry point.
fn run_showcase(args: &[&str]) -> ExitCode {
    let repo_root = match find_repo_root() {
        Ok(path) => path,
        Err(e) => {
            eprintln!("xtask: could not locate repo root: {e}");
            return ExitCode::from(1);
        }
    };

    // --list: just print available showcases and exit
    if args.contains(&"--list") {
        println!("Available showcases:");
        for (short, dir, desc) in SHOWCASES {
            println!("  {short:<10} examples/{dir:<20} {desc}");
        }
        return ExitCode::SUCCESS;
    }

    // Resolve which showcases to run.
    let targets: Vec<&(&str, &str, &str)> = if let Some(filter) = args.first() {
        match SHOWCASES
            .iter()
            .find(|(short, dir, _)| *short == *filter || *dir == *filter)
        {
            Some(matched) => vec![matched],
            None => {
                eprintln!(
                    "xtask: unknown showcase `{filter}`. Use `cargo xtask showcase --list` \
                     to see available names."
                );
                return ExitCode::from(2);
            }
        }
    } else {
        SHOWCASES.iter().collect()
    };

    let total = targets.len();
    let mut failures: Vec<&str> = Vec::new();
    let overall_start = Instant::now();

    for (idx, (_short, dir, desc)) in targets.iter().enumerate() {
        println!();
        println!("=== [{}/{total}] {dir} — {desc} ===", idx + 1);
        println!();

        let manifest = repo_root.join("examples").join(dir).join("Cargo.toml");
        let t0 = Instant::now();
        let status = Command::new("cargo")
            .arg("run")
            .arg("--release")
            .arg("--manifest-path")
            .arg(&manifest)
            .status();

        let elapsed = t0.elapsed();
        match status {
            Ok(s) if s.success() => {
                println!();
                println!("    [ok] {dir} completed in {:.1}s", elapsed.as_secs_f32());
            }
            Ok(s) => {
                eprintln!();
                eprintln!(
                    "    [fail] {dir} exited with code {:?} after {:.1}s",
                    s.code(),
                    elapsed.as_secs_f32()
                );
                failures.push(dir);
            }
            Err(e) => {
                eprintln!();
                eprintln!("    [fail] could not launch {dir}: {e}");
                failures.push(dir);
            }
        }
    }

    println!();
    println!("---");
    let overall = overall_start.elapsed();
    if failures.is_empty() {
        println!(
            "All {total} showcase(s) passed in {:.1}s.",
            overall.as_secs_f32()
        );
        ExitCode::SUCCESS
    } else {
        println!(
            "{}/{total} showcase(s) passed in {:.1}s; {} failed: {}",
            total - failures.len(),
            overall.as_secs_f32(),
            failures.len(),
            failures.join(", ")
        );
        ExitCode::from(1)
    }
}

/// Ordered list of benchmark test harnesses wired to `cargo xtask bench`.
/// `(header, test_name, description)`. The short `--list` form mirrors
/// `cargo xtask showcase --list`; passing a name filters to that bench.
const BENCHES: &[(&str, &str, &str)] = &[
    (
        "=== Matmul tensor-core benchmark (KAIO f16 × f16 → f32 vs cuBLAS sgemm) ===",
        "matmul_tc_bench",
        "f16 × f16 → f32 TC sync + async vs cuBLAS sgemm",
    ),
    (
        "=== matmul_int8 benchmark (KAIO i8 × i8 → f32 vs cuBLAS sgemm, rough reference) ===",
        "matmul_int8_bench",
        "W8A8 symmetric INT8 matmul; cuBLAS sgemm column is apples-to-oranges",
    ),
    (
        "=== matmul_int4 benchmark (KAIO s4 × f16 → f32 vs cuBLAS sgemm, rough reference) ===",
        "matmul_int4_bench",
        "W4A16 GPTQ-style INT4 matmul; cuBLAS sgemm column is apples-to-oranges",
    ),
    (
        "=== QKV projection benchmark (INT4 fused vs 3x matmul_int4; INT8 absolute TOPS) ===",
        "qkv_project_bench",
        "Fused tri-output QKV projection; INT4 ratio vs 3x matmul_int4, INT8 absolute TOPS",
    ),
    (
        "=== attention_tc benchmark (single-head self-attention, f16 Q/K/V -> f32 out) ===",
        "attention_tc_bench",
        "Tensor-core attention (plain + causal) latency + seq/s + attn_scores/s",
    ),
];

/// `cargo xtask bench [<name>|--list]` entry point. Runs one or all
/// benchmark harnesses with `--ignored --nocapture`. Continue-on-error
/// so one failing bench doesn't block the others from running.
fn run_bench(args: &[&str]) -> ExitCode {
    let repo_root = match find_repo_root() {
        Ok(path) => path,
        Err(e) => {
            eprintln!("xtask: could not locate repo root: {e}");
            return ExitCode::from(1);
        }
    };

    if args.contains(&"--list") {
        println!("Available benchmarks:");
        for (_, test_name, desc) in BENCHES {
            println!("  {test_name:<22} {desc}");
        }
        return ExitCode::SUCCESS;
    }

    let targets: Vec<&(&str, &str, &str)> = if let Some(filter) = args.first() {
        match BENCHES
            .iter()
            .find(|(_, test_name, _)| *test_name == *filter)
        {
            Some(matched) => vec![matched],
            None => {
                eprintln!(
                    "xtask: unknown benchmark `{filter}`. Use `cargo xtask bench --list` \
                     to see available names."
                );
                return ExitCode::from(2);
            }
        }
    } else {
        BENCHES.iter().collect()
    };

    let mut failures: Vec<&str> = Vec::new();
    let t_total = Instant::now();

    for (header, test_name, _desc) in &targets {
        println!();
        println!("{header}");
        println!();
        println!("  Hardware: uses the first CUDA device visible to the driver.");
        println!(
            "  Methodology: 5 warmup + 20 timed iterations per shape (bench-specific shape table printed below)."
        );
        println!();

        let t0 = Instant::now();
        let status = Command::new("cargo")
            .current_dir(&repo_root)
            .args([
                "test",
                "-p",
                "kaio-ops",
                "--release",
                "--test",
                test_name,
                "--",
                "--ignored",
                "--nocapture",
            ])
            .status();

        let elapsed = t0.elapsed();
        match status {
            Ok(s) if s.success() => {
                println!();
                println!("---");
                println!("{test_name} completed in {:.1}s.", elapsed.as_secs_f32());
            }
            Ok(s) => {
                eprintln!();
                eprintln!(
                    "[fail] {test_name} exited with code {:?} after {:.1}s",
                    s.code(),
                    elapsed.as_secs_f32()
                );
                failures.push(test_name);
            }
            Err(e) => {
                eprintln!();
                eprintln!("[fail] could not launch {test_name}: {e}");
                failures.push(test_name);
            }
        }
    }

    println!();
    println!("===");
    let total_elapsed = t_total.elapsed().as_secs_f32();
    if failures.is_empty() {
        println!(
            "All {} benchmark(s) completed in {total_elapsed:.1}s.",
            targets.len()
        );
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "{}/{} benchmark(s) failed after {total_elapsed:.1}s: {}",
            failures.len(),
            targets.len(),
            failures.join(", ")
        );
        ExitCode::from(1)
    }
}

/// `cargo xtask all` — run showcase then bench. Continue-on-error; the
/// final exit code is non-zero if either step had failures.
fn run_all(_args: &[&str]) -> ExitCode {
    let showcase_code = run_showcase(&[]);
    println!();
    let bench_code = run_bench(&[]);
    println!();
    println!("===");
    println!(
        "cargo xtask all — showcase: {}, bench: {}",
        exit_label(&showcase_code),
        exit_label(&bench_code)
    );
    if showcase_code == ExitCode::SUCCESS && bench_code == ExitCode::SUCCESS {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn exit_label(code: &ExitCode) -> &'static str {
    // ExitCode doesn't expose its inner u8 directly; compare to SUCCESS by
    // round-tripping. This is the idiomatic way on stable.
    if format!("{code:?}") == format!("{:?}", ExitCode::SUCCESS) {
        "ok"
    } else {
        "fail"
    }
}

/// Walk up from the current dir to find the workspace root (Cargo.toml
/// with `[workspace]`). xtask is invoked via `cargo xtask` which sets the
/// working directory to the invoking workspace root, but being robust lets
/// users run the compiled binary directly from anywhere.
fn find_repo_root() -> Result<PathBuf, String> {
    let start = std::env::current_dir().map_err(|e| e.to_string())?;
    let mut cur: &Path = &start;
    loop {
        let candidate = cur.join("Cargo.toml");
        if candidate.exists() {
            let content = std::fs::read_to_string(&candidate).map_err(|e| e.to_string())?;
            if content.contains("[workspace]") {
                return Ok(cur.to_path_buf());
            }
        }
        match cur.parent() {
            Some(parent) => cur = parent,
            None => {
                return Err(format!(
                    "no workspace Cargo.toml found walking up from {}",
                    start.display()
                ));
            }
        }
    }
}
