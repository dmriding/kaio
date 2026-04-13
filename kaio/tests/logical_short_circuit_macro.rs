//! Sprint 7.0 D4: GPU round-trip tests for short-circuit `&&` / `||`.
//!
//! Covers the two lowering paths separately (branch-direct inside `if`
//! conditions; materialized predicate in expression position) plus the
//! nested / interleaved cases that stress label allocation.
//!
//! AD4 signature canary: `logical_and_bounds_guard` launches with
//! `i == n` on the last thread so `arr[i]` would be OOB if the `&&` ever
//! stopped short-circuiting. That test is the signature correctness gate
//! for Phase 7's DSL completeness story.

use kaio::prelude::*;

// ---------------------------------------------------------------------------
// If-condition (branch-direct) path
// ---------------------------------------------------------------------------

#[gpu_kernel(block_size = 128)]
fn bounds_guard_and(arr: &[u32], out: &mut [u32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    // Short-circuit MUST skip `arr[idx]` when idx >= n. Without it, the
    // last threads would read past the end of `arr` — undefined behavior,
    // often produces large nonsense values on Ampere+.
    if idx < n && arr[idx] > 0 {
        out[idx] = 1;
    }
}

#[gpu_kernel(block_size = 128)]
fn early_success_or(flag: u32, arr: &[u32], out: &mut [u32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        // `flag != 0 || arr[idx] > 100` — when `flag != 0`, RHS is short-circuited.
        // If LHS is false, RHS is evaluated normally.
        if flag > 0 || arr[idx] > 100 {
            out[idx] = 1;
        } else {
            out[idx] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Expression-position (materialized) path
// ---------------------------------------------------------------------------

#[gpu_kernel(block_size = 128)]
fn materialized_and(a: &[u32], b: &[u32], out: &mut [u32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        // `mask` materializes the short-circuit result into a .pred register.
        // The downstream `if mask { ... }` then consumes the predicate.
        let mask = a[idx] > 0 && b[idx] > 0;
        if mask {
            out[idx] = 1;
        } else {
            out[idx] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Nested / interleaved cases
// ---------------------------------------------------------------------------

#[gpu_kernel(block_size = 128)]
fn nested_and_or(a: &[u32], b: &[u32], c: &[u32], out: &mut [u32], n: u32) {
    // `if (a && b) || c { ... }` — inner short-circuit inside outer.
    // Label allocation must not collide (each logical op gets unique labels).
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        if (a[idx] > 0 && b[idx] > 0) || c[idx] > 0 {
            out[idx] = 1;
        } else {
            out[idx] = 0;
        }
    }
}

#[gpu_kernel(block_size = 128)]
fn complex_mask(a: &[u32], b: &[u32], c: &[u32], d: &[u32], out: &mut [u32], n: u32) {
    // Expression-position nested: `let m = (a && b) || (c && d);`
    // Four-way label interleaving across a materialized predicate.
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let m = (a[idx] > 0 && b[idx] > 0) || (c[idx] > 0 && d[idx] > 0);
        if m {
            out[idx] = 1;
        } else {
            out[idx] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore] // requires NVIDIA GPU
fn logical_and_bounds_guard_no_oob() {
    // AD4 signature canary. n=100 threads launched, block_size=128 means last
    // 28 threads have idx >= n. For those threads, `arr[idx]` would be OOB
    // read if `&&` doesn't short-circuit. We compare against CPU reference
    // that emulates the short-circuit: out[idx] is 1 only if idx<n AND arr[idx]>0.
    let device = KaioDevice::new(0).expect("GPU required");
    let n: u32 = 100;

    // Build arr[100] with known values. Allocate only n elements to make OOB
    // reads maximally visible — in practice GPU reads past the end will
    // either return garbage or produce compute-sanitizer errors.
    let arr_host: Vec<u32> = (0..n).map(|i| if i % 2 == 0 { 5 } else { 0 }).collect();
    let out_size = 128; // full block
    let arr = device.alloc_from(&arr_host).expect("alloc arr");
    let mut out = device.alloc_zeros::<u32>(out_size).expect("alloc out");

    bounds_guard_and::launch(&device, &arr, &mut out, n).expect("launch");

    let result = out.to_host(&device).expect("to_host");
    // Thread i: out[i] == 1 iff (i < n && arr[i] > 0). For i >= n, out[i] must be 0.
    let expected: Vec<u32> = (0..out_size as u32)
        .map(|i| {
            if i < n && arr_host[i as usize] > 0 {
                1
            } else {
                0
            }
        })
        .collect();
    assert_eq!(
        result, expected,
        "bounds_guard_and produced wrong results — suggests && did not short-circuit"
    );
}

#[test]
#[ignore] // requires NVIDIA GPU
fn logical_or_early_success_flag_on() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n: u32 = 64;

    // flag=1 → LHS true → RHS (arr[idx] > 100) is short-circuited. All out[i] == 1.
    let arr_host: Vec<u32> = vec![0; n as usize];
    let arr = device.alloc_from(&arr_host).expect("alloc arr");
    let mut out = device.alloc_zeros::<u32>(n as usize).expect("alloc out");
    early_success_or::launch(&device, 1, &arr, &mut out, n).expect("launch");

    let result = out.to_host(&device).expect("to_host");
    assert_eq!(result, vec![1u32; n as usize]);
}

#[test]
#[ignore] // requires NVIDIA GPU
fn logical_or_early_success_flag_off() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n: u32 = 64;

    // flag=0 → LHS false → RHS evaluated. out[i] == (arr[i] > 100).
    let arr_host: Vec<u32> = (0..n).map(|i| i * 10).collect(); // 0, 10, 20, ..., 630
    let arr = device.alloc_from(&arr_host).expect("alloc arr");
    let mut out = device.alloc_zeros::<u32>(n as usize).expect("alloc out");
    early_success_or::launch(&device, 0, &arr, &mut out, n).expect("launch");

    let result = out.to_host(&device).expect("to_host");
    let expected: Vec<u32> = arr_host
        .iter()
        .map(|&v| if v > 100 { 1 } else { 0 })
        .collect();
    assert_eq!(result, expected);
}

#[test]
#[ignore] // requires NVIDIA GPU
fn logical_and_expression_position_materialized() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n: u32 = 32;
    // Cover all 4 (LHS, RHS) permutations across threads.
    let a_host: Vec<u32> = (0..n).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
    let b_host: Vec<u32> = (0..n).map(|i| if i < 16 { 1 } else { 0 }).collect();
    let expected: Vec<u32> = a_host
        .iter()
        .zip(&b_host)
        .map(|(&a, &b)| if a > 0 && b > 0 { 1 } else { 0 })
        .collect();

    let a = device.alloc_from(&a_host).expect("alloc a");
    let b = device.alloc_from(&b_host).expect("alloc b");
    let mut out = device.alloc_zeros::<u32>(n as usize).expect("alloc out");
    materialized_and::launch(&device, &a, &b, &mut out, n).expect("launch");

    let result = out.to_host(&device).expect("to_host");
    assert_eq!(result, expected);
}

#[test]
#[ignore] // requires NVIDIA GPU
fn logical_nested_and_or_all_combos() {
    // Exercises `(a && b) || c` across thread configurations that cover:
    // - inner && true,  outer || fires through LHS
    // - inner && false, outer || saves via c = true
    // - inner && false, outer || false → all zero
    let device = KaioDevice::new(0).expect("GPU required");
    let n: u32 = 8;

    let a_host: Vec<u32> = vec![1, 1, 0, 0, 1, 0, 1, 0];
    let b_host: Vec<u32> = vec![1, 0, 1, 0, 1, 1, 0, 0];
    let c_host: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 1, 0];
    // (a && b) || c:
    // idx 0: (1&&1)||0 = 1
    // idx 1: (1&&0)||0 = 0
    // idx 2: (0&&1)||0 = 0
    // idx 3: (0&&0)||0 = 0
    // idx 4: (1&&1)||1 = 1
    // idx 5: (0&&1)||1 = 1
    // idx 6: (1&&0)||1 = 1
    // idx 7: (0&&0)||0 = 0
    let expected: Vec<u32> = vec![1, 0, 0, 0, 1, 1, 1, 0];

    let a = device.alloc_from(&a_host).expect("alloc a");
    let b = device.alloc_from(&b_host).expect("alloc b");
    let c = device.alloc_from(&c_host).expect("alloc c");
    let mut out = device.alloc_zeros::<u32>(n as usize).expect("alloc out");
    nested_and_or::launch(&device, &a, &b, &c, &mut out, n).expect("launch");

    let result = out.to_host(&device).expect("to_host");
    assert_eq!(
        result, expected,
        "nested (a && b) || c failed — likely a label collision or \
         bad short-circuit nesting in D4 lowering"
    );
}

#[test]
#[ignore] // requires NVIDIA GPU
fn logical_complex_mask_expression_nested() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n: u32 = 16;

    // `let m = (a && b) || (c && d);` — four-way label interleaving in
    // expression position. Three permutations hitting distinct paths.
    let a_host: Vec<u32> = vec![1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0];
    let b_host: Vec<u32> = vec![1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1];
    let c_host: Vec<u32> = vec![0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1];
    let d_host: Vec<u32> = vec![0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0];
    let expected: Vec<u32> = (0..n as usize)
        .map(|i| {
            let a_true = a_host[i] > 0;
            let b_true = b_host[i] > 0;
            let c_true = c_host[i] > 0;
            let d_true = d_host[i] > 0;
            if (a_true && b_true) || (c_true && d_true) {
                1
            } else {
                0
            }
        })
        .collect();

    let a = device.alloc_from(&a_host).expect("alloc a");
    let b = device.alloc_from(&b_host).expect("alloc b");
    let c = device.alloc_from(&c_host).expect("alloc c");
    let d = device.alloc_from(&d_host).expect("alloc d");
    let mut out = device.alloc_zeros::<u32>(n as usize).expect("alloc out");
    complex_mask::launch(&device, &a, &b, &c, &d, &mut out, n).expect("launch");

    let result = out.to_host(&device).expect("to_host");
    assert_eq!(
        result, expected,
        "complex_mask (a && b) || (c && d) in expression position failed — \
         likely label interleaving / fresh-label collision in D4 materialized path"
    );
}
