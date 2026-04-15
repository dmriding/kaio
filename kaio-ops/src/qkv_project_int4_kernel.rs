//! Fused tri-output INT4 QKV projection — W4A16 (f16 activations,
//! packed signed-INT4 weights, f16 group scales).
//!
//! Sprint 7.3 contingent deliverable. Tri-output extension of
//! [`matmul_int4`][crate::matmul_int4] — same packing convention
//! (8 signed `s4` per `u32`, K-contiguous), same group-scale semantics
//! (`group_size = 128` fixed, f16 scales `[num_groups, N]` row-major),
//! same DEQUANT-F16 pipeline (`shr.s32` sign-extend → `cvt.rn.f32.s32`
//! → `cvt.rn.f16.f32` → scale-fold `mul.f16` → `MovPack` → fragment B).
//!
//! One kernel launch produces three f16 outputs (Q, K, V) from three
//! packed weight tensors and three separate scale tensors, sharing a
//! single load of X. `attention_tc`-ready outputs.
//!
//! # Contingent ship
//!
//! Per the Sprint 7.3 plan, INT4 is a **contingent second deliverable**
//! shipped after the `qkv_project_int8` MVS ship point at D4 if the
//! D2.5 register budget and D5/D6/D7 correctness gates stay clean.
//! Sprint may also ship INT8-only via rollback #5 if INT4 exposes
//! unexpected complexity in the tri-output + group-scale-reload
//! context.
//!
//! # Unified mma path with INT8
//!
//! Both `qkv_project_int4` and [`qkv_project_int8`][super::qkv_project_int8_kernel]
//! target the same `mma.sync.m16n8k16.f16.f16.f32` shape with
//! `K_TILE_SHARED = 16`. The shared store-out helper
//! ([`emit_store_fragment_c_f32_to_f16_packed`][super::store_out::emit_store_fragment_c_f32_to_f16_packed]
//! once wired) casts the three f32 fragment-C banks to f16 on store-out;
//! INT4 passes `scale = None` because the group scale is folded into
//! fragment B during dequant, not applied post-accumulation.
//!
//! # Register budget and tile shape
//!
//! Same envelope as INT8 (see `qkv_project_int8_kernel`): 4-warp block
//! with 64×32 output tile, 3 f32 fragment-C banks persistent across
//! the K-loop (48 regs/lane for accumulators). D2.5 checkpoint runs
//! the minimal skeleton through `ptxas -v` before the full kernel body
//! is authored.

use kaio::prelude::*;

// --- mma.sync.m16n8k16 instance shape (f16 inputs, f32 accumulator) ---
// Matches qkv_project_int8 and matmul_int4.
#[allow(dead_code)] // wired up in D5
pub(crate) const BM: u32 = 16; // mma m dim
#[allow(dead_code)] // wired up in D5
pub(crate) const BN: u32 = 8; // mma n dim

// --- Multi-warp block tiling (halved N vs matmul_int4 for tri-output accumulators) ---
#[allow(dead_code)] // wired up in D5
pub(crate) const BM_BLOCK: u32 = 64;
#[allow(dead_code)] // wired up in D5
pub(crate) const BN_BLOCK: u32 = 32;
#[allow(dead_code)] // wired up in D5
pub(crate) const WARP_QUAD_M: u32 = 32;
#[allow(dead_code)] // wired up in D5
pub(crate) const WARP_QUAD_N: u32 = 16;
#[allow(dead_code)] // wired up in D5
pub(crate) const MMAS_PER_WARP_M: u32 = WARP_QUAD_M / BM; // 2
#[allow(dead_code)] // wired up in D5
pub(crate) const MMAS_PER_WARP_N: u32 = WARP_QUAD_N / BN; // 2
#[allow(dead_code)] // wired up in D5
pub(crate) const WARPS_PER_BLOCK: u32 = 4;
#[allow(dead_code)] // wired up in D5
pub(crate) const THREADS_PER_BLOCK: u32 = WARPS_PER_BLOCK * 32; // 128

// --- INT4 packing + group scales (mirrored from matmul_int4) ---
#[allow(dead_code)] // wired up in D5
pub(crate) const NIBBLES_PER_U32: u32 = 8;
#[allow(dead_code)] // wired up in D5
pub(crate) const GROUP_SIZE: u32 = 128;

// --- K-tile granularity (unified across both QKV variants) ---
#[allow(dead_code)] // wired up in D5
pub(crate) const K_TILE_SHARED: u32 = 16;

// --- Group-scale reload cadence ---
// group_idx = k_tile / K_TILE_GROUP_RATIO; reload every time k_tile crosses a group boundary.
#[allow(dead_code)] // wired up in D5
pub(crate) const K_TILE_GROUP_RATIO: u32 = GROUP_SIZE / K_TILE_SHARED; // 8 K-tiles per group

/// Validate shape + alignment preconditions for `qkv_project_int4`.
///
/// Sprint 7.3 D1. Enforces the **W4A16 MHA** contract:
///
/// - `M`, `N`, `K` all non-zero.
/// - `group_size == GROUP_SIZE` (=128) — non-128 group sizes are a
///   follow-up extension inherited from `matmul_int4`.
/// - `K % GROUP_SIZE == 0` (implies `K % K_TILE_SHARED == 0`).
/// - `N % 2 == 0` — store-out packs adjacent f16 output pairs into one `.b32`.
/// - `N_q == N_k == N_v == N` — v1 is strict MHA. Grouped-query attention
///   (GQA) is a follow-up op; users with GQA weights should call three
///   separate `matmul_int4`s.
/// - Buffer-size sanity:
///   - `x >= M * K`
///   - each of `w_q_packed`, `w_k_packed`, `w_v_packed` >= `N * (K / 8)` u32
///   - each of `scales_q`, `scales_k`, `scales_v` >= `(K / group_size) * N` f16
///   - each of `q_out`, `k_out`, `v_out` >= `M * N` f16
///
/// `M` and `N` may be any positive value — edge-tile predication in the
/// kernel handles ragged output (same posture as `matmul_int4`).
///
/// # Pointer distinctness
///
/// `q_out`, `k_out`, `v_out` must be **three distinct allocations**.
/// Same pathological-caller guard as `qkv_project_int8` — Rust's `&mut`
/// borrow rules prevent variable-level aliasing and KAIO does not
/// expose buffer-splitting APIs that could produce overlapping views.
#[allow(dead_code)] // wired up in D6
pub(crate) fn validate_dims_qkv_int4(
    x: &GpuBuffer<half::f16>,
    w_q_packed: &GpuBuffer<u32>,
    w_k_packed: &GpuBuffer<u32>,
    w_v_packed: &GpuBuffer<u32>,
    scales_q: &GpuBuffer<half::f16>,
    scales_k: &GpuBuffer<half::f16>,
    scales_v: &GpuBuffer<half::f16>,
    q_out: &GpuBuffer<half::f16>,
    k_out: &GpuBuffer<half::f16>,
    v_out: &GpuBuffer<half::f16>,
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(KaioError::InvalidConfig(
            "qkv_project_int4: M, N, K dimensions must be non-zero".to_string(),
        ));
    }
    if group_size != GROUP_SIZE {
        return Err(KaioError::InvalidConfig(format!(
            "qkv_project_int4: group_size must be {GROUP_SIZE} (got {group_size}). \
             Non-128 group sizes are deferred to a follow-up sprint."
        )));
    }
    if !k.is_multiple_of(GROUP_SIZE) {
        return Err(KaioError::InvalidConfig(format!(
            "qkv_project_int4: K must be a multiple of group_size={GROUP_SIZE} (got {k}). \
             Partial groups are not supported."
        )));
    }
    if !n.is_multiple_of(2) {
        return Err(KaioError::InvalidConfig(format!(
            "qkv_project_int4: N must be even (got {n}). The store-out path packs \
             adjacent f16 output pairs into one .b32; odd N would leave a ragged \
             last column that the current store epilogue does not handle."
        )));
    }

    let mk = (m as usize) * (k as usize);
    let packed_words = ((k as usize) / (NIBBLES_PER_U32 as usize)) * (n as usize);
    let num_groups = (k as usize) / (group_size as usize);
    let scales_cells = num_groups * (n as usize);
    let mn = (m as usize) * (n as usize);

    if x.len() < mk {
        return Err(KaioError::InvalidConfig(format!(
            "qkv_project_int4: X buffer too small: need {mk} f16 ({m}×{k}), got {}",
            x.len()
        )));
    }
    for (label, buf) in [
        ("W_Q_packed", w_q_packed),
        ("W_K_packed", w_k_packed),
        ("W_V_packed", w_v_packed),
    ] {
        if buf.len() < packed_words {
            return Err(KaioError::InvalidConfig(format!(
                "qkv_project_int4: {label} buffer too small: need {packed_words} u32 \
                 ({n} cols × {} K-words), got {}",
                (k as usize) / (NIBBLES_PER_U32 as usize),
                buf.len()
            )));
        }
    }
    for (label, buf) in [
        ("scales_q", scales_q),
        ("scales_k", scales_k),
        ("scales_v", scales_v),
    ] {
        if buf.len() < scales_cells {
            return Err(KaioError::InvalidConfig(format!(
                "qkv_project_int4: {label} buffer too small: need {scales_cells} f16 \
                 ({num_groups} groups × {n} cols), got {}",
                buf.len()
            )));
        }
    }
    for (label, buf) in [("Q_out", q_out), ("K_out", k_out), ("V_out", v_out)] {
        if buf.len() < mn {
            return Err(KaioError::InvalidConfig(format!(
                "qkv_project_int4: {label} buffer too small: need {mn} f16 ({m}×{n}), got {}",
                buf.len()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_device() -> Result<KaioDevice> {
        KaioDevice::new(0)
    }

    #[test]
    fn validate_accepts_canonical_shape() {
        let Ok(device) = make_device() else {
            return;
        };
        let m = 64u32;
        let n = 64u32;
        let k = 128u32; // exactly one group
        let x = device.alloc_zeros::<half::f16>((m * k) as usize).unwrap();
        let w = device
            .alloc_zeros::<u32>(((k / NIBBLES_PER_U32) * n) as usize)
            .unwrap();
        let num_groups = k / GROUP_SIZE;
        let s = device
            .alloc_zeros::<half::f16>((num_groups * n) as usize)
            .unwrap();
        let o = device.alloc_zeros::<half::f16>((m * n) as usize).unwrap();
        assert!(
            validate_dims_qkv_int4(&x, &w, &w, &w, &s, &s, &s, &o, &o, &o, m, n, k, GROUP_SIZE)
                .is_ok()
        );
    }

    #[test]
    fn validate_rejects_non_default_group_size() {
        let Ok(device) = make_device() else {
            return;
        };
        let x = device.alloc_zeros::<half::f16>(1024).unwrap();
        let w = device.alloc_zeros::<u32>(1024).unwrap();
        let s = device.alloc_zeros::<half::f16>(1024).unwrap();
        let o = device.alloc_zeros::<half::f16>(1024).unwrap();
        let err = validate_dims_qkv_int4(&x, &w, &w, &w, &s, &s, &s, &o, &o, &o, 64, 64, 128, 64)
            .unwrap_err();
        match err {
            KaioError::InvalidConfig(msg) => assert!(msg.contains("group_size must be")),
            _ => panic!("expected InvalidConfig"),
        }
    }

    #[test]
    fn validate_rejects_k_not_multiple_of_group_size() {
        let Ok(device) = make_device() else {
            return;
        };
        let x = device.alloc_zeros::<half::f16>(1024).unwrap();
        let w = device.alloc_zeros::<u32>(1024).unwrap();
        let s = device.alloc_zeros::<half::f16>(1024).unwrap();
        let o = device.alloc_zeros::<half::f16>(1024).unwrap();
        // K=64 (< GROUP_SIZE=128) fails multiple-of check.
        let err = validate_dims_qkv_int4(
            &x, &w, &w, &w, &s, &s, &s, &o, &o, &o, 64, 64, 64, GROUP_SIZE,
        )
        .unwrap_err();
        match err {
            KaioError::InvalidConfig(msg) => assert!(msg.contains("K must be a multiple")),
            _ => panic!("expected InvalidConfig"),
        }
    }

    #[test]
    fn validate_rejects_small_packed_weight_buffer() {
        let Ok(device) = make_device() else {
            return;
        };
        let m = 64u32;
        let n = 64u32;
        let k = 128u32;
        let x = device.alloc_zeros::<half::f16>((m * k) as usize).unwrap();
        // Undersized packed W — should hold (k/8)*n=1024 u32 but we give 16.
        let w_bad = device.alloc_zeros::<u32>(16).unwrap();
        let w_ok = device
            .alloc_zeros::<u32>(((k / NIBBLES_PER_U32) * n) as usize)
            .unwrap();
        let s = device.alloc_zeros::<half::f16>((n) as usize).unwrap();
        let o = device.alloc_zeros::<half::f16>((m * n) as usize).unwrap();
        let err = validate_dims_qkv_int4(
            &x, &w_bad, &w_ok, &w_ok, &s, &s, &s, &o, &o, &o, m, n, k, GROUP_SIZE,
        )
        .unwrap_err();
        match err {
            KaioError::InvalidConfig(msg) => assert!(msg.contains("W_Q_packed buffer too small")),
            _ => panic!("expected InvalidConfig"),
        }
    }

    #[test]
    fn tile_constants_are_self_consistent() {
        assert_eq!(MMAS_PER_WARP_M * BM, WARP_QUAD_M);
        assert_eq!(MMAS_PER_WARP_N * BN, WARP_QUAD_N);
        assert_eq!(WARP_QUAD_M * 2, BM_BLOCK);
        assert_eq!(WARP_QUAD_N * 2, BN_BLOCK);
        assert_eq!(THREADS_PER_BLOCK, 128);
        // Group-scale reload cadence: 128 / 16 = 8 K-tiles per group.
        assert_eq!(K_TILE_GROUP_RATIO, 8);
    }
}
