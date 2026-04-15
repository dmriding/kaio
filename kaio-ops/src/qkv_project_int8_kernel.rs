//! Fused tri-output INT8 QKV projection — W8A16 (f16 activations, INT8
//! weights, scalar per-projection scales).
//!
//! Sprint 7.3 MVS. Single kernel launch produces three f16 outputs
//! (Q, K, V) ready to feed [`attention_tc`][crate::attention_tc] — saves
//! 2× global reads of the shared activation X compared to three
//! separate [`matmul_int8`][crate::matmul_int8] calls, and amortizes
//! kernel launch overhead (dominant at autoregressive-decode batch
//! sizes).
//!
//! # W8A16 vs W8A8
//!
//! [`matmul_int8`][crate::matmul_int8] (Sprint 7.1) is **W8A8** (`i8`
//! activations × `i8` weights, native `mma.sync.m16n8k32.s8.s8.s32`).
//! `qkv_project_int8` is **W8A16** (`f16` activations × `i8` weights,
//! `mma.sync.m16n8k16.f16.f16.f32` after per-weight `cvt.rn.f16.s8`).
//! The W8A16 contract lets the op drop in between f16-boundary layers
//! of a real LLM without forcing the caller to quantize X at every
//! attention block. Users who genuinely need W8A8 call `matmul_int8`
//! three times and manage their own activation quantization.
//!
//! # Unified mma path with INT4
//!
//! Both `qkv_project_int8` and [`qkv_project_int4`][super::qkv_project_int4_kernel]
//! target the same `mma.sync.m16n8k16.f16.f16.f32` shape with
//! `K_TILE_SHARED = 16`. Fragment-C layout and the store-out helper
//! ([`emit_store_fragment_c_f32_to_f16_packed`][super::store_out::emit_store_fragment_c_f32_to_f16_packed]
//! once wired) are shared — only fragment-B dequant differs
//! (INT8: `cvt.rn.f16.s8` + scalar scale; INT4: nibble-extract +
//! sign-extend + group-scale fold).
//!
//! # Tri-output design (Serial fusion, Design S)
//!
//! Per K-tile iteration: cooperative-load X tile once into shared,
//! then sequentially for each projection P ∈ {Q, K, V}: load W_P i8
//! tile into a shared weight slot, dequant fragment B per-warp,
//! `mma` into a per-projection f32 accumulator bank. Three fragment-C
//! banks persist across the entire K-loop; scalar scales are applied
//! at store-out, not pre-mma.
//!
//! # Register budget
//!
//! Tripled fragment-C (3 × 16 f32 regs per lane = 48) is the dominant
//! live state. `MMAS_PER_WARP_N` is halved from `matmul_int4`'s 4 to 2
//! to stay inside the 64-reg occupancy cliff on sm_89 — output tile
//! becomes 64×32 per block. D2.5 register-pressure skeleton checkpoint
//! runs ahead of the full kernel body to catch any surprise.

use kaio::prelude::*;

// --- mma.sync.m16n8k16 instance shape (f16 inputs, f32 accumulator) ---
// Matches the shape used by matmul_int4 and matmul_tc. Both QKV projection
// variants (INT8 and INT4) share this mma after the W8A16 switch.
#[allow(dead_code)] // wired up in D3
pub(crate) const BM: u32 = 16; // mma m dim
#[allow(dead_code)] // wired up in D3
pub(crate) const BN: u32 = 8; // mma n dim

// --- Multi-warp block tiling (halved N vs matmul_int4/int8 to fit tri-output accumulators) ---
#[allow(dead_code)] // wired up in D3
pub(crate) const BM_BLOCK: u32 = 64; // output rows per block
#[allow(dead_code)] // wired up in D3
pub(crate) const BN_BLOCK: u32 = 32; // output cols per block — halved vs standalone (see register budget)
#[allow(dead_code)] // wired up in D3
pub(crate) const WARP_QUAD_M: u32 = 32; // rows per warp quadrant
#[allow(dead_code)] // wired up in D3
pub(crate) const WARP_QUAD_N: u32 = 16; // cols per warp quadrant
#[allow(dead_code)] // wired up in D3
pub(crate) const MMAS_PER_WARP_M: u32 = WARP_QUAD_M / BM; // 2
#[allow(dead_code)] // wired up in D3
pub(crate) const MMAS_PER_WARP_N: u32 = WARP_QUAD_N / BN; // 2
#[allow(dead_code)] // wired up in D3
pub(crate) const WARPS_PER_BLOCK: u32 = 4;
#[allow(dead_code)] // wired up in D3
pub(crate) const THREADS_PER_BLOCK: u32 = WARPS_PER_BLOCK * 32; // 128

// --- K-tile granularity (unified with matmul_int4 after W8A16 switch) ---
#[allow(dead_code)] // wired up in D3
pub(crate) const K_TILE_SHARED: u32 = 16;

/// Validate shape + alignment preconditions for `qkv_project_int8`.
///
/// Sprint 7.3 D1. Enforces the **W8A16 MHA** contract:
///
/// - `M`, `N`, `K` all non-zero.
/// - `K % K_TILE_SHARED == 0` — mma K-tile is structural, no ragged tail inside a K-tile.
/// - `N % 2 == 0` — store-out packs adjacent f16 output pairs into one `.b32`.
/// - `N_q == N_k == N_v == N` — v1 is strict MHA. Grouped-query attention (GQA),
///   where `N_kv < N_q`, is a follow-up op (`qkv_project_gqa`); users with GQA
///   weights should call three separate `matmul_int{4,8}`s.
/// - Buffer-size sanity:
///   - `x >= M * K`
///   - each of `w_q_i8`, `w_k_i8`, `w_v_i8` >= `K * N`
///   - each of `q_out`, `k_out`, `v_out` >= `M * N`
///
/// `M` and `N` may be any positive value — edge-tile predication in the
/// kernel handles ragged output (same posture as `matmul_int4` / `matmul_int8`).
///
/// # Pointer distinctness
///
/// `q_out`, `k_out`, `v_out` must be **three distinct allocations**.
/// Aliasing (two outputs pointing into overlapping device memory) would
/// cause silent data corruption — the tri-output store epilogue writes
/// all three banks unconditionally, and overlapping writes race.
/// Rust's `&mut` borrow rules prevent accidental aliasing at the
/// variable level, and KAIO does not expose buffer-splitting APIs that
/// could produce overlapping `GpuBuffer` views, so this is a
/// pathological-caller guard enforced at the documentation level; a
/// cudarc-level device-ptr check is deferred (requires a `CudaStream`
/// argument, which validate is meant to run without).
#[allow(dead_code)] // wired up in D4
pub(crate) fn validate_dims_qkv_int8(
    x: &GpuBuffer<half::f16>,
    w_q_i8: &GpuBuffer<i8>,
    w_k_i8: &GpuBuffer<i8>,
    w_v_i8: &GpuBuffer<i8>,
    q_out: &GpuBuffer<half::f16>,
    k_out: &GpuBuffer<half::f16>,
    v_out: &GpuBuffer<half::f16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(KaioError::InvalidConfig(
            "qkv_project_int8: M, N, K dimensions must be non-zero".to_string(),
        ));
    }
    if !k.is_multiple_of(K_TILE_SHARED) {
        return Err(KaioError::InvalidConfig(format!(
            "qkv_project_int8: K must be a multiple of K_TILE_SHARED={K_TILE_SHARED} \
             (got {k}). The mma.sync.m16n8k16 instance shape requires K-tile size 16; \
             K is not edge-padded inside a K-tile."
        )));
    }
    if !n.is_multiple_of(2) {
        return Err(KaioError::InvalidConfig(format!(
            "qkv_project_int8: N must be even (got {n}). The store-out path packs \
             adjacent f16 output pairs into one .b32; odd N would leave a ragged \
             last column that the current store epilogue does not handle."
        )));
    }

    let mk = (m as usize) * (k as usize);
    let kn = (k as usize) * (n as usize);
    let mn = (m as usize) * (n as usize);

    if x.len() < mk {
        return Err(KaioError::InvalidConfig(format!(
            "qkv_project_int8: X buffer too small: need {mk} f16 ({m}×{k}), got {}",
            x.len()
        )));
    }
    for (label, buf) in [("W_Q", w_q_i8), ("W_K", w_k_i8), ("W_V", w_v_i8)] {
        if buf.len() < kn {
            return Err(KaioError::InvalidConfig(format!(
                "qkv_project_int8: {label} buffer too small: need {kn} i8 ({k}×{n}), got {}",
                buf.len()
            )));
        }
    }
    for (label, buf) in [("Q_out", q_out), ("K_out", k_out), ("V_out", v_out)] {
        if buf.len() < mn {
            return Err(KaioError::InvalidConfig(format!(
                "qkv_project_int8: {label} buffer too small: need {mn} f16 ({m}×{n}), got {}",
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
            return; // no GPU in CI host build — skip
        };
        let m = 64;
        let n = 64;
        let k = 128;
        let x = device.alloc_zeros::<half::f16>((m * k) as usize).unwrap();
        let w_q = device.alloc_zeros::<i8>((k * n) as usize).unwrap();
        let w_k = device.alloc_zeros::<i8>((k * n) as usize).unwrap();
        let w_v = device.alloc_zeros::<i8>((k * n) as usize).unwrap();
        let q_out = device.alloc_zeros::<half::f16>((m * n) as usize).unwrap();
        let k_out = device.alloc_zeros::<half::f16>((m * n) as usize).unwrap();
        let v_out = device.alloc_zeros::<half::f16>((m * n) as usize).unwrap();
        assert!(
            validate_dims_qkv_int8(&x, &w_q, &w_k, &w_v, &q_out, &k_out, &v_out, m, n, k).is_ok()
        );
    }

    #[test]
    fn validate_rejects_zero_dim() {
        let Ok(device) = make_device() else {
            return;
        };
        let x = device.alloc_zeros::<half::f16>(64).unwrap();
        let w = device.alloc_zeros::<i8>(64).unwrap();
        let o = device.alloc_zeros::<half::f16>(64).unwrap();
        let err = validate_dims_qkv_int8(&x, &w, &w, &w, &o, &o, &o, 0, 64, 16).unwrap_err();
        assert!(matches!(err, KaioError::InvalidConfig(_)));
    }

    #[test]
    fn validate_rejects_k_not_multiple_of_tile() {
        let Ok(device) = make_device() else {
            return;
        };
        let x = device.alloc_zeros::<half::f16>(1024).unwrap();
        let w = device.alloc_zeros::<i8>(1024).unwrap();
        let o = device.alloc_zeros::<half::f16>(1024).unwrap();
        // K=17 is not a multiple of K_TILE_SHARED=16
        let err = validate_dims_qkv_int8(&x, &w, &w, &w, &o, &o, &o, 16, 16, 17).unwrap_err();
        match err {
            KaioError::InvalidConfig(msg) => assert!(msg.contains("K_TILE_SHARED")),
            _ => panic!("expected InvalidConfig"),
        }
    }

    #[test]
    fn validate_rejects_odd_n() {
        let Ok(device) = make_device() else {
            return;
        };
        let x = device.alloc_zeros::<half::f16>(1024).unwrap();
        let w = device.alloc_zeros::<i8>(1024).unwrap();
        let o = device.alloc_zeros::<half::f16>(1024).unwrap();
        // N=7 is odd
        let err = validate_dims_qkv_int8(&x, &w, &w, &w, &o, &o, &o, 16, 7, 16).unwrap_err();
        match err {
            KaioError::InvalidConfig(msg) => assert!(msg.contains("N must be even")),
            _ => panic!("expected InvalidConfig"),
        }
    }

    #[test]
    fn validate_rejects_small_buffer() {
        let Ok(device) = make_device() else {
            return;
        };
        // Allocate smaller than M*K=64*32=2048; X undersized.
        let x = device.alloc_zeros::<half::f16>(128).unwrap();
        let w = device.alloc_zeros::<i8>(4096).unwrap();
        let o = device.alloc_zeros::<half::f16>(4096).unwrap();
        let err = validate_dims_qkv_int8(&x, &w, &w, &w, &o, &o, &o, 64, 64, 32).unwrap_err();
        match err {
            KaioError::InvalidConfig(msg) => assert!(msg.contains("X buffer too small")),
            _ => panic!("expected InvalidConfig"),
        }
    }

    #[test]
    fn tile_constants_are_self_consistent() {
        // Sanity: the derived MMAS_PER_WARP_* must multiply back to the warp quadrant.
        assert_eq!(MMAS_PER_WARP_M * BM, WARP_QUAD_M);
        assert_eq!(MMAS_PER_WARP_N * BN, WARP_QUAD_N);
        // And 4 warps × 32×16 quadrant cover the 64×32 block.
        assert_eq!(WARP_QUAD_M * 2, BM_BLOCK); // 2 warp rows × 32 = 64
        assert_eq!(WARP_QUAD_N * 2, BN_BLOCK); // 2 warp cols × 16 = 32
        assert_eq!(THREADS_PER_BLOCK, 128);
    }
}
