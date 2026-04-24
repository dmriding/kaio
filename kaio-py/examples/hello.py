"""End-to-end smoke demo for the KAIO Python bindings.

Creates a device, loads two f16 matrices from NumPy, runs KAIO's
tensor-core matmul, converts the result back to NumPy, and sanity-
checks against a NumPy f32 reference.

Run with `python examples/hello.py` after `maturin develop`.
"""

import numpy as np
import kaio


def main() -> int:
    device = kaio.Device(0)
    print(f"GPU: {device.name} (sm_{device.compute_capability[0]}{device.compute_capability[1]})")
    print()

    M, N, K = 64, 64, 64
    np.random.seed(42)
    a_np = np.random.uniform(-1, 1, (M, K)).astype(np.float16)
    b_np = np.random.uniform(-1, 1, (K, N)).astype(np.float16)

    a = kaio.Tensor.from_numpy(device, a_np)
    b = kaio.Tensor.from_numpy(device, b_np)
    print(f"inputs:  A={a!r}, B={b!r}")

    c = kaio.matmul_tc(a, b)
    c_np = c.to_numpy()
    print(f"output:  C={c!r}")

    ref = a_np.astype(np.float32) @ b_np.astype(np.float32)
    max_abs_err = float(np.max(np.abs(c_np - ref)))
    print()
    print(f"output shape:   {c_np.shape}, dtype: {c_np.dtype}")
    print(f"max abs error:  {max_abs_err:.4f}  (vs NumPy f32 reference)")

    if c_np.shape != (M, N):
        print(f"FAIL: expected shape ({M}, {N})")
        return 1
    if c_np.dtype != np.float32:
        print(f"FAIL: expected float32 output, got {c_np.dtype}")
        return 1
    # f16 matmul tolerance at this scale + amplitude range
    if max_abs_err > 0.5:
        print(f"FAIL: max abs error {max_abs_err} exceeds f16 tolerance 0.5")
        return 1

    print()
    print("hello, KAIO from Python.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
