//! Matrix multiplication via kaio-ops — real-world GPU compute.
//!
//! Run: cargo run --example matmul -p kaio-ops

use kaio::prelude::*;
use kaio_ops::matmul;

fn main() -> Result<()> {
    let device = KaioDevice::new(0)?;

    // 4x3 * 3x2 = 4x2
    let (m, n, k) = (4u32, 2u32, 3u32);

    #[rustfmt::skip]
    let a = vec![
        1.0f32, 2.0, 3.0,
        4.0,    5.0, 6.0,
        7.0,    8.0, 9.0,
        10.0,  11.0, 12.0,
    ];
    #[rustfmt::skip]
    let b = vec![
        1.0f32, 2.0,
        3.0,    4.0,
        5.0,    6.0,
    ];

    let d_a = device.alloc_from(&a)?;
    let d_b = device.alloc_from(&b)?;
    let mut d_c = device.alloc_zeros::<f32>((m * n) as usize)?;

    matmul(&device, &d_a, &d_b, &mut d_c, m, n, k)?;

    let result = d_c.to_host(&device)?;

    // CPU reference
    let mut expected = vec![0.0f32; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            for p in 0..k as usize {
                expected[i * n as usize + j] += a[i * k as usize + p] * b[p * n as usize + j];
            }
        }
    }

    println!("A ({m}x{k}):");
    for i in 0..m as usize {
        println!("  {:?}", &a[i * k as usize..(i + 1) * k as usize]);
    }
    println!("B ({k}x{n}):");
    for i in 0..k as usize {
        println!("  {:?}", &b[i * n as usize..(i + 1) * n as usize]);
    }
    println!("C = A * B ({m}x{n}):");
    for i in 0..m as usize {
        println!("  {:?}", &result[i * n as usize..(i + 1) * n as usize]);
    }
    println!("matches CPU: {}", result == expected);

    Ok(())
}
