//! Every generator here is origin-centered at circumradius `r`.

use glam::Vec4;

pub fn pentatope_vertices(r: f32) -> Vec<Vec4> {
    let k = r;
    let base_w = -r * 0.25;
    let base_r = r * (15.0_f32).sqrt() / 4.0;
    let t = base_r / 3.0_f32.sqrt();
    vec![
        Vec4::new(0.0, 0.0, 0.0, k),
        Vec4::new(t, t, t, base_w),
        Vec4::new(t, -t, -t, base_w),
        Vec4::new(-t, t, -t, base_w),
        Vec4::new(-t, -t, t, base_w),
    ]
}

/// `(±r/2, ±r/2, ±r/2, ±r/2)` gives circumradius `r`.
pub fn tesseract_vertices(r: f32) -> Vec<Vec4> {
    let a = r * 0.5;
    let mut v = Vec::with_capacity(16);
    for &w in &[-a, a] {
        for &z in &[-a, a] {
            for &y in &[-a, a] {
                for &x in &[-a, a] {
                    v.push(Vec4::new(x, y, z, w));
                }
            }
        }
    }
    v
}

pub fn cell16_vertices(r: f32) -> Vec<Vec4> {
    vec![
        Vec4::new(r, 0.0, 0.0, 0.0),
        Vec4::new(-r, 0.0, 0.0, 0.0),
        Vec4::new(0.0, r, 0.0, 0.0),
        Vec4::new(0.0, -r, 0.0, 0.0),
        Vec4::new(0.0, 0.0, r, 0.0),
        Vec4::new(0.0, 0.0, -r, 0.0),
        Vec4::new(0.0, 0.0, 0.0, r),
        Vec4::new(0.0, 0.0, 0.0, -r),
    ]
}

/// All 24 permutations of `(±r/√2, ±r/√2, 0, 0)`.
pub fn cell24_vertices(r: f32) -> Vec<Vec4> {
    let k = r / 2.0_f32.sqrt();
    let mut v = Vec::with_capacity(24);
    for i in 0..4 {
        for j in (i + 1)..4 {
            for &si in &[-k, k] {
                for &sj in &[-k, k] {
                    let mut c = [0.0_f32; 4];
                    c[i] = si;
                    c[j] = sj;
                    v.push(Vec4::new(c[0], c[1], c[2], c[3]));
                }
            }
        }
    }
    v
}

fn even_permutations_4<T: Copy>(arr: [T; 4]) -> [[T; 4]; 12] {
    [
        [arr[0], arr[1], arr[2], arr[3]],
        [arr[1], arr[2], arr[0], arr[3]], // (012)
        [arr[2], arr[0], arr[1], arr[3]], // (021)
        [arr[1], arr[3], arr[2], arr[0]], // (013)
        [arr[3], arr[0], arr[2], arr[1]], // (031)
        [arr[2], arr[1], arr[3], arr[0]], // (023)
        [arr[3], arr[1], arr[0], arr[2]], // (032)
        [arr[0], arr[2], arr[3], arr[1]], // (123)
        [arr[0], arr[3], arr[1], arr[2]], // (132)
        [arr[1], arr[0], arr[3], arr[2]], // (01)(23)
        [arr[2], arr[3], arr[0], arr[1]], // (02)(13)
        [arr[3], arr[2], arr[1], arr[0]], // (03)(12)
    ]
}

// Wikipedia, 600-cell.
/// Vertex set at circumradius 1 (Wikipedia "600-cell").
pub fn cell600_vertices(r: f32) -> Vec<Vec4> {
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let mut v = Vec::with_capacity(120);

    for axis in 0..4 {
        for sign in [r, -r] {
            let mut c = [0.0_f32; 4];
            c[axis] = sign;
            v.push(Vec4::from_array(c));
        }
    }

    let h = r * 0.5;
    for s in 0..16u32 {
        let x = if s & 1 == 1 { -h } else { h };
        let y = if (s >> 1) & 1 == 1 { -h } else { h };
        let z = if (s >> 2) & 1 == 1 { -h } else { h };
        let w = if (s >> 3) & 1 == 1 { -h } else { h };
        v.push(Vec4::new(x, y, z, w));
    }

    let base = [0.0_f32, r * 0.5, r * phi * 0.5, r / (2.0 * phi)];
    for perm in even_permutations_4(base) {
        for sign_mask in 0..8u32 {
            let mut x = perm;
            let mut k = 0usize;
            for xi in x.iter_mut() {
                if *xi != 0.0 {
                    if (sign_mask >> k) & 1 == 1 {
                        *xi = -*xi;
                    }
                    k += 1;
                }
            }
            v.push(Vec4::from_array(x));
        }
    }

    v
}

// Wikipedia, 120-cell.
/// Vertex set at circumradius `2√2` before rescaling (Wikipedia "120-cell").
pub fn cell120_vertices(r: f32) -> Vec<Vec4> {
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let phi2 = phi * phi;
    let inv_phi = 1.0 / phi;
    let inv_phi2 = inv_phi * inv_phi;
    let sqrt5 = 5.0_f32.sqrt();
    let scale = r / (2.0 * 2.0_f32.sqrt());
    let mut v = Vec::with_capacity(600);

    for i in 0..4 {
        for j in (i + 1)..4 {
            for si in [2.0_f32, -2.0] {
                for sj in [2.0_f32, -2.0] {
                    let mut c = [0.0_f32; 4];
                    c[i] = si * scale;
                    c[j] = sj * scale;
                    v.push(Vec4::from_array(c));
                }
            }
        }
    }

    let mut emit_one_special = |special: f32, common: f32| {
        for special_pos in 0..4 {
            for sm in 0..16u32 {
                let mut c = [0.0_f32; 4];
                for (i, ci) in c.iter_mut().enumerate() {
                    let val = if i == special_pos { special } else { common };
                    let sign = if (sm >> i) & 1 == 1 { -1.0 } else { 1.0 };
                    *ci = val * sign * scale;
                }
                v.push(Vec4::from_array(c));
            }
        }
    };

    emit_one_special(sqrt5, 1.0);
    emit_one_special(phi2, inv_phi);
    emit_one_special(inv_phi2, phi);

    let mut emit_even_zero = |a: f32, b: f32, c: f32| {
        let base = [0.0_f32, a, b, c];
        for perm in even_permutations_4(base) {
            for sign_mask in 0..8u32 {
                let mut x = perm;
                let mut k = 0usize;
                for xi in x.iter_mut() {
                    if *xi != 0.0 {
                        if (sign_mask >> k) & 1 == 1 {
                            *xi = -*xi;
                        }
                        k += 1;
                    }
                }
                for ci in &mut x {
                    *ci *= scale;
                }
                v.push(Vec4::from_array(x));
            }
        }
    };

    emit_even_zero(inv_phi2, 1.0, phi2);
    emit_even_zero(inv_phi, phi, sqrt5);

    let base7 = [inv_phi, 1.0, phi, 2.0_f32];
    for perm in even_permutations_4(base7) {
        for sm in 0..16u32 {
            let mut x = perm;
            for (i, xi) in x.iter_mut().enumerate() {
                if (sm >> i) & 1 == 1 {
                    *xi = -*xi;
                }
            }
            for ci in &mut x {
                *ci *= scale;
            }
            v.push(Vec4::from_array(x));
        }
    }

    v
}
