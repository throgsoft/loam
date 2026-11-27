use glam::Vec4;

#[derive(Debug, Clone)]
pub(super) struct Closest {
    pub point: Vec4,
    kept: [usize; 5],
    kept_len: usize,
}

impl Closest {
    pub fn kept(&self) -> &[usize] {
        &self.kept[..self.kept_len]
    }
}

pub(super) fn closest_to_origin(simplex: &[Vec4]) -> Closest {
    let n = simplex.len();
    assert!((1..=5).contains(&n), "simplex size {n} out of 1..=5");
    let mut best_dist_sq = f32::MAX;
    let mut best = Closest {
        point: simplex[0],
        kept: [0; 5],
        kept_len: 1,
    };
    for mask in 1u32..(1u32 << n) {
        let mut subset = [0; 5];
        let mut count = 0;
        for i in 0..n {
            if mask & (1 << i) != 0 {
                subset[count] = i;
                count += 1;
            }
        }
        let Some((point, weights)) = project_origin_onto_affine_hull(&subset[..count], simplex)
        else {
            continue;
        };
        if !weights[..count].iter().all(|&w| w >= 0.0) {
            continue;
        }
        let dist_sq = point.length_squared();
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best = Closest {
                point,
                kept: subset,
                kept_len: count,
            };
        }
    }
    best
}

pub(super) fn project_origin_onto_affine_hull(
    subset: &[usize],
    simplex: &[Vec4],
) -> Option<(Vec4, [f32; 5])> {
    let n = subset.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some((simplex[subset[0]], [1.0, 0.0, 0.0, 0.0, 0.0]));
    }
    let v0 = simplex[subset[0]].as_dvec4();
    let k = n - 1;
    let mut dirs = [glam::DVec4::ZERO; 4];
    for (dir, &i) in dirs.iter_mut().zip(&subset[1..]) {
        *dir = simplex[i].as_dvec4() - v0;
    }
    let scale = dirs[..k].iter().map(|v| v.length()).fold(0.0_f64, f64::max);
    if !scale.is_finite() || scale == 0.0 {
        return None;
    }
    let alphas = if k == 4 {
        let mut matrix = [[0.0; 4]; 4];
        for (row, values) in matrix.iter_mut().enumerate() {
            for (col, value) in values.iter_mut().enumerate() {
                *value = dirs[col][row];
            }
        }
        solve_full_rank(matrix, (-v0).to_array(), scale)?
    } else {
        // Golub and Van Loan, Matrix Computations (2013), Chapter 5.
        let mut q = [glam::DVec4::ZERO; 4];
        let mut r = [[0.0; 4]; 4];
        for col in 0..k {
            let mut residual = dirs[col];
            for _ in 0..2 {
                for row in 0..col {
                    let projection = q[row].dot(residual);
                    r[row][col] += projection;
                    residual -= projection * q[row];
                }
            }
            let norm = residual.length();
            if norm <= scale * 1e-12 {
                return None;
            }
            r[col][col] = norm;
            q[col] = residual / norm;
        }
        let mut alphas = [0.0; 4];
        for row in (0..k).rev() {
            let remainder: f64 = ((row + 1)..k).map(|col| r[row][col] * alphas[col]).sum();
            alphas[row] = (-q[row].dot(v0) - remainder) / r[row][row];
        }
        alphas
    };
    let mut weights = [0.0; 5];
    weights[0] = (1.0 - alphas[..k].iter().sum::<f64>()) as f32;
    for (weight, alpha) in weights[1..n].iter_mut().zip(&alphas[..k]) {
        *weight = *alpha as f32;
    }
    let mut point = v0;
    for (i, &alpha) in alphas[..k].iter().enumerate() {
        point += dirs[i] * alpha;
    }
    Some((point.as_vec4(), weights))
}

fn solve_full_rank(mut matrix: [[f64; 4]; 4], mut rhs: [f64; 4], scale: f64) -> Option<[f64; 4]> {
    for col in 0..4 {
        let mut pivot = col;
        for row in (col + 1)..4 {
            if matrix[row][col].abs() > matrix[pivot][col].abs() {
                pivot = row;
            }
        }
        if matrix[pivot][col].abs() <= scale * 1e-12 {
            return None;
        }
        matrix.swap(col, pivot);
        rhs.swap(col, pivot);
        let pivot_row = matrix[col];
        for row in (col + 1)..4 {
            let factor = matrix[row][col] / pivot_row[col];
            for (value, &pivot_value) in matrix[row][col..].iter_mut().zip(&pivot_row[col..]) {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut solution = [0.0; 4];
    for row in (0..4).rev() {
        let remainder: f64 = ((row + 1)..4)
            .map(|col| matrix[row][col] * solution[col])
            .sum();
        solution[row] = (rhs[row] - remainder) / matrix[row][row];
    }
    Some(solution)
}
