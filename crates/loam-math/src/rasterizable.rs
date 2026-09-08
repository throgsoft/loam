//! Projection consumes canonical ambient coordinates; tessellation emits samples without owning storage.

use glam::{Vec3, Vec4};

use crate::space::Space;
use crate::{EuclideanR3, EuclideanR4};

const PROJECTION_DENOM_EPSILON: f32 = 1e-4;

/// Floor on `1 - dot(p, pole)` at the projection singularity.
pub const STEREOGRAPHIC_POLE_EPSILON: f32 = 1e-4;

// do Carmo, Differential Geometry of Curves and Surfaces, 1976, §1.4.
fn perp_frame(n: Vec4) -> (Vec4, Vec4, Vec4) {
    let ax = n.x.abs();
    let ay = n.y.abs();
    let az = n.z.abs();
    let aw = n.w.abs();
    let drop = ax.max(ay).max(az).max(aw);
    let mut seeds = [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W];
    let drop_idx = if drop == ax {
        0
    } else if drop == ay {
        1
    } else if drop == az {
        2
    } else {
        3
    };
    seeds[drop_idx] = Vec4::ZERO;

    let mut basis = [Vec4::ZERO; 3];
    let mut count = 0usize;
    for s in seeds {
        if s == Vec4::ZERO {
            continue;
        }
        let mut v = s - s.dot(n) * n;
        for b in basis.iter().take(count) {
            v -= v.dot(*b) * *b;
        }
        basis[count] = v.normalize();
        count += 1;
    }
    (basis[0], basis[1], basis[2])
}

// Wikipedia, Stereographic projection.
pub(crate) fn stereographic_to_r3(p: Vec4, pole: Vec4) -> Vec3 {
    let dot = p.dot(pole).clamp(-1.0, 1.0);
    let denom = (1.0 - dot).max(STEREOGRAPHIC_POLE_EPSILON);
    if pole == Vec4::W {
        return Vec3::new(p.x, p.y, p.z) / denom;
    }
    let perp = p - dot * pole;
    let scaled = perp / denom;
    let (e1, e2, e3) = perp_frame(pole);
    Vec3::new(scaled.dot(e1), scaled.dot(e2), scaled.dot(e3))
}

/// Unsupported dimension/variant combinations return `Vec3::ZERO`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Projection<const N: usize> {
    /// Keeps the first three components, padding with zero if needed.
    #[default]
    Identity,

    /// Drops the indexed axis; an out-of-range index returns `Vec3::ZERO`.
    Orthographic { drop_axis: usize },

    /// Pinhole at `(0, 0, 0, focal_distance)`; callers must keep every vertex below the eye in w.
    Perspective4D { focal_distance: f32 },

    // Coxeter, Regular Polytopes, 3rd ed., ch. 13.
    /// Projects onto the cell plane; the normal must be outward and unit, and the eye outside the polytope.
    Schlegel {
        cell_normal: Vec4,
        /// Signed plane offset: the cell lies in `{x : dot(cell_normal, x) = cell_offset}`.
        cell_offset: f32,
        /// Eye distance along `cell_normal`; must exceed `cell_offset`.
        viewpoint_distance: f32,
        basis: [Vec4; 3],
    },

    // Wikipedia, Stereographic projection.
    /// Projects unit S³ from a unit pole into its perpendicular 3-flat.
    Stereographic {
        /// Unit pole; `Vec4::W` gives the usual xyz readout.
        pole: Vec4,
    },
}

impl Projection<4> {
    pub fn schlegel(cell_normal: Vec4, cell_offset: f32, viewpoint_distance: f32) -> Projection<4> {
        let (e1, e2, e3) = perp_frame(cell_normal);
        Self::schlegel_with_basis(cell_normal, cell_offset, viewpoint_distance, [e1, e2, e3])
    }

    pub fn schlegel_with_basis(
        cell_normal: Vec4,
        cell_offset: f32,
        viewpoint_distance: f32,
        basis: [Vec4; 3],
    ) -> Projection<4> {
        Projection::Schlegel {
            cell_normal,
            cell_offset,
            viewpoint_distance,
            basis,
        }
    }
}

/// `N` is the const-generic ambient dimension matching the `Visualizable<N>` mesh data in `loam-shape`.
pub trait RasterizableSpace<const N: usize>: Space {
    fn point_to_array(p: Self::Point) -> [f32; N];

    fn array_to_point(arr: [f32; N]) -> Self::Point;

    fn project_point(point: Self::Point, projection: &Projection<N>) -> Vec3;

    /// Emits both endpoints and `samples - 1` interior points when `samples > 1`.
    fn tessellate_segment(
        p0: Self::Point,
        p1: Self::Point,
        samples: usize,
        emit: impl FnMut(Self::Point),
    );
}

impl RasterizableSpace<3> for EuclideanR3 {
    fn point_to_array(p: Vec3) -> [f32; 3] {
        p.to_array()
    }

    fn array_to_point(arr: [f32; 3]) -> Vec3 {
        Vec3::from_array(arr)
    }

    fn project_point(point: Vec3, projection: &Projection<3>) -> Vec3 {
        match projection {
            Projection::Identity => point,
            Projection::Orthographic { drop_axis } => match *drop_axis {
                0 => Vec3::new(point.y, point.z, 0.0),
                1 => Vec3::new(point.x, point.z, 0.0),
                2 => Vec3::new(point.x, point.y, 0.0),
                _ => Vec3::ZERO,
            },
            Projection::Perspective4D { .. }
            | Projection::Schlegel { .. }
            | Projection::Stereographic { .. } => Vec3::ZERO,
        }
    }

    fn tessellate_segment(p0: Vec3, p1: Vec3, samples: usize, mut emit: impl FnMut(Vec3)) {
        emit(p0);
        for i in 1..samples {
            let t = i as f32 / samples as f32;
            emit(p0.lerp(p1, t));
        }
        emit(p1);
    }
}

impl RasterizableSpace<4> for EuclideanR4 {
    fn point_to_array(p: Vec4) -> [f32; 4] {
        p.to_array()
    }

    fn array_to_point(arr: [f32; 4]) -> Vec4 {
        Vec4::from_array(arr)
    }

    fn project_point(point: Vec4, projection: &Projection<4>) -> Vec3 {
        match projection {
            Projection::Identity => Vec3::new(point.x, point.y, point.z),
            Projection::Orthographic { drop_axis } => match *drop_axis {
                0 => Vec3::new(point.y, point.z, point.w),
                1 => Vec3::new(point.x, point.z, point.w),
                2 => Vec3::new(point.x, point.y, point.w),
                3 => Vec3::new(point.x, point.y, point.z),
                _ => Vec3::ZERO,
            },
            Projection::Perspective4D { focal_distance } => {
                let denom = (focal_distance - point.w).max(PROJECTION_DENOM_EPSILON);
                let scale = focal_distance / denom;
                Vec3::new(point.x, point.y, point.z) * scale
            }
            Projection::Schlegel {
                cell_normal,
                cell_offset,
                viewpoint_distance,
                basis,
            } => {
                // Coxeter, Regular Polytopes, 3rd ed., ch. 13.
                let n = *cell_normal;
                let eye = *viewpoint_distance * n;
                let n_dot_eye = n.dot(eye);

                let raw_denom = n.dot(point) - n_dot_eye;
                let denom = if raw_denom.abs() < PROJECTION_DENOM_EPSILON {
                    PROJECTION_DENOM_EPSILON.copysign(raw_denom)
                } else {
                    raw_denom
                };
                let t = (*cell_offset - n_dot_eye) / denom;
                let result = eye + t * (point - eye);
                let [e1, e2, e3] = *basis;
                Vec3::new(result.dot(e1), result.dot(e2), result.dot(e3))
            }
            Projection::Stereographic { pole } => stereographic_to_r3(point.normalize(), *pole),
        }
    }

    fn tessellate_segment(p0: Vec4, p1: Vec4, samples: usize, mut emit: impl FnMut(Vec4)) {
        emit(p0);
        for i in 1..samples {
            let t = i as f32 / samples as f32;
            emit(p0.lerp(p1, t));
        }
        emit(p1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SphericalS3Embedded;
    use approx::assert_relative_eq;

    const GOLDEN_TIE_FRAME: Vec3 = Vec3::new(0.5, 0.5, std::f32::consts::FRAC_1_SQRT_2);

    #[test]
    fn stereographic_r4_normalizes_scaled_input() {
        let proj = Projection::Stereographic { pole: Vec4::W };
        for p in [
            Vec4::new(0.3, -0.1, 0.2, -0.5).normalize(),
            Vec4::new(-0.4, 0.6, 0.1, 0.3).normalize(),
            Vec4::new(0.0, 0.0, 0.0, -1.0),
        ] {
            let unit = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
            for k in [0.25_f32, 1.5, 3.0] {
                let scaled = <EuclideanR4 as RasterizableSpace<4>>::project_point(k * p, &proj);
                assert!(
                    scaled.abs_diff_eq(unit, 1e-5),
                    "scale {k}: {scaled:?} vs {unit:?}"
                );
            }
        }
    }

    #[test]
    fn r3_tessellate_one_sample_appends_endpoints() {
        let p0 = Vec3::new(0.0, 0.0, 0.0);
        let p1 = Vec3::new(2.0, 4.0, -6.0);
        let mut out = Vec::new();
        <EuclideanR3 as RasterizableSpace<3>>::tessellate_segment(p0, p1, 1, |p| out.push(p));
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], p0);
        assert_eq!(out[1], p1);
    }

    #[test]
    fn r3_tessellate_four_samples_produces_five_points() {
        let p0 = Vec3::new(0.0, 0.0, 0.0);
        let p1 = Vec3::new(4.0, 0.0, 0.0);
        let mut out = Vec::new();
        <EuclideanR3 as RasterizableSpace<3>>::tessellate_segment(p0, p1, 4, |p| out.push(p));
        assert_eq!(out.len(), 5);
        assert_eq!(out[0], p0);
        assert_eq!(out[1], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(out[2], Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(out[3], Vec3::new(3.0, 0.0, 0.0));
        assert_eq!(out[4], p1);
    }

    #[test]
    fn r3_orthographic_drops_named_axis() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let pj = |drop_axis| {
            <EuclideanR3 as RasterizableSpace<3>>::project_point(
                p,
                &Projection::Orthographic { drop_axis },
            )
        };
        assert_eq!(pj(0), Vec3::new(2.0, 3.0, 0.0));
        assert_eq!(pj(1), Vec3::new(1.0, 3.0, 0.0));
        assert_eq!(pj(2), Vec3::new(1.0, 2.0, 0.0));
        assert_eq!(pj(3), Vec3::ZERO);
        assert_eq!(pj(99), Vec3::ZERO);
    }

    #[test]
    fn r4_identity_drops_w() {
        let p = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let projected =
            <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &Projection::Identity);
        assert_eq!(projected, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn r4_orthographic_drops_each_axis() {
        let p = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let pj = |drop_axis| {
            <EuclideanR4 as RasterizableSpace<4>>::project_point(
                p,
                &Projection::Orthographic { drop_axis },
            )
        };
        assert_eq!(pj(0), Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(pj(1), Vec3::new(1.0, 3.0, 4.0));
        assert_eq!(pj(2), Vec3::new(1.0, 2.0, 4.0));
        assert_eq!(pj(3), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(pj(4), Vec3::ZERO);
    }

    #[test]
    fn r4_perspective4d_w_zero_is_unchanged() {
        let p = Vec4::new(1.0, 2.0, 3.0, 0.0);
        let proj = Projection::Perspective4D {
            focal_distance: 2.0,
        };
        let got = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
        assert_eq!(got, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn r4_perspective4d_cube_within_cube_scaling() {
        let focal = 2.0;
        let proj = Projection::Perspective4D {
            focal_distance: focal,
        };
        let near = Vec4::new(0.5, 0.5, 0.5, 0.5);
        let far = Vec4::new(0.5, 0.5, 0.5, -0.5);
        let pn = <EuclideanR4 as RasterizableSpace<4>>::project_point(near, &proj);
        let pf = <EuclideanR4 as RasterizableSpace<4>>::project_point(far, &proj);

        let r_near = (pn.length() / 0.5_f32.mul_add(3.0_f32.sqrt(), 0.0)).abs();
        let r_far = (pf.length() / 0.5_f32.mul_add(3.0_f32.sqrt(), 0.0)).abs();
        assert!((r_near - 4.0 / 3.0).abs() < 1e-5, "near scale {r_near}");
        assert!((r_far - 4.0 / 5.0).abs() < 1e-5, "far scale {r_far}");
        assert!(pn.length() > pf.length(), "near={pn:?} far={pf:?}");
    }

    #[test]
    fn r4_perspective4d_at_viewer_clamps_finite() {
        let p = Vec4::new(0.1, 0.2, 0.3, 2.0);
        let proj = Projection::Perspective4D {
            focal_distance: 2.0,
        };
        let got = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
        for c in [got.x, got.y, got.z] {
            assert!(
                c.is_finite(),
                "expected finite output at viewer, got {got:?}"
            );
        }
    }

    #[test]
    fn r3_perspective4d_returns_zero() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let proj = Projection::Perspective4D {
            focal_distance: 2.0,
        };
        let got = <EuclideanR3 as RasterizableSpace<3>>::project_point(p, &proj);
        assert_eq!(got, Vec3::ZERO);
    }

    const TESSERACT_VERTS: [Vec4; 16] = [
        Vec4::new(0.5, 0.5, 0.5, 0.5),
        Vec4::new(-0.5, 0.5, 0.5, 0.5),
        Vec4::new(0.5, -0.5, 0.5, 0.5),
        Vec4::new(-0.5, -0.5, 0.5, 0.5),
        Vec4::new(0.5, 0.5, -0.5, 0.5),
        Vec4::new(-0.5, 0.5, -0.5, 0.5),
        Vec4::new(0.5, -0.5, -0.5, 0.5),
        Vec4::new(-0.5, -0.5, -0.5, 0.5),
        Vec4::new(0.5, 0.5, 0.5, -0.5),
        Vec4::new(-0.5, 0.5, 0.5, -0.5),
        Vec4::new(0.5, -0.5, 0.5, -0.5),
        Vec4::new(-0.5, -0.5, 0.5, -0.5),
        Vec4::new(0.5, 0.5, -0.5, -0.5),
        Vec4::new(-0.5, 0.5, -0.5, -0.5),
        Vec4::new(0.5, -0.5, -0.5, -0.5),
        Vec4::new(-0.5, -0.5, -0.5, -0.5),
    ];

    #[test]
    fn schlegel_chosen_cell_renders_undistorted() {
        let cell_offset = 0.5;
        let proj = Projection::schlegel(Vec4::W, cell_offset, 1.5 * cell_offset);
        let cell: Vec<Vec4> = TESSERACT_VERTS.iter().take(8).copied().collect();
        let projected: Vec<Vec3> = cell
            .iter()
            .map(|v| <EuclideanR4 as RasterizableSpace<4>>::project_point(*v, &proj))
            .collect();

        for i in 0..cell.len() {
            for j in (i + 1)..cell.len() {
                let orig = (Vec3::new(cell[i].x, cell[i].y, cell[i].z)
                    - Vec3::new(cell[j].x, cell[j].y, cell[j].z))
                .length();
                let got = (projected[i] - projected[j]).length();
                assert!(
                    (orig - got).abs() < 1e-5,
                    "chosen-cell distance v{i}-v{j} should be {orig}, got {got}"
                );
            }
        }
    }

    #[test]
    fn schlegel_non_axis_aligned_cell_is_not_flattened() {
        let verts = [
            Vec4::X,
            -Vec4::X,
            Vec4::Y,
            -Vec4::Y,
            Vec4::Z,
            -Vec4::Z,
            Vec4::W,
            -Vec4::W,
        ];

        let centroid = (Vec4::X + Vec4::Y + Vec4::Z + Vec4::W) / 4.0;
        let cell_offset = centroid.length();
        let cell_normal = centroid / cell_offset;
        let proj = Projection::schlegel(cell_normal, cell_offset, 1.5 * cell_offset);

        let boundary = [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W];
        let inner = [-Vec4::X, -Vec4::Y, -Vec4::Z, -Vec4::W];
        let proj_pt = |v: Vec4| <EuclideanR4 as RasterizableSpace<4>>::project_point(v, &proj);

        let boundary_r: Vec<f32> = boundary.iter().map(|&v| proj_pt(v).length()).collect();
        let inner_r: Vec<f32> = inner.iter().map(|&v| proj_pt(v).length()).collect();
        let r0 = boundary_r[0];
        assert!(r0 > 1e-3, "boundary must not collapse to the origin");
        for r in &boundary_r {
            assert!(
                (r - r0).abs() < 1e-5,
                "boundary tetrahedron must be regular, radii {boundary_r:?}"
            );
        }
        for r in &inner_r {
            assert!(
                *r < r0 - 1e-3,
                "every nested vertex must sit inside the boundary, inner {inner_r:?} vs {r0}"
            );
        }

        let proj_boundary: Vec<Vec3> = boundary.iter().map(|&v| proj_pt(v)).collect();
        let edge_len = 2.0_f32.sqrt();
        for i in 0..proj_boundary.len() {
            for j in (i + 1)..proj_boundary.len() {
                let got = (proj_boundary[i] - proj_boundary[j]).length();
                assert!(
                    (got - edge_len).abs() < 1e-5,
                    "oblique boundary edge {i}-{j} should stay {edge_len}, got {got}"
                );
            }
        }

        let all: Vec<Vec3> = verts.iter().map(|&v| proj_pt(v)).collect();
        for axis in 0..3 {
            let comp = |p: Vec3| [p.x, p.y, p.z][axis];
            let spread = all
                .iter()
                .map(|&p| comp(p))
                .fold(f32::NEG_INFINITY, f32::max)
                - all.iter().map(|&p| comp(p)).fold(f32::INFINITY, f32::min);
            assert!(
                spread > 0.5,
                "axis {axis} should have real spread, got {spread}"
            );
        }
    }

    #[test]
    fn schlegel_uses_supplied_basis_for_readout() {
        let p = Vec4::new(0.5, -0.25, 0.125, 0.5);
        let xyz = Projection::schlegel_with_basis(Vec4::W, 0.5, 0.75, [Vec4::X, Vec4::Y, Vec4::Z]);
        let yxz = Projection::schlegel_with_basis(Vec4::W, 0.5, 0.75, [Vec4::Y, Vec4::X, Vec4::Z]);

        let a = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &xyz);
        let b = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &yxz);

        assert_eq!(a, Vec3::new(0.5, -0.25, 0.125));
        assert_eq!(b, Vec3::new(-0.25, 0.5, 0.125));
    }

    #[test]
    fn schlegel_zero_denominator_clamps_finite() {
        let viewpoint_distance = 0.75;
        let proj = Projection::schlegel(Vec4::W, 0.5, viewpoint_distance);

        let p = Vec4::new(0.3, -0.2, 0.1, viewpoint_distance);
        let got = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);
        for c in [got.x, got.y, got.z] {
            assert!(
                c.is_finite(),
                "degenerate denominator should clamp finite, got {got:?}"
            );
        }
    }

    // Wikipedia, Stereographic projection.
    fn stereo_inverse_w_pole(q: Vec3) -> Vec4 {
        let s = q.length_squared();
        let w = (s - 1.0) / (s + 1.0);
        let xyz = q * (1.0 - w);
        Vec4::new(xyz.x, xyz.y, xyz.z, w)
    }

    fn stereo_inverse_general(q: Vec3, pole: Vec4) -> Vec4 {
        let (e1, e2, e3) = perp_frame(pole);
        let perp = q.x * e1 + q.y * e2 + q.z * e3;

        let s = perp.length_squared();
        let dot = (s - 1.0) / (s + 1.0);
        dot * pole + (1.0 - dot) * perp
    }

    #[test]
    fn stereographic_default_pole_is_drop_w_of_scaled() {
        for p in [
            Vec4::new(0.5, 0.5, 0.5, 0.5),
            Vec4::new(-0.5, 0.5, 0.5, -0.5),
            Vec4::new(-0.6, 0.0, 0.8, 0.0), // unit, w = 0
        ] {
            let got = stereographic_to_r3(p, Vec4::W);
            let want = Vec3::new(p.x, p.y, p.z) / (1.0 - p.w);
            assert_eq!(
                got, want,
                "fast path must match canonical formula for {p:?}"
            );
        }

        let general = {
            let p = Vec4::new(0.5, 0.5, 0.5, 0.5);
            let dot = p.dot(Vec4::W).clamp(-1.0, 1.0);
            let denom = (1.0 - dot).max(STEREOGRAPHIC_POLE_EPSILON);
            let perp = p - dot * Vec4::W;
            let scaled = perp / denom;
            let (e1, e2, e3) = perp_frame(Vec4::W);
            Vec3::new(scaled.dot(e1), scaled.dot(e2), scaled.dot(e3))
        };
        assert_eq!(
            general,
            stereographic_to_r3(Vec4::new(0.5, 0.5, 0.5, 0.5), Vec4::W)
        );
    }

    #[test]
    fn stereographic_inverts_off_pole() {
        let proj_w = Projection::Stereographic { pole: Vec4::W };
        for p in [
            Vec4::new(0.2, 0.1, -0.3, 0.4).normalize(),
            Vec4::new(-0.5, 0.5, 0.5, -0.5),
            Vec4::new(0.7, -0.2, 0.1, 0.1).normalize(),
        ] {
            let img = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj_w);
            let back = stereo_inverse_w_pole(img);
            assert_relative_eq!(back.x, p.x, epsilon = 1e-5);
            assert_relative_eq!(back.y, p.y, epsilon = 1e-5);
            assert_relative_eq!(back.z, p.z, epsilon = 1e-5);
            assert_relative_eq!(back.w, p.w, epsilon = 1e-5);
        }
        let pole = Vec4::new(0.1, -0.2, 0.3, 0.9).normalize();
        let proj_n = Projection::Stereographic { pole };
        for p in [
            Vec4::new(0.6, 0.5, -0.2, 0.0).normalize(),
            Vec4::new(-0.3, 0.4, 0.5, -0.2).normalize(),
        ] {
            let img = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj_n);
            let back = stereo_inverse_general(img, pole);
            assert_relative_eq!(back.x, p.x, epsilon = 1e-5);
            assert_relative_eq!(back.y, p.y, epsilon = 1e-5);
            assert_relative_eq!(back.z, p.z, epsilon = 1e-5);
            assert_relative_eq!(back.w, p.w, epsilon = 1e-5);
        }
    }

    #[test]
    fn stereographic_pole_denominator_clamped_finite() {
        for pole in [Vec4::W, Vec4::new(0.5, 0.5, 0.5, 0.5)] {
            let proj = Projection::Stereographic { pole };
            let at_pole = <EuclideanR4 as RasterizableSpace<4>>::project_point(pole, &proj);
            for c in [at_pole.x, at_pole.y, at_pole.z] {
                assert!(
                    c.is_finite(),
                    "pole input must clamp finite, got {at_pole:?}"
                );
            }

            let (e1, _, _) = perp_frame(pole);
            let near = (pole * 0.9999 + e1 * 0.01).normalize();
            let near_img = <EuclideanR4 as RasterizableSpace<4>>::project_point(near, &proj);
            for c in [near_img.x, near_img.y, near_img.z] {
                assert!(
                    c.is_finite(),
                    "near-pole input must stay finite, got {near_img:?}"
                );
            }
        }
    }

    #[test]
    fn stereographic_antipode_maps_to_origin() {
        for pole in [Vec4::W, Vec4::new(0.1, -0.2, 0.3, 0.9).normalize()] {
            let proj = Projection::Stereographic { pole };
            let got = <EuclideanR4 as RasterizableSpace<4>>::project_point(-pole, &proj);
            assert_relative_eq!(got.x, 0.0, epsilon = 1e-6);
            assert_relative_eq!(got.y, 0.0, epsilon = 1e-6);
            assert_relative_eq!(got.z, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn stereographic_frame_breaks_axis_ties_in_index_order() {
        let pole = Vec4::new(0.0, 0.0, 1.0, 1.0).normalize();
        let proj = Projection::Stereographic { pole };
        let p = Vec4::new(0.5, 0.5, -0.5, 0.5);
        let first = <EuclideanR4 as RasterizableSpace<4>>::project_point(p, &proj);

        assert_relative_eq!(first.x, GOLDEN_TIE_FRAME.x, epsilon = 1e-6);
        assert_relative_eq!(first.y, GOLDEN_TIE_FRAME.y, epsilon = 1e-6);
        assert_relative_eq!(first.z, GOLDEN_TIE_FRAME.z, epsilon = 1e-6);
    }

    #[test]
    fn stereographic_frame_orthonormal_for_every_pole() {
        let mut poles = vec![Vec4::X, Vec4::Y, Vec4::Z, Vec4::W];
        for axis in [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W] {
            for k in 1..6 {
                let t = k as f32 / 6.0;
                poles.push((Vec4::splat(0.5) * (1.0 - t) + axis * t).normalize());
            }
        }
        for pole in poles {
            let (e1, e2, e3) = perp_frame(pole);
            for e in [e1, e2, e3] {
                assert!(e.is_finite(), "frame must be finite for pole {pole:?}");
            }
            assert_relative_eq!(e1.dot(e1), 1.0, epsilon = 1e-5);
            assert_relative_eq!(e2.dot(e2), 1.0, epsilon = 1e-5);
            assert_relative_eq!(e3.dot(e3), 1.0, epsilon = 1e-5);
            assert!(e1.dot(e2).abs() < 1e-5, "e1·e2 for pole {pole:?}");
            assert!(e1.dot(e3).abs() < 1e-5, "e1·e3 for pole {pole:?}");
            assert!(e2.dot(e3).abs() < 1e-5, "e2·e3 for pole {pole:?}");
            assert!(e1.dot(pole).abs() < 1e-5, "e1 ⟂ pole {pole:?}");
            assert!(e2.dot(pole).abs() < 1e-5, "e2 ⟂ pole {pole:?}");
            assert!(e3.dot(pole).abs() < 1e-5, "e3 ⟂ pole {pole:?}");
        }
    }

    #[test]
    fn stereographic_is_conformal() {
        let s = SphericalS3Embedded;
        let pole = Vec4::W;
        let proj = Projection::Stereographic { pole };
        let v = Vec4::new(0.3, -0.1, 0.2, -0.5).normalize();
        let a = Vec4::new(0.5, 0.4, -0.1, -0.3).normalize();
        let b = Vec4::new(-0.2, 0.3, 0.6, -0.4).normalize();

        let ta = s.log(v, a);
        let tb = s.log(v, b);
        let intrinsic = (ta.dot(tb) / (ta.length() * tb.length()))
            .clamp(-1.0, 1.0)
            .acos();

        let step = 1e-3;
        let pv = <EuclideanR4 as RasterizableSpace<4>>::project_point(v, &proj);
        let pa = <EuclideanR4 as RasterizableSpace<4>>::project_point(
            s.exp(v, ta.normalize() * step),
            &proj,
        );
        let pb = <EuclideanR4 as RasterizableSpace<4>>::project_point(
            s.exp(v, tb.normalize() * step),
            &proj,
        );
        let da = pa - pv;
        let db = pb - pv;
        let projected = (da.dot(db) / (da.length() * db.length()))
            .clamp(-1.0, 1.0)
            .acos();
        assert_relative_eq!(projected, intrinsic, epsilon = 1e-2);
    }

    #[test]
    fn r4_tessellate_lerps_all_components() {
        let p0 = Vec4::new(0.0, 0.0, 0.0, 0.0);
        let p1 = Vec4::new(4.0, 8.0, 12.0, 16.0);
        let mut out = Vec::new();
        <EuclideanR4 as RasterizableSpace<4>>::tessellate_segment(p0, p1, 2, |p| out.push(p));
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], p0);
        assert_eq!(out[1], Vec4::new(2.0, 4.0, 6.0, 8.0));
        assert_eq!(out[2], p1);
    }
}
