use glam::Vec4;
use loam_math::{Rotor, Rotor4};

use super::simplex_r4::{closest_to_origin, project_origin_onto_affine_hull};

pub trait SupportFn4 {
    fn support(&self, direction: Vec4) -> Vec4;
}

pub struct ConvexHull4<'a> {
    pub vertices: &'a [Vec4],
}

impl<'a> SupportFn4 for ConvexHull4<'a> {
    fn support(&self, direction: Vec4) -> Vec4 {
        let mut best = self.vertices[0];
        let mut best_d = best.dot(direction);
        for &v in &self.vertices[1..] {
            let d = v.dot(direction);
            if d > best_d {
                best_d = d;
                best = v;
            }
        }
        best
    }
}

/// Support queries transform the direction into the body frame.
pub struct PosedHull4<'a> {
    pub local: &'a [Vec4],
    pub position: Vec4,
    pub rotation: Rotor4,
}

impl SupportFn4 for PosedHull4<'_> {
    fn support(&self, direction: Vec4) -> Vec4 {
        let local_dir = self.rotation.inverse().apply(direction);
        let mut best = self.local[0];
        let mut best_d = best.dot(local_dir);
        for &v in &self.local[1..] {
            let d = v.dot(local_dir);
            if d > best_d {
                best_d = d;
                best = v;
            }
        }
        self.rotation.apply(best) + self.position
    }
}

pub struct Sphere4 {
    pub center: Vec4,
    pub radius: f32,
}

impl SupportFn4 for Sphere4 {
    fn support(&self, direction: Vec4) -> Vec4 {
        let d = direction.length_squared();
        let dir = if d > 1e-12 {
            direction / d.sqrt()
        } else {
            Vec4::Y
        };
        self.center + dir * self.radius
    }
}

/// `sa` and `sb` are the contributing support points on A and B.
#[derive(Clone, Copy, Debug)]
pub struct MinkowskiPoint4 {
    pub point: Vec4,
    pub sa: Vec4,
    pub sb: Vec4,
}

pub fn minkowski_support_r4<A: SupportFn4, B: SupportFn4>(
    a: &A,
    b: &B,
    direction: Vec4,
) -> MinkowskiPoint4 {
    let sa = a.support(direction);
    let sb = b.support(-direction);
    MinkowskiPoint4 {
        point: sa - sb,
        sa,
        sb,
    }
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum GjkResult4 {
    Intersecting { simplex: [MinkowskiPoint4; 5] },
    Separated,
}

const GJK_MAX_ITERATIONS: u32 = 48;
const GJK_EPS: f32 = 1e-6;

/// On overlap the returned 4-simplex is the seed EPA expects.
pub fn gjk_intersect_r4<A: SupportFn4, B: SupportFn4>(
    a: &A,
    b: &B,
    initial_direction: Vec4,
) -> GjkResult4 {
    let mut dir = if initial_direction.length_squared() > GJK_EPS {
        initial_direction
    } else {
        Vec4::X
    };
    let mut simplex = [minkowski_support_r4(a, b, dir); 5];
    let mut len = 1;
    dir = -simplex[0].point;

    for _ in 0..GJK_MAX_ITERATIONS {
        if dir.length_squared() < GJK_EPS {
            if let Some(seed) = complete_simplex(a, b, simplex, len) {
                return GjkResult4::Intersecting { simplex: seed };
            }
        }
        let direction = dir.try_normalize().unwrap_or(Vec4::X);
        let new_point = minkowski_support_r4(a, b, direction);
        if new_point.point.dot(direction) < 0.0 {
            return GjkResult4::Separated;
        }
        if simplex[..len].iter().any(|p| p.point == new_point.point) {
            return GjkResult4::Separated;
        }
        simplex[len] = new_point;
        len += 1;
        let points = simplex.map(|p| p.point);
        let closest = closest_to_origin(&points[..len]);
        let previous = simplex;
        for (slot, &i) in closest.kept().iter().enumerate() {
            simplex[slot] = previous[i];
        }
        len = closest.kept().len();
        if len == 5 {
            return GjkResult4::Intersecting { simplex };
        }
        dir = -closest.point;
    }
    GjkResult4::Separated
}

fn complete_simplex<A: SupportFn4, B: SupportFn4>(
    a: &A,
    b: &B,
    mut simplex: [MinkowskiPoint4; 5],
    len: usize,
) -> Option<[MinkowskiPoint4; 5]> {
    if len == 5 {
        let points = simplex.map(|p| p.point);
        let (_, weights) = project_origin_onto_affine_hull(&[0, 1, 2, 3, 4], &points)?;
        return weights
            .iter()
            .all(|&w| w.is_finite() && w >= 0.0)
            .then_some(simplex);
    }
    let probe = orthogonal_to_hull(&simplex[..len])?;
    for direction in [probe, -probe] {
        let support = minkowski_support_r4(a, b, direction);
        if simplex[..len].iter().any(|p| p.point == support.point) {
            continue;
        }
        simplex[len] = support;
        if let Some(seed) = complete_simplex(a, b, simplex, len + 1) {
            return Some(seed);
        }
    }
    None
}

fn orthogonal_to_hull(simplex: &[MinkowskiPoint4]) -> Option<Vec4> {
    let mut onb = [Vec4::ZERO; 4];
    let mut rank = 0;
    for p in &simplex[1..] {
        let mut r = p.point - simplex[0].point;
        for o in &onb[..rank] {
            r -= *o * r.dot(*o);
        }
        let m = r.length_squared();
        if m > 1e-10 {
            onb[rank] = r / m.sqrt();
            rank += 1;
        }
    }
    let mut best: Option<(f32, Vec4)> = None;
    for axis in [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W] {
        let mut r = axis;
        for o in &onb[..rank] {
            r -= *o * r.dot(*o);
        }
        let mag_sq = r.length_squared();
        if mag_sq > 1e-8 && best.is_none_or(|(m, _)| mag_sq > m) {
            best = Some((mag_sq, r));
        }
    }
    best.map(|(_, v)| v.normalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::{Bivector, Bivector4, Plane4};

    #[test]
    fn a_posed_hull_supports_where_the_world_vertices_would() {
        let local = [
            Vec4::new(0.6, 0.1, -0.2, 0.35),
            Vec4::new(-0.4, 0.7, 0.15, -0.5),
            Vec4::new(0.05, -0.8, 0.45, 0.2),
            Vec4::new(-0.3, 0.2, -0.65, -0.1),
            Vec4::new(0.25, 0.4, 0.5, 0.6),
        ];
        let position = Vec4::new(3.0, -1.5, 0.75, -2.25);
        let simple = (Plane4::Xw.unit_bivector() * 0.8).exp().normalize();
        let double = (Bivector4::new(0.5, 0.0, 0.0, 0.0, 0.0, -0.9).exp()).normalize();

        for rotation in [Rotor4::identity(), simple, double] {
            let world: Vec<Vec4> = local
                .iter()
                .map(|v| rotation.apply(*v) + position)
                .collect();
            let posed = PosedHull4 {
                local: &local,
                position,
                rotation,
            };
            let materialised = ConvexHull4 { vertices: &world };
            for dir in [
                Vec4::X,
                Vec4::W,
                Vec4::new(1.0, 1.0, 1.0, 1.0),
                Vec4::new(-0.3, 0.9, -0.2, 0.7),
                Vec4::new(0.0, -1.0, 0.4, -0.4),
            ] {
                let (a, b) = (posed.support(dir), materialised.support(dir));
                assert!(
                    (a - b).length() < 1e-5,
                    "posed support {a:?} against materialised {b:?} for {dir:?}"
                );
            }
        }
    }

    #[test]
    fn a_small_positive_gap_is_not_an_intersection() {
        for gap in [1e-5, 1e-4, 5e-4, 1e-3] {
            let a = Sphere4 {
                center: Vec4::ZERO,
                radius: 1.0,
            };
            let b = Sphere4 {
                center: Vec4::X * (2.0 + gap),
                radius: 1.0,
            };
            for direction in [Vec4::X, -Vec4::X, Vec4::Y] {
                assert!(
                    matches!(gjk_intersect_r4(&a, &b, direction), GjkResult4::Separated),
                    "gap={gap}, direction={direction:?}"
                );
            }
        }
    }

    #[test]
    fn rotated_hulls_with_disjoint_x_intervals_are_separated() {
        use crate::euclidean_r4::tesseract_vertices;
        let rotor = Bivector4::new(0.37, -0.21, 0.43, 0.17, -0.29, 0.13).exp();
        for scale in [0.1, 1.0, 100.0] {
            let a_vertices = tesseract_vertices(scale);
            let rotated: Vec<_> = a_vertices.iter().map(|&v| rotor.apply(v)).collect();
            let right_a = a_vertices
                .iter()
                .map(|v| v.x)
                .fold(f32::NEG_INFINITY, f32::max);
            let left_b = rotated.iter().map(|v| v.x).fold(f32::INFINITY, f32::min);
            for relative_gap in [1e-5, 1e-3, 0.1] {
                let translation = Vec4::new(
                    right_a - left_b + relative_gap * scale,
                    0.2 * scale,
                    -0.15 * scale,
                    0.1 * scale,
                );
                let b_vertices: Vec<_> = rotated.iter().map(|&v| v + translation).collect();
                assert!(b_vertices.iter().all(|v| v.x > right_a));
                let a = ConvexHull4 {
                    vertices: &a_vertices,
                };
                let b = ConvexHull4 {
                    vertices: &b_vertices,
                };
                for direction in [Vec4::Y, Vec4::Z, Vec4::new(0.31, -0.7, 0.5, 0.2)] {
                    assert!(
                        matches!(gjk_intersect_r4(&a, &b, direction), GjkResult4::Separated),
                        "scale={scale}, gap={relative_gap}, direction={direction:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn separated_spheres() {
        let a = Sphere4 {
            center: Vec4::new(-5.0, 0.0, 0.0, 0.0),
            radius: 1.0,
        };
        let b = Sphere4 {
            center: Vec4::new(5.0, 0.0, 0.0, 0.0),
            radius: 1.0,
        };
        match gjk_intersect_r4(&a, &b, Vec4::X) {
            GjkResult4::Separated => {}
            _ => panic!("expected Separated"),
        }
    }

    #[test]
    fn overlapping_spheres_complete_lower_dimensional_searches() {
        for center in [
            Vec4::X,
            Vec4::new(1.1, 0.7, 0.0, 0.0),
            Vec4::new(1.1, 0.7, 0.3, 0.2),
        ] {
            let a = Sphere4 {
                center: Vec4::ZERO,
                radius: 2.0,
            };
            let b = Sphere4 {
                center,
                radius: 2.0,
            };
            assert!(center.length() < a.radius + b.radius);
            for direction in [Vec4::X, Vec4::Y, Vec4::W] {
                assert!(
                    matches!(
                        gjk_intersect_r4(&a, &b, direction),
                        GjkResult4::Intersecting { .. }
                    ),
                    "center={center:?}, direction={direction:?}"
                );
            }
        }
    }

    #[test]
    fn tesseracts_overlap_past_touching() {
        use crate::euclidean_r4::tesseract_vertices;
        let va: Vec<Vec4> = tesseract_vertices(1.0);
        let vb: Vec<Vec4> = tesseract_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.6, 0.6, 0.6, 0.6))
            .collect();
        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        assert!(matches!(
            gjk_intersect_r4(&a, &b, Vec4::X),
            GjkResult4::Intersecting { .. }
        ));
    }

    #[test]
    fn deeply_overlapping_pentatopes() {
        use crate::euclidean_r4::pentatope_vertices;
        let va: Vec<Vec4> = pentatope_vertices(1.0);
        let vb: Vec<Vec4> = pentatope_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(0.2, 0.0, 0.0, 0.0))
            .collect();
        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        assert!(matches!(
            gjk_intersect_r4(&a, &b, Vec4::X),
            GjkResult4::Intersecting { .. }
        ));
    }

    #[test]
    fn fully_separated_pentatopes() {
        use crate::euclidean_r4::pentatope_vertices;
        let va: Vec<Vec4> = pentatope_vertices(1.0);
        let vb: Vec<Vec4> = pentatope_vertices(1.0)
            .into_iter()
            .map(|v| v + Vec4::new(10.0, 0.0, 0.0, 0.0))
            .collect();
        let a = ConvexHull4 { vertices: &va };
        let b = ConvexHull4 { vertices: &vb };
        assert!(matches!(
            gjk_intersect_r4(&a, &b, Vec4::X),
            GjkResult4::Separated
        ));
    }

    #[test]
    fn tesseract_face_touching_is_bracketed_and_never_resolves_to_a_depth() {
        use crate::collision::epa_r4::epa_r4;
        use crate::euclidean_r4::tesseract_vertices;

        let va: Vec<Vec4> = tesseract_vertices(1.0);
        let a = ConvexHull4 { vertices: &va };
        for (shift, expect_overlap) in [
            (1.0 - 1e-1, true),
            (1.0 - 1e-2, true),
            (1.0, false),
            (1.0 + 1e-1, false),
            (1.0 + 1.0, false),
        ] {
            let vb: Vec<Vec4> = tesseract_vertices(1.0)
                .into_iter()
                .map(|v| v + Vec4::new(shift, 0.0, 0.0, 0.0))
                .collect();
            let b = ConvexHull4 { vertices: &vb };
            let depth = match gjk_intersect_r4(&a, &b, Vec4::X) {
                GjkResult4::Intersecting { simplex } => {
                    epa_r4(&a, &b, simplex, 2.0).map_or(0.0, |c| c.penetration)
                }
                GjkResult4::Separated => 0.0,
            };
            if expect_overlap {
                assert!(
                    (depth - (1.0 - shift)).abs() < 1e-2,
                    "shift {shift} overlaps by {} but resolved to {depth}",
                    1.0 - shift
                );
            } else {
                assert_eq!(depth, 0.0, "shift {shift} is clear but resolved to {depth}");
            }
        }
    }

    #[test]
    fn sphere_and_tesseract_inside() {
        use crate::euclidean_r4::tesseract_vertices;
        let sphere = Sphere4 {
            center: Vec4::ZERO,
            radius: 0.1,
        };
        let vs: Vec<Vec4> = tesseract_vertices(1.0);
        let tess = ConvexHull4 { vertices: &vs };
        assert!(matches!(
            gjk_intersect_r4(&sphere, &tess, Vec4::X),
            GjkResult4::Intersecting { .. }
        ));
    }
}
