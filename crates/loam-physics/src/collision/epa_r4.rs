use glam::Vec4;

use super::gjk_r4::{minkowski_support_r4, MinkowskiPoint4, SupportFn4};
use super::simplex_r4::project_origin_onto_affine_hull;

const EPA_MAX_ITERATIONS: u32 = 96;

const EPA_TOLERANCE: f32 = 1e-3;

#[derive(Clone, Copy, Debug)]
pub struct ContactInfo4 {
    pub normal: Vec4,
    pub penetration: f32,
    pub point: Vec4,
}

#[derive(Clone, Copy, Debug)]
struct Face4 {
    v: [usize; 4],
    normal: Vec4,
    distance: f32,
}

#[derive(Clone, Copy)]
struct Thresholds {
    support_gap: f32,
    coplanar_band: f32,
    wedge_norm: f32,
}

impl Thresholds {
    fn for_scale(scale: f32) -> Self {
        Self {
            support_gap: EPA_TOLERANCE * scale,
            coplanar_band: FACE_COPLANAR_EPS * scale,
            wedge_norm: FACE_DEGENERATE_WEDGE * scale.powi(3),
        }
    }
}

struct Polytope4 {
    vertices: Vec<MinkowskiPoint4>,
    faces: Vec<Face4>,
    horizon: Vec<Triangle>,
    centroid: Vec4,
    thresholds: Thresholds,
}

impl Polytope4 {
    fn from_simplex(simplex: [MinkowskiPoint4; 5], thresholds: Thresholds) -> Self {
        let vertices = simplex.to_vec();
        let centroid = (simplex[0].point
            + simplex[1].point
            + simplex[2].point
            + simplex[3].point
            + simplex[4].point)
            * 0.2;

        let mut faces = Vec::with_capacity(5);
        for l in 0..5 {
            let mut tet = [0usize; 4];
            let mut idx = 0;
            for i in 0..5 {
                if i != l {
                    tet[idx] = i;
                    idx += 1;
                }
            }
            if let Some(face) = build_face(
                &vertices,
                tet[0],
                tet[1],
                tet[2],
                tet[3],
                centroid,
                thresholds.wedge_norm,
            ) {
                faces.push(face);
            }
        }
        Self {
            vertices,
            faces,
            horizon: Vec::new(),
            centroid,
            thresholds,
        }
    }

    // Zero-distance faces must remain candidates for the near-side normal.
    fn closest_face(&self) -> Option<usize> {
        self.faces
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.distance.total_cmp(&b.1.distance))
            .map(|(idx, _)| idx)
    }

    fn expand(&mut self, support: MinkowskiPoint4) {
        let new_idx = self.vertices.len();
        self.vertices.push(support);

        self.horizon.clear();
        let horizon = &mut self.horizon;
        let vertices = &self.vertices;

        let coplanar_band = self.thresholds.coplanar_band;
        self.faces.retain(|f| {
            let view = support.point - vertices[f.v[0]].point;
            if f.normal.dot(view) > -coplanar_band {
                for tri in tet_triangles(&f.v) {
                    add_or_remove_triangle(horizon, tri);
                }
                false
            } else {
                true
            }
        });

        let centroid = self.centroid;
        let wedge_norm = self.thresholds.wedge_norm;
        for tri in horizon.iter() {
            if let Some(face) = build_face(
                &self.vertices,
                tri.0,
                tri.1,
                tri.2,
                new_idx,
                centroid,
                wedge_norm,
            ) {
                self.faces.push(face);
            }
        }
    }
}

const FACE_COPLANAR_EPS: f32 = 1e-5;

const FACE_DEGENERATE_WEDGE: f32 = 1e-8;

const SEED_DEGENERATE_VOLUME: f32 = 1e-8;

type Triangle = (usize, usize, usize);

fn tet_triangles(tet: &[usize; 4]) -> [Triangle; 4] {
    let (a, b, c, d) = (tet[0], tet[1], tet[2], tet[3]);
    [(a, b, c), (a, b, d), (a, c, d), (b, c, d)]
}

fn add_or_remove_triangle(horizon: &mut Vec<Triangle>, tri: Triangle) {
    let key = sort_triangle(tri);
    if let Some(pos) = horizon.iter().position(|t| sort_triangle(*t) == key) {
        horizon.swap_remove(pos);
    } else {
        horizon.push(tri);
    }
}

fn sort_triangle(t: Triangle) -> (usize, usize, usize) {
    let mut a = [t.0, t.1, t.2];
    a.sort_unstable();
    (a[0], a[1], a[2])
}

// Orient against the centroid: GJK's tolerance can leave the origin on the wrong side of a face.
fn build_face(
    verts: &[MinkowskiPoint4],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    centroid: Vec4,
    wedge_norm_floor: f32,
) -> Option<Face4> {
    let pa = verts[a].point;
    let pb = verts[b].point;
    let pc = verts[c].point;
    let pd = verts[d].point;

    let raw_normal = hodge_dual_of_trivector_wedge(pb - pa, pc - pa, pd - pa);
    let len = raw_normal.length();
    if len < wedge_norm_floor {
        return None;
    }
    let normal = raw_normal / len;

    let flip = normal.dot(centroid - pa) > 0.0;

    let (outward, v_order) = if flip {
        (-normal, [a, b, d, c])
    } else {
        (normal, [a, b, c, d])
    };

    let distance = outward.dot(pa).max(0.0);

    Some(Face4 {
        v: v_order,
        normal: outward,
        distance,
    })
}

fn hodge_dual_of_trivector_wedge(u: Vec4, v: Vec4, w: Vec4) -> Vec4 {
    let t_234 = det3(u.y, u.z, u.w, v.y, v.z, v.w, w.y, w.z, w.w);
    let t_134 = det3(u.x, u.z, u.w, v.x, v.z, v.w, w.x, w.z, w.w);
    let t_124 = det3(u.x, u.y, u.w, v.x, v.y, v.w, w.x, w.y, w.w);
    let t_123 = det3(u.x, u.y, u.z, v.x, v.y, v.z, w.x, w.y, w.z);

    Vec4::new(t_234, -t_134, t_124, -t_123)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn det3(
    a00: f32,
    a01: f32,
    a02: f32,
    a10: f32,
    a11: f32,
    a12: f32,
    a20: f32,
    a21: f32,
    a22: f32,
) -> f32 {
    a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
}

/// `scale` is the sum of the two bounding radii.
pub fn epa_r4<A: SupportFn4, B: SupportFn4>(
    a: &A,
    b: &B,
    initial_simplex: [MinkowskiPoint4; 5],
    scale: f32,
) -> Option<ContactInfo4> {
    let thresholds = Thresholds::for_scale(scale);

    let p0 = initial_simplex[0].point;
    let d1 = initial_simplex[1].point - p0;
    let d2 = initial_simplex[2].point - p0;
    let d3 = initial_simplex[3].point - p0;
    let d4 = initial_simplex[4].point - p0;
    let volume = det4(d1, d2, d3, d4).abs();
    if volume < SEED_DEGENERATE_VOLUME * scale.powi(4) {
        return None;
    }

    let mut polytope = Polytope4::from_simplex(initial_simplex, thresholds);

    for _ in 0..EPA_MAX_ITERATIONS {
        let face_idx = polytope.closest_face()?;
        let face = polytope.faces[face_idx];

        let support = minkowski_support_r4(a, b, face.normal);
        let new_distance = support.point.dot(face.normal);

        if !new_distance.is_finite() || !support.point.is_finite() {
            return None;
        }

        if (new_distance - face.distance).abs() < thresholds.support_gap {
            return Some(contact_from_face(&polytope, face));
        }

        polytope.expand(support);
    }

    tracing::debug!(
        max_iterations = EPA_MAX_ITERATIONS,
        vertices = polytope.vertices.len(),
        "EPA 4D hit iteration cap; returning best-estimate contact",
    );
    let face_idx = polytope.closest_face()?;
    Some(contact_from_face(&polytope, polytope.faces[face_idx]))
}

fn det4(r0: Vec4, r1: Vec4, r2: Vec4, r3: Vec4) -> f32 {
    r0.x * det3(r1.y, r1.z, r1.w, r2.y, r2.z, r2.w, r3.y, r3.z, r3.w)
        - r0.y * det3(r1.x, r1.z, r1.w, r2.x, r2.z, r2.w, r3.x, r3.z, r3.w)
        + r0.z * det3(r1.x, r1.y, r1.w, r2.x, r2.y, r2.w, r3.x, r3.y, r3.w)
        - r0.w * det3(r1.x, r1.y, r1.z, r2.x, r2.y, r2.z, r3.x, r3.y, r3.z)
}

fn contact_from_face(polytope: &Polytope4, face: Face4) -> ContactInfo4 {
    let tetra = face.v.map(|i| polytope.vertices[i]);

    let closest = face.normal * face.distance;
    let weights = face_barycentrics(&tetra.map(|p| p.point), closest);

    let mut point_a = Vec4::ZERO;
    let mut point_b = Vec4::ZERO;
    for (vertex, w) in tetra.iter().zip(weights) {
        point_a += vertex.sa * w;
        point_b += vertex.sb * w;
    }

    ContactInfo4 {
        normal: face.normal,
        penetration: face.distance,
        point: (point_a + point_b) * 0.5,
    }
}

fn face_barycentrics(points: &[Vec4; 4], closest: Vec4) -> [f32; 4] {
    let shifted = points.map(|p| p - closest);
    match project_origin_onto_affine_hull(&[0, 1, 2, 3], &shifted) {
        Some((_, w)) => [w[0], w[1], w[2], w[3]],
        None => [0.25; 4],
    }
}

#[cfg(test)]
mod tests {
    use super::super::gjk_r4::{gjk_intersect_r4, ConvexHull4, GjkResult4};
    use super::*;

    #[test]
    fn epa_r4_contact_is_equivariant_under_uniform_scaling() {
        use loam_shape::polytope_geom::{
            cell120_vertices, cell16_vertices, cell24_vertices, cell600_vertices,
            pentatope_vertices, tesseract_vertices,
        };

        let root5 = 5.0_f32.sqrt();
        let root6 = 6.0_f32.sqrt();
        let phi = (1.0 + root5) * 0.5;
        let inradius600 = phi * phi / (2.0 * 2.0_f32.sqrt());

        type ScaleFixture = (&'static str, fn(f32) -> Vec<Vec4>, f32, f32);
        let fixtures: [ScaleFixture; 6] = [
            (
                "pentatope",
                pentatope_vertices,
                5.0 / (2.0 * root6) - 0.3 * root5 / root6,
                root5 / root6,
            ),
            ("tesseract", tesseract_vertices, 1.0 - 0.3, 1.0),
            ("16-cell", cell16_vertices, 1.0 - 0.15, 0.5),
            ("24-cell", cell24_vertices, 2.0_f32.sqrt() - 0.3, 1.0),
            (
                "600-cell",
                cell600_vertices,
                inradius600 * (2.0 - 0.3),
                inradius600,
            ),
            (
                "120-cell",
                cell120_vertices,
                phi * phi / 2.0_f32.sqrt() - 0.3,
                1.0,
            ),
        ];

        for (name, vertices_of, unit_depth, unit_normal_x) in fixtures {
            for s in [1e-3_f32, 1e-2, 1.0, 1e2, 1e3] {
                let va = vertices_of(s);
                let vb: Vec<Vec4> = vertices_of(s)
                    .into_iter()
                    .map(|v| v + Vec4::new(0.3 * s, 0.0, 0.0, 0.0))
                    .collect();
                let a = ConvexHull4 { vertices: &va };
                let b = ConvexHull4 { vertices: &vb };
                let simplex = match gjk_intersect_r4(&a, &b, Vec4::X) {
                    GjkResult4::Intersecting { simplex } => simplex,
                    GjkResult4::Separated => panic!("{name} at scale {s} should overlap"),
                };

                let scale = 2.0 * s;
                let contact = epa_r4(&a, &b, simplex, scale)
                    .unwrap_or_else(|| panic!("{name} at scale {s} resolved to no contact"));

                let want_depth = unit_depth * s;
                assert!(
                    (contact.penetration - want_depth).abs() <= EPA_TOLERANCE * scale,
                    "{name} at scale {s}: depth {} is not {want_depth}",
                    contact.penetration
                );
                assert!(
                    (contact.normal.x - unit_normal_x).abs() <= 1e-3,
                    "{name} at scale {s}: normal {:?} has x != {unit_normal_x}",
                    contact.normal
                );
                assert!(
                    (contact.normal.length_squared() - 1.0).abs() < 1e-3,
                    "{name} at scale {s}: normal {:?} is not unit",
                    contact.normal
                );
            }
        }
    }
}
