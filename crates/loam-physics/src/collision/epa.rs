use glam::Vec3;

use super::gjk::{minkowski_support, MinkowskiPoint, SupportFn};

const EPA_MAX_ITERATIONS: u32 = 48;

const EPA_TOLERANCE: f32 = 1e-4;

const FACE_COPLANAR_EPS: f32 = 1e-5;

const FACE_DEGENERATE_CROSS: f32 = 1e-8;

const SEED_DEGENERATE_VOLUME: f32 = 1e-8;

const BARYCENTRIC_SINGULAR_EPS: f32 = 1e-12;

#[derive(Clone, Copy)]
struct Thresholds {
    support_gap: f32,
    coplanar_band: f32,
    cross_norm: f32,
    barycentric_gram: f32,
}

impl Thresholds {
    fn for_scale(scale: f32) -> Self {
        Self {
            support_gap: EPA_TOLERANCE * scale,
            coplanar_band: FACE_COPLANAR_EPS * scale,
            cross_norm: FACE_DEGENERATE_CROSS * scale * scale,
            barycentric_gram: BARYCENTRIC_SINGULAR_EPS * scale.powi(4),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ContactInfo {
    /// Unit, world, from A toward B.
    pub normal: Vec3,
    pub penetration: f32,
    /// Midpoint of the closest features on A and B, in world space.
    pub point: Vec3,
}

#[derive(Clone, Copy, Debug)]
struct Face {
    v: [usize; 3],
    normal: Vec3,
    distance: f32,
}

struct Polytope {
    vertices: Vec<MinkowskiPoint>,
    faces: Vec<Face>,
    horizon: Vec<(usize, usize)>,
    interior: glam::Vec3,
    thresholds: Thresholds,
}

impl Polytope {
    fn from_tetra(tetra: [MinkowskiPoint; 4], thresholds: Thresholds) -> Self {
        let vertices = tetra.to_vec();
        let interior = (tetra[0].point + tetra[1].point + tetra[2].point + tetra[3].point) * 0.25;
        let mut faces = Vec::with_capacity(4);
        for &(i, j, k, l) in &[(0, 1, 2, 3), (0, 3, 1, 2), (0, 2, 3, 1), (1, 3, 2, 0)] {
            faces.push(build_face_vs_point(
                &vertices,
                i,
                j,
                k,
                vertices[l].point,
                thresholds.cross_norm,
            ));
        }
        Self {
            vertices,
            faces,
            horizon: Vec::new(),
            interior,
            thresholds,
        }
    }

    fn closest_face(&self) -> Option<usize> {
        let (idx, _) = self
            .faces
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.distance.total_cmp(&b.1.distance))?;
        Some(idx)
    }

    fn expand(&mut self, support: MinkowskiPoint) {
        let new_idx = self.vertices.len();
        self.vertices.push(support);

        self.horizon.clear();
        let horizon = &mut self.horizon;
        let vertices = &self.vertices;

        let coplanar_band = self.thresholds.coplanar_band;
        self.faces.retain(|f| {
            let view = support.point - vertices[f.v[0]].point;
            if f.normal.dot(view) > -coplanar_band {
                add_or_remove_edge(horizon, f.v[0], f.v[1]);
                add_or_remove_edge(horizon, f.v[1], f.v[2]);
                add_or_remove_edge(horizon, f.v[2], f.v[0]);
                false
            } else {
                true
            }
        });

        let interior = self.interior;
        let cross_norm = self.thresholds.cross_norm;
        for &(i, j) in horizon.iter() {
            self.faces.push(build_face_vs_point(
                &self.vertices,
                i,
                j,
                new_idx,
                interior,
                cross_norm,
            ));
        }
    }
}

fn build_face_vs_point(
    verts: &[MinkowskiPoint],
    a: usize,
    b: usize,
    c: usize,
    interior_point: Vec3,
    cross_norm_floor: f32,
) -> Face {
    let pa = verts[a].point;
    let pb = verts[b].point;
    let pc = verts[c].point;

    let mut normal = (pb - pa).cross(pc - pa);
    let len = normal.length();
    if len < cross_norm_floor {
        normal = Vec3::Y;
    } else {
        normal /= len;
    }

    let to_interior = interior_point - pa;
    let (v_order, outward_normal) = if normal.dot(to_interior) > 0.0 {
        ([a, c, b], -normal)
    } else {
        ([a, b, c], normal)
    };

    let raw_distance = outward_normal.dot(verts[v_order[0]].point);
    let distance = raw_distance.max(0.0);
    Face {
        v: v_order,
        normal: outward_normal,
        distance,
    }
}

fn add_or_remove_edge(horizon: &mut Vec<(usize, usize)>, a: usize, b: usize) {
    if let Some(pos) = horizon.iter().position(|&e| e == (b, a)) {
        horizon.swap_remove(pos);
    } else {
        horizon.push((a, b));
    }
}

fn barycentric(a: Vec3, b: Vec3, c: Vec3, p: Vec3, gram_floor: f32) -> (f32, f32, f32) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < gram_floor {
        return (1.0, 0.0, 0.0);
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    (u, v, w)
}

/// `scale` is the sum of the two bounding radii.
pub fn epa<A: SupportFn, B: SupportFn>(
    a: &A,
    b: &B,
    initial_simplex: [MinkowskiPoint; 4],
    scale: f32,
) -> Option<ContactInfo> {
    let (polytope, face) = expand_to_boundary(a, b, initial_simplex, scale)?;
    Some(contact_from_face(&polytope, face))
}

fn expand_to_boundary<A: SupportFn, B: SupportFn>(
    a: &A,
    b: &B,
    initial_simplex: [MinkowskiPoint; 4],
    scale: f32,
) -> Option<(Polytope, Face)> {
    let thresholds = Thresholds::for_scale(scale);

    let p0 = initial_simplex[0].point;
    let p1 = initial_simplex[1].point;
    let p2 = initial_simplex[2].point;
    let p3 = initial_simplex[3].point;
    let volume6 = (p1 - p0).dot((p2 - p0).cross(p3 - p0)).abs();
    if volume6 < SEED_DEGENERATE_VOLUME * scale.powi(3) {
        return None;
    }

    let mut polytope = Polytope::from_tetra(initial_simplex, thresholds);

    for _ in 0..EPA_MAX_ITERATIONS {
        let face_idx = polytope.closest_face()?;
        let face = polytope.faces[face_idx];

        let support = minkowski_support(a, b, face.normal);
        let new_distance = support.point.dot(face.normal);

        if !new_distance.is_finite() || !support.point.is_finite() {
            return None;
        }

        if (new_distance - face.distance).abs() < thresholds.support_gap {
            return Some((polytope, face));
        }

        polytope.expand(support);
    }

    tracing::debug!(
        max_iterations = EPA_MAX_ITERATIONS,
        vertices = polytope.vertices.len(),
        "EPA 3D hit iteration cap; returning best-estimate contact",
    );
    let face_idx = polytope.closest_face()?;
    let face = polytope.faces[face_idx];
    Some((polytope, face))
}

fn contact_from_face(polytope: &Polytope, face: Face) -> ContactInfo {
    let v0 = polytope.vertices[face.v[0]];
    let v1 = polytope.vertices[face.v[1]];
    let v2 = polytope.vertices[face.v[2]];

    let closest = face.normal * face.distance;

    let (u, v, w) = barycentric(
        v0.point,
        v1.point,
        v2.point,
        closest,
        polytope.thresholds.barycentric_gram,
    );

    let point_a = v0.sa * u + v1.sa * v + v2.sa * w;
    let point_b = v0.sb * u + v1.sb * v + v2.sb * w;

    ContactInfo {
        normal: face.normal,
        penetration: face.distance,
        point: (point_a + point_b) * 0.5,
    }
}
