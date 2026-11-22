use glam::{Vec3, Vec4};
use loam_math::{Bivector, Plane4, Rotor4};
use loam_shape::LineMesh;

/// A 16-cell cell centre, on no coordinate great circle: every ring is finite and congruent.
pub const POLE: Vec4 = Vec4::new(0.5, 0.5, 0.5, 0.5);

const MIN_PLANE_INCIDENCE: f32 = 1e-2;

const MIN_POLE_DISTANCE: f32 = 1e-4;

const IMAGE_AXES: [Vec4; 3] = [
    Vec4::new(0.5, 0.5, -0.5, -0.5),
    Vec4::new(0.5, -0.5, 0.5, -0.5),
    Vec4::new(0.5, -0.5, -0.5, 0.5),
];

fn to_r3(q: Vec4) -> Vec3 {
    Vec3::new(
        IMAGE_AXES[0].dot(q),
        IMAGE_AXES[1].dot(q),
        IMAGE_AXES[2].dot(q),
    )
}

fn plane_axes(plane: Plane4) -> (Vec4, Vec4) {
    match plane {
        Plane4::Xy => (Vec4::X, Vec4::Y),
        Plane4::Xz => (Vec4::X, Vec4::Z),
        Plane4::Xw => (Vec4::X, Vec4::W),
        Plane4::Yz => (Vec4::Y, Vec4::Z),
        Plane4::Yw => (Vec4::Y, Vec4::W),
        Plane4::Zw => (Vec4::Z, Vec4::W),
    }
}

fn wrap_pi(angle: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let wrapped = angle.rem_euclid(TAU);
    if wrapped > PI {
        wrapped - TAU
    } else {
        wrapped
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Hypergimbal {
    pub center: Vec3,
    pub scale: f32,
}

impl Hypergimbal {
    /// `None` within f32 noise of [`POLE`], where the image is unbounded.
    // Coxeter, Introduction to Geometry (1969), section 6.9.
    pub fn project(&self, p: Vec4) -> Option<Vec3> {
        let denom = 1.0 - p.dot(POLE);
        if denom.abs() < MIN_POLE_DISTANCE {
            return None;
        }
        Some(self.center + to_r3(p) / denom * self.scale)
    }

    /// All six rings, in [`Plane4::ALL`] order.
    pub fn rings(&self) -> [Ring; 6] {
        Plane4::ALL.map(|plane| self.ring(plane))
    }

    pub fn ring(&self, plane: Plane4) -> Ring {
        let (a, b) = plane_axes(plane);
        let alpha = a.dot(POLE);
        let beta = b.dot(POLE);
        let pole_overlap = (alpha * alpha + beta * beta).sqrt();
        let s = (1.0 - pole_overlap * pole_overlap).sqrt();
        let (a_phase, b_phase) = (
            (a * alpha + b * beta) / pole_overlap,
            (b * alpha - a * beta) / pole_overlap,
        );
        let u = to_r3((a_phase - POLE * pole_overlap) / s);
        Ring {
            plane,
            center: self.center + u * (self.scale * pole_overlap / s),
            u,
            v: to_r3(b_phase),
            radius: self.scale / s,
            pole_overlap,
            phase: beta.atan2(alpha),
        }
    }

    /// Nearest ring within `tolerance` of the plane hit; ties break by depth.
    pub fn pick(&self, ray_origin: Vec3, ray_direction: Vec3, tolerance: f32) -> Option<Ring> {
        let mut best: Option<(f32, Ring)> = None;
        for ring in self.rings() {
            let Some(hit) = ring.ray_hit(ray_origin, ray_direction) else {
                continue;
            };
            if ((hit - ring.center).length() - ring.radius).abs() > tolerance {
                continue;
            }
            let depth = (hit - ray_origin).length();
            if best.is_none_or(|(nearest, _)| depth < nearest) {
                best = Some((depth, ring));
            }
        }
        best.map(|(_, ring)| ring)
    }

    pub fn append_line_mesh(&self, style: &RingStyle, out: &mut LineMesh<3>) {
        let step = std::f32::consts::TAU / style.segments as f32;
        for ring in self.rings() {
            let color = style.colors[ring.plane as usize];
            let mut prev = ring.point(0.0);
            for k in 1..=style.segments {
                let next = ring.point(k as f32 * step);
                out.segments.push((prev.to_array(), next.to_array()));
                out.colors.push((color, color));
                out.widths.push(style.width_px);
                prev = next;
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Ring {
    pub plane: Plane4,
    pub center: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub radius: f32,
    pub pole_overlap: f32,
    pub phase: f32,
}

impl Ring {
    pub fn normal(&self) -> Vec3 {
        self.u.cross(self.v)
    }

    pub fn point(&self, chi: f32) -> Vec3 {
        self.center + (self.u * chi.cos() + self.v * chi.sin()) * self.radius
    }

    pub fn ring_angle(&self, world: Vec3) -> f32 {
        let offset = world - self.center;
        offset.dot(self.v).atan2(offset.dot(self.u))
    }

    // Danby, Fundamentals of Celestial Mechanics (1992), section 6.3.
    pub fn arc_angle(&self, chi: f32) -> f32 {
        let s = (1.0 - self.pole_overlap * self.pole_overlap).sqrt();
        (s * chi.sin()).atan2(chi.cos() + self.pole_overlap) + self.phase
    }

    pub fn ray_hit(&self, ray_origin: Vec3, ray_direction: Vec3) -> Option<Vec3> {
        let normal = self.normal();
        let incidence = ray_direction.dot(normal);
        if incidence.abs() < MIN_PLANE_INCIDENCE * ray_direction.length() {
            return None;
        }
        let t = (self.center - ray_origin).dot(normal) / incidence;
        (t > 0.0).then(|| ray_origin + ray_direction * t)
    }

    pub fn drag_angle(&self, grab: Vec3, cursor: Vec3) -> f32 {
        wrap_pi(self.arc_angle(self.ring_angle(cursor)) - self.arc_angle(self.ring_angle(grab)))
    }

    pub fn drag_rotor(&self, grab: Vec3, cursor: Vec3) -> Rotor4 {
        (self.plane.unit_bivector() * self.drag_angle(grab, cursor)).exp()
    }
}

#[derive(Clone, Debug)]
pub struct RingStyle {
    pub segments: usize,
    pub width_px: f32,
    /// RGBA per plane, in [`Plane4::ALL`] order.
    pub colors: [[f32; 4]; 6],
}

impl RingStyle {
    pub const PAIRED_HUE_COLORS: [[f32; 4]; 6] = [
        [0.95, 0.30, 0.32, 1.0],
        [0.35, 0.85, 0.40, 1.0],
        [0.30, 0.42, 0.75, 1.0],
        [0.38, 0.62, 0.98, 1.0],
        [0.24, 0.55, 0.30, 1.0],
        [0.70, 0.25, 0.28, 1.0],
    ];
}

impl Default for RingStyle {
    fn default() -> Self {
        Self {
            segments: 96,
            width_px: 2.5,
            colors: Self::PAIRED_HUE_COLORS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{PI, TAU};

    const WIDGET: Hypergimbal = Hypergimbal {
        center: Vec3::new(0.3, -0.7, 1.1),
        scale: 0.8,
    };

    fn great_circle_point(plane: Plane4, theta: f32) -> Vec4 {
        let (a, b) = plane_axes(plane);
        a * theta.cos() + b * theta.sin()
    }

    fn assert_close(actual: f32, expected: f32, tol: f32, what: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{what}: {actual} != {expected}"
        );
    }

    #[test]
    fn drag_yields_the_rotation_its_plane_predicts() {
        for plane in Plane4::ALL {
            let ring = WIDGET.ring(plane);
            for grab_step in 0..8 {
                let theta = -PI + grab_step as f32 / 8.0 * TAU;
                for delta in [-1.1_f32, -0.4, 0.05, 0.9, 2.2] {
                    let start = great_circle_point(plane, theta);
                    let end = great_circle_point(plane, theta + delta);
                    let grab = WIDGET.project(start).unwrap();
                    let cursor = WIDGET.project(end).unwrap();

                    assert_close(
                        wrap_pi(ring.drag_angle(grab, cursor) - delta),
                        0.0,
                        1e-3,
                        &format!("{plane:?} drag angle from θ={theta} by {delta}"),
                    );
                }
            }
        }
    }

    #[test]
    fn grazing_and_backward_rays_miss_the_ring_plane() {
        let ring = WIDGET.ring(Plane4::Xy);
        let eye = ring.center + ring.normal() * 3.0;
        assert!(ring.ray_hit(eye, -ring.normal()).is_some());
        assert!(
            ring.ray_hit(eye, ring.normal()).is_none(),
            "plane behind the eye must not hit"
        );
        assert!(
            ring.ray_hit(eye, ring.u).is_none(),
            "parallel ray must miss"
        );
        let grazing = (ring.u + ring.normal() * (MIN_PLANE_INCIDENCE * 0.5)).normalize();
        assert!(
            ring.ray_hit(eye, grazing).is_none(),
            "grazing ray must miss"
        );
    }
}
