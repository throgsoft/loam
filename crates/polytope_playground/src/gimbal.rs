use glam::Vec3;
use loam::math::{Plane4, Rotor, Rotor4};
use loam::render::hypergimbal::{Hypergimbal, RingDrag, RingStyle};
use loam::runtime::{ImageRay, SegmentRecord};
use loam::shape::LineMesh;

const SCALE: f32 = 0.55;

const PICK_TOLERANCE: f32 = 0.09;

const HIGHLIGHT: [f32; 4] = [1.0, 0.94, 0.55, 1.0];

const RING_WIDTH_PX: f32 = 1.8;

#[derive(Clone, Copy, Default)]
pub(crate) struct Gimbal {
    pub(crate) enabled: bool,
    drag: Option<RingDrag>,
    applied: Rotor4,
    hover: Option<Plane4>,
}

#[derive(Default)]
pub(crate) struct GimbalRenderer {
    mesh: LineMesh<3>,
}

pub(crate) fn widget(center: Vec3) -> Hypergimbal {
    Hypergimbal {
        center,
        scale: SCALE,
    }
}

impl Gimbal {
    pub(crate) fn press(&mut self, ray: &ImageRay, center: Vec3) -> bool {
        let (origin, direction) = (Vec3::from(ray.origin), Vec3::from(ray.direction));
        self.drag = widget(center)
            .pick(origin, direction, PICK_TOLERANCE)
            .and_then(|ring| RingDrag::press(ring, origin, direction));
        self.applied = Rotor4::IDENTITY;
        self.hover = self.drag.map(|drag| drag.ring.plane);
        self.drag.is_some()
    }

    pub(crate) fn turn(&mut self, ray: &ImageRay) -> Option<Rotor4> {
        let drag = self.drag?;
        let total = drag.rotor(Vec3::from(ray.origin), Vec3::from(ray.direction))?;
        let step = total * self.applied.inverse();
        self.applied = total;
        Some(step.normalize())
    }

    pub(crate) fn release(&mut self) {
        self.drag = None;
        self.applied = Rotor4::IDENTITY;
    }

    pub(crate) fn held(&self) -> bool {
        self.drag.is_some()
    }

    pub(crate) fn aim(&mut self, ray: Option<&ImageRay>, center: Vec3) {
        if self.drag.is_some() {
            return;
        }
        self.hover = ray.and_then(|ray| {
            widget(center)
                .pick(
                    Vec3::from(ray.origin),
                    Vec3::from(ray.direction),
                    PICK_TOLERANCE,
                )
                .map(|ring| ring.plane)
        });
    }
}

impl GimbalRenderer {
    pub(crate) fn rings(&mut self, gimbal: &Gimbal, center: Vec3, out: &mut Vec<SegmentRecord>) {
        if !gimbal.enabled {
            return;
        }
        let mut style = RingStyle::default();
        if let Some(plane) = gimbal.hover {
            style.colors[plane as usize] = HIGHLIGHT;
        }
        style.width_px = RING_WIDTH_PX;
        let mesh = &mut self.mesh;
        mesh.segments.clear();
        mesh.colors.clear();
        mesh.widths.clear();
        widget(center).append_line_mesh(&style, mesh);
        for ((start, end), ((from, to), width)) in mesh
            .segments
            .iter()
            .zip(mesh.colors.iter().zip(mesh.widths.iter()))
        {
            out.push(SegmentRecord {
                start: *start,
                _pad0: 0.0,
                end: *end,
                _pad1: 0.0,
                start_color: *from,
                end_color: *to,
                width_px: *width,
                _pad2: [0.0; 3],
            });
        }
    }
}
