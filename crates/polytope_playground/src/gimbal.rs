use glam::Vec3;
use loam_math::{Rotor, Rotor4};
use loam_render::gizmo::{Handle, HandleDrag, HandleId};
use loam_render::hypergimbal::{Hypergimbal, RingStyle};
use loam_runtime::{ImageRay, SegmentRecord};
use loam_shape::LineMesh;

const SCALE: f32 = 0.55;

const PICK_TOLERANCE: f32 = 0.09;

const HIGHLIGHT: [f32; 4] = [1.0, 0.94, 0.55, 1.0];

const RING_WIDTH_PX: f32 = 1.8;

#[derive(Default)]
pub(crate) struct Gimbal {
    pub(crate) enabled: bool,
    drag: Option<HandleDrag>,
    applied: Rotor4,
    hover: Option<HandleId>,
    mesh: LineMesh<3>,
    segments: Vec<SegmentRecord>,
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
            .and_then(|ring| HandleDrag::press(Handle::Rotate(ring), origin, direction));
        self.applied = Rotor4::IDENTITY;
        self.hover = self.drag.map(|drag| drag.id());
        self.drag.is_some()
    }

    pub(crate) fn turn(&mut self, ray: &ImageRay) -> Option<Rotor4> {
        let drag = self.drag?;
        let total = drag
            .delta(Vec3::from(ray.origin), Vec3::from(ray.direction))?
            .rotor();
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
                .map(|ring| HandleId::Rotate(ring.plane))
        });
    }

    pub(crate) fn rings(&mut self, center: Vec3) -> &[SegmentRecord] {
        self.segments.clear();
        if !self.enabled {
            return &self.segments;
        }
        let mut style = RingStyle::default();
        if let Some(HandleId::Rotate(plane)) = self.hover {
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
            self.segments.push(SegmentRecord {
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
        &self.segments
    }
}
