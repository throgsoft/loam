use glam::{Vec2, Vec3};
use loam::math::{EuclideanR4, Rotor};
use loam::runtime::{DomainHandle, PointRecord, SegmentRecord, Session};

use crate::consts::FLOOR_Y;
use crate::mode::Mode;
use crate::toy::{ARENA_HALF, BODY_SIZE};
use crate::Playground;

const HANDLE_COLOR: [f32; 4] = [1.0, 0.85, 0.25, 1.0];

#[derive(Default)]
pub(crate) struct Guides {
    pub(crate) lines: Vec<SegmentRecord>,
    pub(crate) points: Vec<PointRecord>,
}

impl Guides {
    pub(crate) fn update(
        &mut self,
        session: &Session<Playground>,
        domain: DomainHandle<EuclideanR4>,
    ) {
        self.lines.clear();
        self.points.clear();
        if *session.app.mode.get() != Mode::Toybox
            || session.app.strip.get().on
            || !*session.app.guides.get()
        {
            return;
        }
        self.rectangle(
            Vec2::splat(-ARENA_HALF),
            Vec2::splat(ARENA_HALF),
            [0.55, 0.60, 0.68, 1.0],
        );
        let Some(drag) = session.dragging() else {
            return;
        };
        let Some(polytope) = session
            .app
            .slots
            .get(drag.entity)
            .and_then(|slot| slot.entry.shape.polytope4())
        else {
            return;
        };
        let Ok(r4) = session.domains().read(domain) else {
            return;
        };
        if *session.app.toybox_debug.get() {
            let placement = session
                .views()
                .to_root(drag.image)
                .and_then(|to| to.rigid());
            let anchor = r4
                .physics()
                .and_then(|physics| physics.held_anchor(drag.entity));
            if let (Some(anchor), Some(placement), Some(view), Some(eye)) = (
                anchor,
                placement,
                r4.view(drag.view),
                r4.view_eye(drag.view),
            ) {
                if let Some(point) = r4
                    .poses()
                    .get(eye)
                    .and_then(|eye| view.mapping().image_point(eye, anchor))
                {
                    self.points.push(PointRecord {
                        position: placement.apply(point),
                        radius_px: 6.0,
                        color: HANDLE_COLOR,
                    });
                }
            }
        }
        let Some(pose) = r4.poses().get(drag.entity) else {
            return;
        };
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for vertex in polytope.topology().vertices {
            let point = pose.point + pose.frame.apply(*vertex * BODY_SIZE);
            let xz = Vec2::new(point.x, point.z);
            min = min.min(xz);
            max = max.max(xz);
        }
        self.rectangle(min, max, HANDLE_COLOR);
    }

    fn rectangle(&mut self, min: Vec2, max: Vec2, color: [f32; 4]) {
        let y = FLOOR_Y + 0.002;
        let corners = [
            Vec3::new(min.x, y, min.y),
            Vec3::new(max.x, y, min.y),
            Vec3::new(max.x, y, max.y),
            Vec3::new(min.x, y, max.y),
        ];
        for index in 0..4 {
            self.lines.push(SegmentRecord {
                start: corners[index].to_array(),
                end: corners[(index + 1) % 4].to_array(),
                start_color: color,
                end_color: color,
                width_px: 1.5,
                ..SegmentRecord::default()
            });
        }
    }
}
