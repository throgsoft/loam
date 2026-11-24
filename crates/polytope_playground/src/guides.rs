use glam::{Vec2, Vec3};
use loam::prelude::*;

use crate::consts::FLOOR_Y;
use crate::mode::Mode;
use crate::toy::{ARENA_HALF, BODY_SIZE};
use crate::Playground;

const HANDLE_COLOR: [f32; 4] = [1.0, 0.85, 0.25, 1.0];

pub(crate) fn draw(ctx: &mut Ctx<'_, Playground>, domain: DomainHandle<EuclideanR4>) {
    if *ctx.app.mode.get() != Mode::Toybox || ctx.app.strip.get().on || !*ctx.app.guides.get() {
        return;
    }
    let drag = ctx.dragging();
    let alpha = ctx.alpha;
    let overlay = &mut *ctx.overlay;
    rectangle(
        &mut overlay.lines,
        Vec2::splat(-ARENA_HALF),
        Vec2::splat(ARENA_HALF),
        [0.55, 0.60, 0.68, 1.0],
    );
    let Some(drag) = drag else {
        return;
    };
    let Some(polytope) = ctx
        .app
        .slots
        .get(drag.entity)
        .and_then(|slot| slot.entry.shape.polytope4())
    else {
        return;
    };
    let Ok(r4) = ctx.domains.read(domain) else {
        return;
    };
    if *ctx.app.toybox_debug.get() {
        let anchor = r4
            .physics()
            .and_then(|physics| physics.held_anchor(drag.entity));
        if let (Some(anchor), Some(tick), Some(drawn), Some(view), Some(eye)) = (
            anchor,
            r4.poses().get(drag.entity),
            r4.drawn_pose(drag.entity, alpha),
            r4.view(drag.view),
            r4.view_eye(drag.view)
                .and_then(|eye| r4.drawn_pose(eye, alpha)),
        ) {
            let anchor = drawn.point
                + drawn
                    .frame
                    .apply(tick.frame.inverse().apply(anchor - tick.point));
            if let Some(point) = view.mapping().image_point(&eye, anchor) {
                overlay.points.push(PointRecord {
                    position: point,
                    radius_px: 6.0,
                    color: HANDLE_COLOR,
                });
            }
        }
    }
    let Some(pose) = r4.drawn_pose(drag.entity, alpha) else {
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
    rectangle(&mut overlay.lines, min, max, HANDLE_COLOR);
}

fn rectangle(lines: &mut Vec<SegmentRecord>, min: Vec2, max: Vec2, color: [f32; 4]) {
    let y = FLOOR_Y + 0.002;
    let corners = [
        Vec3::new(min.x, y, min.y),
        Vec3::new(max.x, y, min.y),
        Vec3::new(max.x, y, max.y),
        Vec3::new(min.x, y, max.y),
    ];
    for index in 0..4 {
        lines.push(SegmentRecord {
            start: corners[index].to_array(),
            end: corners[(index + 1) % 4].to_array(),
            start_color: color,
            end_color: color,
            width_px: 1.5,
            ..SegmentRecord::default()
        });
    }
}
