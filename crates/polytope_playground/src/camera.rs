use loam::prelude::*;

use crate::consts::BODY_Y;
use crate::mode::Mode;
use crate::strip::Strip;
use crate::Playground;

const CAMERA_DISTANCE: f32 = 8.0;

pub(crate) fn focus_camera(
    orbit: &mut Orbit,
    camera_focus: &mut Option<(bool, usize, Mode)>,
    filmstrip_return: &mut Option<Orbit>,
    strip: Strip,
    focus: (bool, usize, Mode),
    center: glam::Vec3,
) {
    if strip.on {
        if filmstrip_return.is_none() {
            let mut centered = Orbit::around([0.0; 3], orbit.distance);
            centered.yaw = orbit.yaw;
            centered.pitch = orbit.pitch;
            *filmstrip_return = Some(std::mem::replace(orbit, centered));
            *camera_focus = Some(focus);
        }
        orbit.target[1] = if strip.w && strip.t { BODY_Y } else { 0.0 };
        return;
    }
    if let Some(returned) = filmstrip_return.take() {
        *orbit = returned;
    }
    if *camera_focus == Some(focus) {
        return;
    }
    let reset =
        camera_focus.is_some_and(|previous| previous.2 != focus.2 || previous.0 && !focus.0);
    let target = [center.x, 0.0, center.z];
    if reset {
        *orbit = home(target);
    } else {
        orbit.target = target;
    }
    *camera_focus = Some(focus);
}

pub(crate) fn home(target: [f32; 3]) -> Orbit {
    Orbit {
        pitch: -0.25,
        ..Orbit::around(target, CAMERA_DISTANCE)
    }
}

pub(crate) fn scene_center(
    app: &Playground,
    domains: &Domains,
    domain: DomainHandle<EuclideanR4>,
    alpha: f32,
) -> glam::Vec3 {
    let active = *app.active;
    let single = app.display.single;
    let Ok(r4) = domains.read(domain) else {
        return glam::Vec3::ZERO;
    };
    let mut sum = glam::Vec3::ZERO;
    let mut count = 0;
    for (entity, slot) in app.slots.iter() {
        if single && slot.index != active {
            continue;
        }
        let Some(pose) = r4.drawn_pose(entity, alpha) else {
            continue;
        };
        sum += pose.point.truncate();
        count += 1;
    }
    sum / count.max(1) as f32
}
