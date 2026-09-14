use loam::math::{Bivector, EuclideanR4, Plane4, Rotor, Rotor4};
use loam::runtime::{Dispatch, DomainError, DomainHandle, Domains, Rejection};

use crate::consts::BASE_ROTATION_RATE;
use crate::mode::Mode;
use crate::Playground;

pub(crate) const DEFAULT_WIREFRAME_WIDTH_PX: f32 = 1.8;

pub(crate) const MAX_WIREFRAME_WIDTH_PX: f32 = 16.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Surface {
    #[default]
    Raster,
    Sdf,
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Display {
    pub(crate) surface: Surface,
    pub(crate) wireframe: bool,
    pub(crate) section_perimeter: bool,
    pub(crate) wireframe_width_px: f32,
    pub(crate) wireframe_opacity: f32,
    pub(crate) single: bool,
}

impl Default for Display {
    fn default() -> Self {
        Self {
            surface: Surface::Raster,
            wireframe: false,
            section_perimeter: true,
            wireframe_width_px: DEFAULT_WIREFRAME_WIDTH_PX,
            wireframe_opacity: 1.0,
            single: false,
        }
    }
}

pub(crate) fn set(app: &mut Playground, display: Display) -> Result<(), Rejection> {
    if !display.wireframe_width_px.is_finite()
        || display.wireframe_width_px <= 0.0
        || display.wireframe_width_px > MAX_WIREFRAME_WIDTH_PX
    {
        return Err(Rejection::Unsupported(
            "wireframe width must be finite and in (0, 16]",
        ));
    }
    if !display.wireframe_opacity.is_finite()
        || display.wireframe_opacity <= 0.0
        || display.wireframe_opacity > 1.0
    {
        return Err(Rejection::Unsupported(
            "wireframe opacity must be finite and in (0, 1]",
        ));
    }
    app.display.set(display);
    Ok(())
}

pub(crate) fn rotor_at(app: &Playground, time: f32) -> Rotor4 {
    match *app.mode.get() {
        Mode::Rotate => {
            let mut rotor = Rotor4::IDENTITY;
            for (index, plane) in Plane4::ALL.into_iter().enumerate() {
                let angle = app.angles.get()[index]
                    + if app.spin.get().planes[index] {
                        time * BASE_ROTATION_RATE
                    } else {
                        0.0
                    };
                rotor = (plane.unit_bivector() * angle).exp() * rotor;
            }
            rotor.normalize()
        }
        Mode::Toybox => Rotor4::IDENTITY,
    }
}

fn apply_turn(
    app: &Playground,
    domains: &mut Domains,
    domain: DomainHandle<EuclideanR4>,
    turn: Rotor4,
) -> Result<(), DomainError> {
    let r4 = domains.typed(domain)?;
    for (entity, _) in app.slots.iter() {
        if let Some(mut pose) = r4.poses().get(entity).copied() {
            pose.frame = (turn * pose.frame).normalize();
            r4.set_pose(entity, pose)?;
        }
    }
    Ok(())
}

pub(crate) fn seek(
    app: &mut Playground,
    domains: &mut Domains,
    domain: DomainHandle<EuclideanR4>,
    time: f32,
) -> Result<(), DomainError> {
    if !time.is_finite() || time < 0.0 {
        return Err(DomainError::Unsupported(
            "playback time must be finite and nonnegative",
        ));
    }
    if *app.mode.get() == Mode::Toybox {
        return Err(DomainError::Unsupported("playback belongs to Rotate"));
    }
    let before = rotor_at(app, *app.time.get());
    let after = rotor_at(app, time);
    apply_turn(app, domains, domain, after * before.inverse())?;
    app.time.set(time);
    Ok(())
}

pub(crate) fn set_plane_angle(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
    plane: usize,
    angle: f32,
) -> Result<(), Rejection> {
    if plane >= 6 || !angle.is_finite() {
        return Err(Rejection::Unsupported("invalid plane angle"));
    }
    let time = *dispatch.app.time.get();
    let before = rotor_at(dispatch.app, time);
    let advancing = dispatch.app.spin.get().planes[plane];
    dispatch.app.angles.get_mut()[plane] = angle
        - if advancing {
            time * BASE_ROTATION_RATE
        } else {
            0.0
        };
    let after = rotor_at(dispatch.app, time);
    apply_turn(
        dispatch.app,
        dispatch.domains,
        domain,
        after * before.inverse(),
    )
    .map_err(Rejection::Domain)
}
