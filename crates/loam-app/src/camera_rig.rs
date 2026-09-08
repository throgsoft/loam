use anyhow::{anyhow, Result};
use loam_egui::Console;

use crate::freecam::{CursorMode, Freecam};
use crate::{Camera, CameraController, Input, OrbitController, UiCapture};
use loam_math::EuclideanR3;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CameraMode {
    #[default]
    Orbit,
    FreeRoam,
}

impl CameraMode {
    pub fn label(self) -> &'static str {
        match self {
            CameraMode::Orbit => "orbit",
            CameraMode::FreeRoam => "freecam",
        }
    }
}

#[derive(Debug, Default)]
pub struct CameraRig {
    pub mode: CameraMode,
    pub freecam: Freecam,
}

impl CameraRig {
    pub fn is_flying(&self) -> bool {
        self.mode == CameraMode::FreeRoam
    }

    pub fn toggle(&mut self) {
        self.mode = match self.mode {
            CameraMode::Orbit => CameraMode::FreeRoam,
            CameraMode::FreeRoam => CameraMode::Orbit,
        };
    }

    /// Pass the chosen orbit button in `input.buttons.left`.
    pub fn advance(
        &mut self,
        input: Input,
        capture: UiCapture,
        camera: &mut Camera<EuclideanR3>,
        orbit: &mut OrbitController<EuclideanR3>,
        dt: f32,
        runtime: &crate::Runtime,
    ) {
        let flying = self.is_flying();
        self.freecam.set_active(flying, camera, runtime);
        match self.mode {
            CameraMode::Orbit if !capture.pointer => orbit.advance(input, camera, &EuclideanR3, dt),
            CameraMode::FreeRoam if !(capture.pointer || capture.keyboard) => {
                self.freecam.advance(input, camera, dt, runtime);
            }
            _ => {}
        }
    }
}

pub fn register_camera_command<Ctx: 'static>(
    console: &mut Console<Ctx>,
    reach: fn(&mut Ctx) -> &mut CameraRig,
) {
    console.register(
        loam_egui::cmd::<Ctx, _>(
            "camera",
            "camera [orbit|freecam|speed <N>|cursor <hold|toggle>]; bare cycles",
            move |args, ctx, out| {
                let rig = reach(ctx);
                match args.first().copied() {
                    None => {
                        rig.toggle();
                    }
                    Some("orbit") => rig.mode = CameraMode::Orbit,
                    Some("freecam") => rig.mode = CameraMode::FreeRoam,
                    Some("speed") => return set_speed(rig, args.get(1).copied(), out),
                    Some("cursor") => return set_cursor(rig, args.get(1).copied(), out),
                    Some(other) => {
                        return Err(anyhow!(
                            "camera: unknown arg `{other}` (try orbit|freecam|speed|cursor)"
                        ));
                    }
                }
                out.line(format!("camera: {}", rig.mode.label()));
                Ok(())
            },
        )
        .with_args(&[&["orbit", "freecam", "speed", "cursor"]]),
    );
}

fn set_speed(
    rig: &mut CameraRig,
    value: Option<&str>,
    out: &mut loam_egui::ConsoleWriter,
) -> Result<()> {
    let Some(value) = value else {
        out.line(format!("camera speed: {:.2} u/sec", rig.freecam.speed));
        return Ok(());
    };
    let parsed: f32 = value
        .parse()
        .map_err(|e| anyhow!("camera speed: invalid `{value}`: {e}"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(anyhow!("camera speed must be finite and positive"));
    }
    rig.freecam.speed = parsed;
    out.line(format!("camera speed: set to {parsed:.2} u/sec"));
    Ok(())
}

fn set_cursor(
    rig: &mut CameraRig,
    value: Option<&str>,
    out: &mut loam_egui::ConsoleWriter,
) -> Result<()> {
    let mode = match value {
        None => {
            out.line(format!("camera cursor: {:?}", rig.freecam.cursor_mode()));
            return Ok(());
        }
        Some("hold") => CursorMode::Hold,
        Some("toggle") => CursorMode::Toggle,
        Some(other) => {
            return Err(anyhow!(
                "camera cursor: unknown `{other}` (try hold|toggle)"
            ));
        }
    };
    rig.freecam.set_cursor_mode(mode);
    out.line(format!("camera cursor: {mode:?}"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_change_activates_freecam_before_movement() {
        let runtime = crate::Runtime::default();
        let mut rig = CameraRig::default();
        let mut camera = Camera::at_origin();
        let mut orbit = OrbitController::default();
        rig.toggle();
        rig.advance(
            Input {
                move_forward: 1.0,
                ..Default::default()
            },
            UiCapture::default(),
            &mut camera,
            &mut orbit,
            1.0,
            &runtime,
        );
        assert!((camera.position.z + rig.freecam.speed).abs() < 1e-6);
        assert!(rig.freecam.active());
        rig.toggle();
        rig.advance(
            Input::default(),
            UiCapture::default(),
            &mut camera,
            &mut orbit,
            0.0,
            &runtime,
        );
        assert!(!rig.freecam.active());
        assert!(!rig.freecam.cursor_grabbed());
    }
}
