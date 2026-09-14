use anyhow::{anyhow, bail, Result};
use loam::app::environment::Environment;
use loam::app::session::SessionApp;
use loam::app::CursorPolicy;
use loam::runtime::Dispatch;

use crate::camera::CameraMode;
use crate::color::ColorMode;
use crate::display::{Surface, MAX_WIREFRAME_WIDTH_PX};
use crate::mode::Mode;
use crate::projection::Family;
use crate::{Action, Playground};

const MAX_SEEK_SECONDS: f32 = 1.0e6;

fn number(token: &str, name: &str) -> Result<f32> {
    token
        .parse()
        .map_err(|error| anyhow!("{name}: invalid `{token}`: {error}"))
}

fn toggle(args: &[&str], usage: &str) -> Result<Option<bool>> {
    match args {
        [] => Ok(None),
        ["on"] => Ok(Some(true)),
        ["off"] => Ok(Some(false)),
        _ => bail!("{usage}"),
    }
}

fn color(token: &str) -> Option<ColorMode> {
    match token {
        "vertex-gradient" => Some(ColorMode::VertexGradient),
        "unique-edge" => Some(ColorMode::UniqueEdge),
        _ => ColorMode::from_token(token),
    }
}

fn projection(token: &str) -> Option<Family> {
    match token {
        "w-pinhole" => Some(Family::Perspective),
        _ => Family::from_token(token),
    }
}

fn set_camera_mode(dispatch: &mut Dispatch<'_, Playground>, mode: CameraMode) {
    let previous = dispatch.app.camera.get().mode;
    if mode == CameraMode::Freecam && previous != CameraMode::Freecam {
        let eye = dispatch.views.root_mut().eye;
        dispatch.app.camera.get_mut().free.eye = eye;
    }
    dispatch.app.camera.get_mut().mode = mode;
}

pub(crate) fn install(mut app: SessionApp<Playground>) -> SessionApp<Playground> {
    app = app.command(
        "demo",
        "show or select the active demo: rotate | toybox",
        move |args, submit, _out| match args {
            [] => {
                submit.inspect("demo", |dispatch, out| {
                    out.line(format!("demo: {}", dispatch.app.mode.get().name()));
                });
                Ok(())
            }
            [token] => {
                let mode = Mode::from_token(token)
                    .ok_or_else(|| anyhow!("unknown demo `{token}` (try rotate|toybox)"))?;
                submit.app(Action::Mode(mode));
                Ok(())
            }
            _ => bail!("usage: demo [rotate|toybox]"),
        },
    );

    app = app.command(
        "reset",
        "reset the active demo in place",
        move |args, submit, _out| {
            if !args.is_empty() {
                bail!("usage: reset");
            }
            submit.app(Action::Reset);
            Ok(())
        },
    );

    app = app.command(
        "spin",
        "play or pause rotation; optional on | off",
        |args, submit, _out| {
            submit.app(Action::Running(toggle(args, "usage: spin [on|off]")?));
            Ok(())
        },
    );

    app = app.command(
        "hud",
        "toggle the state readout; optional on | off",
        |args, submit, _out| {
            submit.app(Action::Hud(toggle(args, "usage: hud [on|off]")?));
            Ok(())
        },
    );

    app = app.command(
        "seek",
        "set rotation time in seconds",
        |args, submit, _out| {
            let [token] = args else {
                bail!("usage: seek <seconds>");
            };
            let time = number(token, "seek")?;
            if !time.is_finite() || !(0.0..=MAX_SEEK_SECONDS).contains(&time) {
                bail!("time must be between 0 and 1000000 seconds");
            }
            submit.app(Action::Time(time));
            Ok(())
        },
    );

    app = app.command(
        "rate",
        "set rotation speed from 0.25 to 4",
        |args, submit, _out| {
            let [token] = args else {
                bail!("usage: rate <multiplier>");
            };
            let rate = number(token, "rate")?;
            if !rate.is_finite() || !(0.25..=4.0).contains(&rate) {
                bail!("rate must be between 0.25 and 4");
            }
            submit.app(Action::Rate(rate));
            Ok(())
        },
    );

    app = app.command("slice", "set the w cross-section", |args, submit, _out| {
        let [token] = args else {
            bail!("usage: slice <w>");
        };
        let w = number(token, "slice")?;
        if !w.is_finite() {
            bail!("slice must be finite");
        }
        submit.app(Action::ExactSlice(w));
        Ok(())
    });

    app = app.command(
        "controls",
        "toggle the controls panel; optional on | off",
        |args, submit, _out| {
            submit.app(Action::Controls(toggle(args, "usage: controls [on|off]")?));
            Ok(())
        },
    );

    app = app.command(
        "formula",
        "toggle the rotation formula; optional on | off",
        |args, submit, _out| {
            submit.app(Action::Formula(toggle(args, "usage: formula [on|off]")?));
            Ok(())
        },
    );

    app = app.command(
        "handles",
        "toggle the 4D rotation handles; optional on | off",
        |args, submit, _out| {
            submit.app(Action::Gimbal(toggle(args, "usage: handles [on|off]")?));
            Ok(())
        },
    );

    app = app.command(
        "floor",
        "toggle the ground plane; optional on | off",
        |args, submit, _out| {
            let setting = toggle(args, "usage: floor [on|off]")?;
            submit.app_fn("floor", move |dispatch: &mut Dispatch<'_, Playground>| {
                let shown = &mut dispatch.app.environment.get_mut().floor_visible;
                *shown = setting.unwrap_or(!*shown);
            });
            Ok(())
        },
    );

    app = app.command(
        "ground",
        "read or set checker colors and fog: dark | light | fog | reset",
        |args, submit, _out| {
            let field = match args {
                [] => Some(None),
                ["dark"] => Some(Some("dark")),
                ["light"] => Some(Some("light")),
                ["fog"] => Some(Some("fog")),
                _ => None,
            };
            if let Some(field) = field {
                submit.inspect("ground", move |dispatch, out| {
                    out.line(dispatch.app.environment.get().report(field));
                });
                return Ok(());
            }
            Environment::default().apply(args)?;
            let owned = args
                .iter()
                .map(|token| (*token).to_string())
                .collect::<Vec<_>>();
            submit.app_fn("ground", move |dispatch: &mut Dispatch<'_, Playground>| {
                let args = owned.iter().map(String::as_str).collect::<Vec<_>>();
                let mut environment = *dispatch.app.environment.get();
                if environment.apply(&args).is_ok() {
                    dispatch.app.environment.set(environment);
                }
            });
            Ok(())
        },
    );

    app = app.command(
        "surface",
        "set the surface mode: raster | sdf | off; bare selects off",
        |args, submit, _out| {
            let surface = match args {
                [] | ["off"] => Surface::Off,
                ["raster"] => Surface::Raster,
                ["sdf"] => Surface::Sdf,
                _ => bail!("usage: surface [raster|sdf|off]"),
            };
            submit.app_fn("surface", move |dispatch: &mut Dispatch<'_, Playground>| {
                dispatch.app.display.get_mut().surface = surface;
            });
            Ok(())
        },
    );

    app = app.command(
        "wireframe",
        "wireframe [on|off|width [px]|alpha [value]|color [mode]|perspective [mode]]",
        |args, submit, _out| {
            match args {
                [] => submit.app_fn(
                    "wireframe",
                    |dispatch: &mut Dispatch<'_, Playground>| {
                        let shown = &mut dispatch.app.display.get_mut().wireframe;
                        *shown = !*shown;
                    },
                ),
                ["on"] | ["off"] => {
                    let shown = args[0] == "on";
                    submit.app_fn(
                        "wireframe",
                        move |dispatch: &mut Dispatch<'_, Playground>| {
                            dispatch.app.display.get_mut().wireframe = shown;
                        },
                    );
                }
                ["width"] => submit.inspect("wireframe width", |dispatch, out| {
                    out.line(format!(
                        "wireframe width: {:.2} px",
                        dispatch.app.display.get().wireframe_width_px
                    ));
                }),
                ["width", token] => {
                    let width_px = number(token, "wireframe width")?;
                    if !width_px.is_finite()
                        || width_px <= 0.0
                        || width_px > MAX_WIREFRAME_WIDTH_PX
                    {
                        bail!(
                            "wireframe width {width_px} out of range; expected (0, {MAX_WIREFRAME_WIDTH_PX}]"
                        );
                    }
                    submit.app_fn(
                        "wireframe width",
                        move |dispatch: &mut Dispatch<'_, Playground>| {
                            dispatch.app.display.get_mut().wireframe_width_px = width_px;
                        },
                    );
                }
                ["alpha"] => submit.inspect("wireframe alpha", |dispatch, out| {
                    out.line(format!(
                        "wireframe alpha: {:.3}",
                        dispatch.app.display.get().wireframe_opacity
                    ));
                }),
                ["alpha", token] => {
                    let opacity = number(token, "wireframe alpha")?;
                    if !opacity.is_finite() || opacity <= 0.0 || opacity > 1.0 {
                        bail!("wireframe alpha {opacity} out of range; expected (0, 1]");
                    }
                    submit.app_fn(
                        "wireframe alpha",
                        move |dispatch: &mut Dispatch<'_, Playground>| {
                            dispatch.app.display.get_mut().wireframe_opacity = opacity;
                        },
                    );
                }
                ["color"] => submit.app(Action::NextColor),
                ["color", token] => {
                    let mode = color(token).ok_or_else(|| {
                        anyhow!("unknown color mode `{token}` (try vertex-gradient|unique-edge|w-depth)")
                    })?;
                    submit.app(Action::Color(mode));
                }
                ["perspective"] => submit.app(Action::NextProjection),
                ["perspective", token] => {
                    let family = projection(token).ok_or_else(|| {
                        anyhow!("unknown projection `{token}` (try shadow|perspective|stereographic)")
                    })?;
                    submit.app(Action::Projection(family));
                }
                _ => bail!(
                    "usage: wireframe [on|off|width [px]|alpha [value]|color [mode]|perspective [mode]]"
                ),
            }
            Ok(())
        },
    );

    app = app.command(
        "section",
        "section perimeter [on|off]",
        |args, submit, _out| {
            match args {
                [] => submit.inspect("section", |dispatch, out| {
                    out.line(format!(
                        "section perimeter: {}",
                        if dispatch.app.display.get().section_perimeter {
                            "on"
                        } else {
                            "off"
                        }
                    ));
                }),
                ["perimeter"] => submit.app_fn(
                    "section perimeter",
                    |dispatch: &mut Dispatch<'_, Playground>| {
                        let shown = &mut dispatch.app.display.get_mut().section_perimeter;
                        *shown = !*shown;
                    },
                ),
                ["perimeter", "on"] | ["perimeter", "off"] => {
                    let shown = args[1] == "on";
                    submit.app_fn(
                        "section perimeter",
                        move |dispatch: &mut Dispatch<'_, Playground>| {
                            dispatch.app.display.get_mut().section_perimeter = shown;
                        },
                    );
                }
                _ => bail!("usage: section perimeter [on|off]"),
            }
            Ok(())
        },
    );

    app = app.command(
        "camera",
        "camera [orbit|freecam|speed <N>|cursor <hold|toggle>]; bare cycles",
        |args, submit, _out| {
            match args {
                [] => submit.app_fn("camera", |dispatch: &mut Dispatch<'_, Playground>| {
                    let mode = match dispatch.app.camera.get().mode {
                        CameraMode::Orbit => CameraMode::Freecam,
                        CameraMode::Freecam => CameraMode::Orbit,
                    };
                    set_camera_mode(dispatch, mode);
                }),
                ["orbit"] | ["freecam"] => {
                    let mode = if args[0] == "orbit" {
                        CameraMode::Orbit
                    } else {
                        CameraMode::Freecam
                    };
                    submit.app_fn("camera", move |dispatch: &mut Dispatch<'_, Playground>| {
                        set_camera_mode(dispatch, mode);
                    });
                }
                ["speed"] => submit.inspect("camera speed", |dispatch, out| {
                    out.line(format!(
                        "camera speed: {:.2} u/sec",
                        dispatch.app.camera.get().speed
                    ));
                }),
                ["speed", token] => {
                    let speed = number(token, "camera speed")?;
                    if !speed.is_finite() || speed <= 0.0 {
                        bail!("camera speed must be finite and positive");
                    }
                    submit.app_fn(
                        "camera speed",
                        move |dispatch: &mut Dispatch<'_, Playground>| {
                            dispatch.app.camera.get_mut().speed = speed;
                        },
                    );
                }
                ["cursor"] => submit.inspect("camera cursor", |dispatch, out| {
                    let policy = match dispatch.app.camera.get().cursor_policy {
                        CursorPolicy::Hold => "hold",
                        CursorPolicy::Toggle => "toggle",
                    };
                    out.line(format!("camera cursor: {policy}"));
                }),
                ["cursor", "hold"] | ["cursor", "toggle"] => {
                    let policy = if args[1] == "hold" {
                        CursorPolicy::Hold
                    } else {
                        CursorPolicy::Toggle
                    };
                    submit.app_fn(
                        "camera cursor",
                        move |dispatch: &mut Dispatch<'_, Playground>| {
                            dispatch.app.camera.get_mut().cursor_policy = policy;
                        },
                    );
                }
                _ => bail!("camera: unknown argument (try orbit|freecam|speed|cursor)"),
            }
            Ok(())
        },
    );

    app
}

#[cfg(test)]
mod tests {
    use glam::Vec4;
    use loam::app::args::Args;
    use loam::runtime::host::HostConfig;
    use loam::runtime::{
        Command, Entity, Eye, Input, Pose, Publication, Rejection, Section4, SpawnBundle, ViewSpec,
    };

    use super::*;
    use crate::catalog::DEFAULT_ROW;
    use crate::{bindings, boot};

    #[test]
    fn console_commands_keep_source_order_and_rejected_styles_preserve_state() {
        let mut booted = boot(&DEFAULT_ROW[..1]).expect("the session boots");
        let eye = Eye::looking_at([2.0, 3.0, 7.0], [0.5, 0.25, -1.0], [0.0, 1.0, 0.0]);
        booted.session.views_mut().root_mut().eye = eye;
        let mut app = install(SessionApp::with_args(
            HostConfig::new("console test", bindings()),
            Args::default(),
        ));
        for line in [
            "demo toybox",
            "reset",
            "demo",
            "slice 2",
            "wireframe width 3",
            "wireframe width",
            "wireframe width NaN",
            "wireframe width 0",
            "wireframe width 17",
            "wireframe alpha 0",
            "wireframe alpha NaN",
            "camera freecam",
        ] {
            app.console_mut().execute(line);
        }
        app.console_mut().dispatch_pending();
        app.boundary(&mut booted.session, Input::default())
            .expect("the console commands apply");

        assert_eq!(*booted.session.app.mode.get(), Mode::Toybox);
        assert_eq!(*booted.session.app.slice.get(), 2.0);
        assert_eq!(booted.session.app.display.get().wireframe_width_px, 3.0);
        assert_eq!(booted.session.app.display.get().wireframe_opacity, 1.0);
        assert_eq!(booted.session.app.camera.get().mode, CameraMode::Freecam);
        assert_eq!(booted.session.app.camera.get().free.eye, eye);
        let history = app
            .console_mut()
            .ui()
            .history()
            .iter()
            .map(|line| line.text.clone())
            .collect::<Vec<_>>();
        assert!(history.iter().any(|line| line == "demo: toybox"));
        assert!(history
            .iter()
            .any(|line| line == "wireframe width: 3.00 px"));
    }

    #[test]
    fn installed_playground_console_recovers_a_fault_at_ordinary_capacity() {
        const ORDINARY_CAPACITY: usize = 1023;

        let mut booted = boot(&DEFAULT_ROW[..1]).expect("the session boots");
        let domain = booted.domain;
        let root = booted.session.views().root();
        let eye = booted
            .session
            .dispatch(|dispatch| -> Result<Entity, Rejection> {
                let eye = dispatch.spawn(SpawnBundle::new().at(domain, Pose::at(Vec4::ZERO)))?;
                dispatch.domains.typed(domain)?.add_view(ViewSpec::new(
                    root,
                    eye,
                    Section4 { w: 0.0 },
                ))?;
                Ok(eye)
            })
            .expect("the invalidated view is installed");
        let epoch = booted.session.scene().epoch();
        let mut app = install(SessionApp::with_args(
            HostConfig::new("console recovery test", bindings()),
            Args::default(),
        ));

        app.sender().submit(Command::Despawn(eye));
        app.boundary(&mut booted.session, Input::default())
            .expect("the eye despawns");
        let mut publication = Publication::default();
        booted
            .session
            .publish(&mut publication)
            .expect_err("the dangling view faults publication");

        for _ in 0..ORDINARY_CAPACITY {
            app.console_mut().execute("reset");
        }
        app.console_mut().execute("recover");
        app.console_mut().dispatch_pending();
        app.boundary(&mut booted.session, Input::default())
            .expect("the host recovers");

        assert_eq!(booted.session.faulted_phase(), None);
        assert_eq!(booted.session.scene().epoch(), epoch.advance());
    }
}
