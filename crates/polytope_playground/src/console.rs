use anyhow::{anyhow, bail, Result};
use loam::app::environment::Environment;
use loam::prelude::*;

use crate::color::ColorMode;
use crate::display::{Surface, MAX_WIREFRAME_WIDTH_PX};
use crate::mode::Mode;
use crate::projection::Family;
use crate::{
    Action, Playground, SHOW_CONTROLS, SHOW_FORMULA, SHOW_GIMBAL, SHOW_GUIDES, TOYBOX_DEBUG,
};

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

fn free_camera(views: &Views) -> &FreeCamera {
    views
        .get(views.root())
        .and_then(|root| root.rig.free())
        .expect("the playground's rig is a pair")
}

fn free_camera_mut(views: &mut Views) -> &mut FreeCamera {
    views
        .root_mut()
        .rig
        .free_mut()
        .expect("the playground's rig is a pair")
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
        "reset poses, rotation, and slice; shapes, view, and camera stay",
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

    app = app.toggle_command(
        "hud",
        "toggle the state readout; optional on | off",
        |app| app.hud.get_mut(),
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

    app = app.command(
        "active",
        "select the active shape by its slot",
        |args, submit, _out| {
            let [token] = args else {
                bail!("usage: active <slot>");
            };
            let slot = token
                .parse()
                .map_err(|error| anyhow!("active: invalid `{token}`: {error}"))?;
            submit.app(Action::Active(slot));
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

    app = app.toggle_command(
        SHOW_CONTROLS.name,
        "toggle the controls panel; optional on | off",
        SHOW_CONTROLS.setting,
    );

    app = app.toggle_command(
        SHOW_FORMULA.name,
        "toggle the rotation formula; optional on | off",
        SHOW_FORMULA.setting,
    );

    app = app.toggle_command(
        "labels",
        "toggle the shape labels; optional on | off",
        |app| app.labels.get_mut(),
    );

    app = app.toggle_command(
        SHOW_GUIDES.name,
        "toggle the toybox floor guides; optional on | off",
        SHOW_GUIDES.setting,
    );

    app = app.command(
        "toybox",
        "toybox debug drawing: debug [on|off]",
        |args, submit, _out| match args {
            ["debug", rest @ ..] => {
                match toggle(rest, "usage: toybox debug [on|off]")? {
                    None => submit.toggle(TOYBOX_DEBUG.name, TOYBOX_DEBUG.setting),
                    Some(on) => submit.set(TOYBOX_DEBUG.name, TOYBOX_DEBUG.setting, on),
                }
                Ok(())
            }
            _ => bail!("usage: toybox debug [on|off]"),
        },
    );

    // A bare `handles` reports under `gimbal`, the name the G key's toggle uses.
    app = app.command(
        "handles",
        "toggle the 4D rotation handles; optional on | off",
        |args, submit, _out| {
            match toggle(args, "usage: handles [on|off]")? {
                None => submit.toggle(SHOW_GIMBAL.name, SHOW_GIMBAL.setting),
                Some(on) => submit.set("handles", SHOW_GIMBAL.setting, on),
            }
            Ok(())
        },
    );

    app = app.toggle_command(
        "floor",
        "toggle the ground plane; optional on | off",
        |app| &mut app.environment.get_mut().floor_visible,
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
                    let root = dispatch.views.root_mut();
                    root.set_flying(!root.rig.flying());
                }),
                ["orbit"] | ["freecam"] => {
                    let free = args[0] == "freecam";
                    submit.app_fn("camera", move |dispatch: &mut Dispatch<'_, Playground>| {
                        dispatch.views.root_mut().set_flying(free);
                    });
                }
                ["speed"] => submit.inspect("camera speed", |dispatch, out| {
                    let speed = free_camera(dispatch.views).speed;
                    out.line(format!("camera speed: {speed:.2} u/sec"));
                }),
                ["speed", token] => {
                    let speed = number(token, "camera speed")?;
                    if !speed.is_finite() || speed <= 0.0 {
                        bail!("camera speed must be finite and positive");
                    }
                    submit.app_fn(
                        "camera speed",
                        move |dispatch: &mut Dispatch<'_, Playground>| {
                            free_camera_mut(dispatch.views).speed = speed;
                        },
                    );
                }
                ["cursor"] => submit.inspect("camera cursor", |dispatch, out| {
                    let policy = match free_camera(dispatch.views).cursor {
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
                            free_camera_mut(dispatch.views).cursor = policy;
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
