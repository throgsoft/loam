use loam::prelude::*;

use crate::action::Action;
use crate::color::{ColorMode, WIREFRAME_COLOR};
use crate::display::{
    Surface, SECTION_PERIMETER, SURFACE, WIREFRAME, WIREFRAME_OPACITY, WIREFRAME_WIDTH,
};
use crate::mode::{Mode, RATE};
use crate::projection::Family;
use crate::{
    Playground, SHOW_CONTROLS, SHOW_FORMULA, SHOW_GIMBAL, SHOW_GUIDES, SHOW_HUD, SHOW_LABELS,
    TOYBOX_DEBUG,
};

const MAX_SEEK_SECONDS: f32 = 1.0e6;

fn number(token: &str, name: &'static str) -> Result<f32, CommandError> {
    token
        .parse()
        .map_err(|error| CommandError::invalid(name, token, error))
}

fn usage(form: &str) -> CommandError {
    CommandError::Usage(form.into())
}

fn toggle(args: &[&str], form: &str) -> Result<Option<bool>, CommandError> {
    match args {
        [] => Ok(None),
        ["on"] => Ok(Some(true)),
        ["off"] => Ok(Some(false)),
        _ => Err(usage(form)),
    }
}

fn color(token: &str) -> Option<ColorMode> {
    match token {
        "vertex-gradient" => Some(ColorMode::VertexGradient),
        "unique-edge" => Some(ColorMode::UniqueEdge),
        _ => WIREFRAME_COLOR.parse(token),
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
                let mode = Mode::from_token(token).ok_or_else(|| CommandError::Unknown {
                    name: "demo",
                    token: token.to_string(),
                    choices: "rotate|toybox",
                })?;
                submit.app(Action::Mode(mode));
                Ok(())
            }
            _ => Err(usage("demo [rotate|toybox]")),
        },
    );

    app = app.command(
        "reset",
        "reset poses, rotation, and slice; shapes, view, and camera stay",
        move |args, submit, _out| {
            if !args.is_empty() {
                return Err(usage("reset"));
            }
            submit.app(Action::Reset);
            Ok(())
        },
    );

    app = app.command(
        "spin",
        "play or pause rotation; optional on | off",
        |args, submit, _out| {
            submit.app(Action::Running(toggle(args, "spin [on|off]")?));
            Ok(())
        },
    );

    app = app.setting(SHOW_HUD, "toggle the state readout; optional on | off");

    app = app.command(
        "seek",
        "set rotation time in seconds",
        |args, submit, _out| {
            let [token] = args else {
                return Err(usage("seek <seconds>"));
            };
            let time = number(token, "seek")?;
            if !time.is_finite() || !(0.0..=MAX_SEEK_SECONDS).contains(&time) {
                return Err(CommandError::OutOfRange {
                    name: "seek",
                    value: time,
                    expected: "a time between 0 and 1000000 seconds",
                });
            }
            submit.app(Action::Time(time));
            Ok(())
        },
    );

    app = app.number(RATE, "set rotation speed from 0.25 to 4");

    app = app.command(
        "active",
        "select the active shape by its slot",
        |args, submit, _out| {
            let [token] = args else {
                return Err(usage("active <slot>"));
            };
            let slot = token
                .parse()
                .map_err(|error| CommandError::invalid("active", token, error))?;
            submit.app(Action::Active(slot));
            Ok(())
        },
    );

    app = app.command(
        "view",
        "show every shape or only the active one: shapes | single",
        |args, submit, _out| {
            let single = match args {
                ["shapes"] => false,
                ["single"] => true,
                _ => return Err(usage("view shapes|single")),
            };
            submit.app(Action::View { single });
            Ok(())
        },
    );

    app = app.command("slice", "set the w cross-section", |args, submit, _out| {
        let [token] = args else {
            return Err(usage("slice <w>"));
        };
        let w = number(token, "slice")?;
        if !w.is_finite() {
            return Err(CommandError::OutOfRange {
                name: "slice",
                value: w,
                expected: "a finite w",
            });
        }
        submit.app(Action::ExactSlice(w));
        Ok(())
    });

    app = app.setting(
        SHOW_CONTROLS,
        "toggle the controls panel; optional on | off",
    );

    app = app.setting(
        SHOW_FORMULA,
        "toggle the rotation formula; optional on | off",
    );

    app = app.setting(SHOW_LABELS, "toggle the shape labels; optional on | off");

    app = app.setting(
        SHOW_GUIDES,
        "toggle the toybox floor guides; optional on | off",
    );

    app = app.command(
        "toybox",
        "toybox debug drawing: debug [on|off]",
        |args, submit, _out| match args {
            ["debug", rest @ ..] => submit.setting(TOYBOX_DEBUG, rest),
            _ => Err(usage("toybox debug [on|off]")),
        },
    );

    app = app.setting(
        SHOW_GIMBAL,
        "toggle the 4D rotation handles; optional on | off",
    );

    app = app.command(
        "surface",
        "set the surface mode: raster | sdf | off; bare selects off",
        |args, submit, _out| match args {
            [] => {
                submit.submit(SURFACE.set(Surface::Off));
                Ok(())
            }
            rest => submit.setting(SURFACE, rest),
        },
    );

    app = app.command(
        "wireframe",
        "wireframe [on|off|width [px]|alpha [value]|color [mode]|perspective [mode]]",
        |args, submit, _out| {
            match args {
                [] | ["on" | "off"] => submit.setting(WIREFRAME, args)?,
                ["width"] => submit.inspect("wireframe width", |dispatch, out| {
                    out.line(format!(
                        "wireframe width: {:.2} px",
                        dispatch.app.display.get().wireframe_width_px
                    ));
                }),
                ["width", rest @ ..] => submit.number(WIREFRAME_WIDTH, rest)?,
                ["alpha"] => submit.inspect("wireframe alpha", |dispatch, out| {
                    out.line(format!(
                        "wireframe alpha: {:.3}",
                        dispatch.app.display.get().wireframe_opacity
                    ));
                }),
                ["alpha", rest @ ..] => submit.number(WIREFRAME_OPACITY, rest)?,
                ["color"] => submit.submit(WIREFRAME_COLOR.toggle()),
                ["color", token] => {
                    let mode = color(token).ok_or_else(|| CommandError::Unknown {
                        name: "wireframe color",
                        token: token.to_string(),
                        choices: "vertex-gradient|unique-edge|w-depth",
                    })?;
                    submit.submit(WIREFRAME_COLOR.set(mode));
                }
                ["perspective"] => submit.app(Action::NextProjection),
                ["perspective", token] => {
                    let family = projection(token).ok_or_else(|| CommandError::Unknown {
                        name: "wireframe perspective",
                        token: token.to_string(),
                        choices: "shadow|perspective|stereographic",
                    })?;
                    submit.app(Action::Projection(family));
                }
                _ => return Err(usage(
                    "wireframe [on|off|width [px]|alpha [value]|color [mode]|perspective [mode]]",
                )),
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
                ["perimeter", rest @ ..] => submit.setting(SECTION_PERIMETER, rest)?,
                _ => return Err(usage("section perimeter [on|off]")),
            }
            Ok(())
        },
    );

    app
}
