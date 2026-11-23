use loam_console::{render_line, CommandLine, LineKind};
use loam_runtime::host::HostError;
use loam_runtime::{Bindings, PointerButton, Publication, Session, Stores};

use super::app::{InputHook, SessionApp};
use super::console::SessionConsole;
use super::cursor::CursorCapture;
use super::input::{dom_key, InputMap};
use crate::script::{ScriptDriver, ScriptStatus};

// A 720p canvas, so pointer deltas and the eye aspect match a common windowed run.
const FRAME: (u32, u32) = (1280, 720);

/// Steps like `host::run_headless` through `app`'s console and input until `driver` finishes, and a line runs after its frame publishes so it lands at the next boundary.
pub fn run_script<A: Stores>(
    session: &mut Session<A>,
    app: &mut SessionApp<A>,
    mut driver: ScriptDriver,
    alpha: f32,
    mut each: impl FnMut(u32, &mut Session<A>, &Publication<A::Published>) -> Result<(), HostError>,
) -> Result<Publication<A::Published>, HostError> {
    session.capture_initial();
    let dt = f64::from(session.config().dt().unwrap_or(0.0));
    let aspect = FRAME.0 as f32 / FRAME.1 as f32;
    let mut input = InputMap::default();
    input.resize(FRAME.0, FRAME.1, 1.0);
    let mut cursor = CursorCapture::new();
    let mut publication = Publication::default();
    loop {
        let frame = driver.frame();
        let step = u32::try_from(frame)
            .map_err(|_| HostError::Host(format!("frame {frame} is past the last step")))?;
        input.fix_clock(f64::from(step) * dt);
        session.views_mut().root_mut().eye.aspect = aspect;
        let gathered = input.take();
        let sender = app.commands.sender();
        if let Some(hook) = app.input.as_mut() {
            hook(&InputHook {
                session,
                input: &gathered,
                #[cfg(feature = "egui")]
                ui: None,
                size: FRAME,
                sender: &sender,
            });
        }
        let stepped = app.boundary(session, gathered).and_then(|_| {
            session.views_mut().root_mut().eye.aspect = aspect;
            session.tick().map_err(HostError::from)
        });
        input.reclaim(session.take_input());
        stepped?;
        session.views_mut().root_mut().eye.aspect = aspect;
        let views = session.views();
        cursor.follow_rig(views.get(views.root()).and_then(|image| image.rig.cursor()));
        if let Some(locked) = cursor.take_request() {
            cursor.applied(locked);
            input.set_cursor_locked(locked);
        }
        session.publish_at(&mut publication, alpha)?;
        each(step, session, &publication)?;

        let mut refused = None;
        let status = driver.advance_with(|line| {
            if refused.is_none() {
                refused = feed(&mut input, &app.config.bindings, &mut app.console, &line)
                    .err()
                    .map(|error| {
                        format!("`{}`: {error}", render_line(&line.name, &line.arg_refs()))
                    });
            }
        });
        if let Some(error) = refused {
            return Err(HostError::Host(format!("frame {frame}: {error}")));
        }
        app.console.dispatch_pending();
        echo(&mut app.console, frame)?;
        if status == ScriptStatus::Finished {
            return Ok(publication);
        }
    }
}

fn feed<A: Stores>(
    input: &mut InputMap,
    bindings: &Bindings,
    console: &mut SessionConsole<A>,
    line: &CommandLine,
) -> Result<(), String> {
    let args = line.arg_refs();
    match (line.name.as_str(), args.as_slice()) {
        ("key", [code, edge]) => {
            let key = dom_key(code).ok_or_else(|| format!("`{code}` is not a key code"))?;
            input.action(bindings, key, pressed(edge)?);
        }
        ("pointer", [x, y, "move"]) => input.moved([number(x)?, number(y)?]),
        ("pointer", [x, y, edge, button @ ..]) => {
            let button = match button {
                [] | ["primary"] => PointerButton::Primary,
                ["secondary"] => PointerButton::Secondary,
                ["middle"] => PointerButton::Middle,
                _ => return Err("the button is primary, secondary or middle".into()),
            };
            input.button([number(x)?, number(y)?], button, pressed(edge)?);
        }
        ("wheel", [lines]) => input.wheel([0.0, number(lines)?]),
        ("look", [dx, dy]) => input.raw_motion(number(dx)?, number(dy)?),
        ("key", _) => return Err("usage: key <code> down|up".into()),
        ("pointer", _) => return Err("usage: pointer <x> <y> down|up [button] | move".into()),
        ("wheel", _) => return Err("usage: wheel <lines>".into()),
        ("look", _) => return Err("usage: look <dx> <dy>".into()),
        _ => console.execute(&render_line(&line.name, &args)),
    }
    Ok(())
}

fn pressed(edge: &str) -> Result<bool, String> {
    match edge {
        "down" => Ok(true),
        "up" => Ok(false),
        _ => Err(format!("`{edge}` is not down or up")),
    }
}

fn number(token: &str) -> Result<f32, String> {
    token
        .parse::<f32>()
        .ok()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("`{token}` is not a finite number"))
}

// A console error line fails the run, so a mistyped verb cannot pass silently.
fn echo<A: Stores>(console: &mut SessionConsole<A>, frame: u64) -> Result<(), HostError> {
    let ui = console.ui_mut();
    let mut refused = None;
    for line in ui.history() {
        eprintln!("{}", line.text);
        if line.kind == LineKind::Error && refused.is_none() {
            refused = Some(line.text.clone());
        }
    }
    ui.clear_history();
    match refused {
        Some(text) => Err(HostError::Host(format!("frame {frame}: {text}"))),
        None => Ok(()),
    }
}
