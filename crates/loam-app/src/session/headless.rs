use loam_console::{render_line, CommandLine, LineKind};
use loam_runtime::host::HostError;
use loam_runtime::{Bindings, Key, PointerButton, Publication, Session, Stores};

use super::app::SessionApp;
use super::console::SessionConsole;
use super::cursor::CursorCapture;
use super::debug_layer::LayerSlot;
use super::frame::advance_session;
use super::input::{dom_key, InputMap};
use crate::script::{ScriptDriver, ScriptStatus, ScriptStep};

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
    let lines = driver
        .steps()
        .iter()
        .map(|step| parse(&app.console, &step.command).map_err(|error| refused(step, &error)))
        .collect::<Result<Vec<_>, _>>()?;
    session.capture_initial();
    let dt = f64::from(session.config().dt().unwrap_or(0.0));
    let mut input = InputMap::default();
    input.resize(FRAME.0, FRAME.1, 1.0);
    let mut cursor = CursorCapture::new();
    let mut publication = Publication::default();
    loop {
        let frame = driver.frame();
        let step = u32::try_from(frame)
            .map_err(|_| HostError::Host(format!("frame {frame} is past the last step")))?;
        input.fix_clock(f64::from(step) * dt);
        advance_session(
            session,
            app,
            &mut input,
            &mut cursor,
            FRAME,
            &LayerSlot::default(),
            false,
            |session, _| session.tick(),
        )?;
        if let Some((number, error)) = app.console.take_refused() {
            return Err(HostError::Host(format!("script line {number}: {error}")));
        }
        if let Some(locked) = cursor.take_request() {
            cursor.applied(locked);
            input.set_cursor_locked(locked);
        }
        session.publish_at(&mut publication, alpha)?;
        each(step, session, &publication)?;

        let mut failed = None;
        let status = driver.advance_with(|index, step| {
            if failed.is_some() {
                return;
            }
            match &lines[index] {
                Line::Input(event) => event.apply(&mut input, &app.config.bindings),
                Line::Console => {
                    let text = render_line(&step.command.name, &step.command.arg_refs());
                    app.console.dispatch_script_line(&text, step.line);
                    failed = echo(&mut app.console).map(|error| refused(step, &error));
                }
            }
        });
        if let Some(error) = failed {
            return Err(error);
        }
        if let Some(error) = echo(&mut app.console) {
            return Err(HostError::Host(format!("frame {frame}: {error}")));
        }
        if status == ScriptStatus::Finished {
            return Ok(publication);
        }
    }
}

enum Line {
    Input(Event),
    Console,
}

#[derive(Clone, Copy)]
enum Event {
    Key(Key, bool),
    Move([f32; 2]),
    Button([f32; 2], PointerButton, bool),
    Wheel(f32),
    Look(f32, f32),
}

impl Event {
    fn apply(self, input: &mut InputMap, bindings: &Bindings) {
        match self {
            Event::Key(key, down) => input.action(bindings, key, down),
            Event::Move(at) => input.moved(at),
            Event::Button(at, button, down) => input.button(at, button, down),
            Event::Wheel(lines) => input.wheel([0.0, lines]),
            Event::Look(dx, dy) => input.raw_motion(dx, dy),
        }
    }
}

fn parse<A: Stores>(console: &SessionConsole<A>, line: &CommandLine) -> Result<Line, String> {
    let args = line.arg_refs();
    let event = match (line.name.as_str(), args.as_slice()) {
        ("key", [code, edge]) => Event::Key(
            dom_key(code).ok_or_else(|| format!("`{code}` is not a key code"))?,
            pressed(edge)?,
        ),
        ("pointer", [x, y, "move"]) => Event::Move([number(x)?, number(y)?]),
        ("pointer", [x, y, edge, button @ ..]) => {
            let button = match button {
                [] | ["primary"] => PointerButton::Primary,
                ["secondary"] => PointerButton::Secondary,
                ["middle"] => PointerButton::Middle,
                _ => return Err("the button is primary, secondary or middle".into()),
            };
            Event::Button([number(x)?, number(y)?], button, pressed(edge)?)
        }
        ("wheel", [lines]) => Event::Wheel(number(lines)?),
        ("look", [dx, dy]) => Event::Look(number(dx)?, number(dy)?),
        ("key", _) => return Err("usage: key <code> down|up".into()),
        ("pointer", _) => return Err("usage: pointer <x> <y> down|up [button] | move".into()),
        ("wheel", _) => return Err("usage: wheel <lines>".into()),
        ("look", _) => return Err("usage: look <dx> <dy>".into()),
        (name, _) if console.has(name) => return Ok(Line::Console),
        (name, _) => return Err(format!("no command '{name}'")),
    };
    Ok(Line::Input(event))
}

fn refused(step: &ScriptStep, error: &str) -> HostError {
    let text = render_line(&step.command.name, &step.command.arg_refs());
    HostError::Host(format!("script line {}: `{text}`: {error}", step.line))
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

// Echoes the console to stderr and returns its first error line, so a refused verb cannot pass silently.
fn echo<A: Stores>(console: &mut SessionConsole<A>) -> Option<String> {
    let ui = console.ui_mut();
    let mut refused = None;
    for line in ui.history() {
        eprintln!("{}", line.text);
        if line.kind == LineKind::Error && refused.is_none() {
            refused = Some(line.text.clone());
        }
    }
    ui.clear_history();
    refused
}
