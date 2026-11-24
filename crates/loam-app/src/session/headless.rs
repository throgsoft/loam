use loam_console::LineKind;
use loam_runtime::{Publication, Session, Stores};
use loam_time::par::Sequential;

use super::app::SessionApp;
use super::console::SessionConsole;
use super::cursor::CursorCapture;
use super::debug_layer::LayerSlot;
use super::frame::advance_session;
use super::input::InputMap;
use crate::host::HostError;
use crate::script::{ScriptDriver, ScriptRun, ScriptStatus};

// A 720p canvas, so pointer deltas and the eye aspect match a common windowed run.
const FRAME: (u32, u32) = (1280, 720);

/// Steps through `app`'s console and input until `driver` finishes, and a line runs after its frame publishes so it lands at the next boundary.
pub fn run_script<A: Stores>(
    session: &mut Session<A>,
    app: &mut SessionApp<A>,
    driver: ScriptDriver,
    alpha: f32,
    mut each: impl FnMut(u32, &mut Session<A>, &Publication<A::Published>) -> Result<(), HostError>,
) -> Result<Publication<A::Published>, HostError> {
    let mut script = ScriptRun::new(driver, &app.console)?;
    session.capture_initial();
    let dt = f64::from(session.config().dt().unwrap_or(0.0));
    let mut input = InputMap::default();
    input.resize(FRAME.0, FRAME.1, 1.0);
    let mut cursor = CursorCapture::new();
    let mut publication = Publication::default();
    loop {
        let frame = script.frame();
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
            |session, _| session.tick(&Sequential),
        )?;
        if let Some(locked) = cursor.take_request() {
            cursor.applied(locked);
            input.set_cursor_locked(locked);
        }
        session.publish(&mut publication, alpha)?;
        each(step, session, &publication)?;

        let advanced = script.advance(&mut input, &app.config.bindings, &mut app.console);
        let refused = echo(&mut app.console);
        let (status, _) = advanced?;
        if let Some(error) = refused {
            return Err(HostError::Host(format!("frame {frame}: {error}")));
        }
        if status == ScriptStatus::Finished {
            return Ok(publication);
        }
    }
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
