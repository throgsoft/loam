#![cfg(not(target_arch = "wasm32"))]

use loam_app::args::Args;
use loam_app::script::{Script, ScriptDriver};
use loam_app::session::{run_script, SessionApp};
use loam_runtime::{
    AppCommand, Bindings, Dispatch, HostConfig, Outcome, Rejection, Session, SimConfig,
};

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Slots {
        active: Value<u32>,
    }
}

const SLOTS: u32 = 3;

struct Pick(u32);

impl AppCommand<Slots> for Pick {
    fn name(&self) -> &'static str {
        "active"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Slots>) -> Result<Outcome, Rejection> {
        if self.0 >= SLOTS {
            return Err(Rejection::Unsupported("no such slot"));
        }
        dispatch.app.active.set(self.0);
        Ok(Outcome::Done)
    }
}

struct Run {
    outcome: Result<(), String>,
    frames: u32,
    active: u32,
}

fn run(source: &str) -> Run {
    let script = Script::parse(source).expect("the script parses");
    let mut session = Session::new(Slots::default(), SimConfig::default());
    let mut app =
        SessionApp::<Slots>::with_args(HostConfig::new("script", Bindings::new()), Args::default())
            .command("active", "pick a slot", |args, submit, _out| {
                let slot: u32 = args.first().copied().unwrap_or_default().parse()?;
                submit.app(Pick(slot));
                Ok(())
            });
    let mut frames = 0;
    let outcome = run_script(
        &mut session,
        &mut app,
        ScriptDriver::new(script),
        1.0,
        |_, _, _| {
            frames += 1;
            Ok(())
        },
    )
    .map(drop)
    .map_err(|error| error.to_string());
    Run {
        outcome,
        frames,
        active: *session.app.active.get(),
    }
}

#[test]
fn a_script_of_accepted_lines_runs_to_its_end_and_applies_them() {
    let ran = run("# pick twice\n0 active 1\n\n2 active 2\n");

    assert_eq!(ran.outcome, Ok(()));
    assert_eq!(ran.active, 2);
}

#[test]
fn a_command_the_session_rejects_fails_the_run_naming_its_script_line() {
    let ran = run("# the slots end at 2\n0 active 1\n\n1 active 9\n20 active 2\n");

    let error = ran.outcome.expect_err("the rejected command passed");
    assert!(
        error.contains("line 4"),
        "the error does not name the rejected line: {error}"
    );
    assert_eq!(ran.active, 1, "the run went on past the rejected line");
}

#[test]
fn a_console_handler_that_refuses_its_argument_fails_the_run_naming_its_script_line() {
    let ran = run("0 active 1\n# not a number\n1 active x\n");

    let error = ran.outcome.expect_err("the refused line passed");
    assert!(
        error.contains("line 3"),
        "the error does not name the refused line: {error}"
    );
}

#[test]
fn a_line_no_frame_could_run_fails_before_the_first_frame_naming_its_line() {
    for (source, line) in [
        ("0 active 1\n\n5 bogus\n", "line 3"),
        ("0 active 1\n5 key Nope down\n", "line 2"),
        ("0 active 1\n# drag\n\n5 pointer 1 x down\n", "line 4"),
    ] {
        let ran = run(source);

        let error = ran.outcome.expect_err(source);
        assert!(error.contains(line), "{source:?} failed with: {error}");
        assert_eq!(
            (ran.frames, ran.active),
            (0, 0),
            "{source:?} stepped before its bad line was found"
        );
    }
}
