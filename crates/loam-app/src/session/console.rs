use std::sync::{Arc, Mutex};

use loam_egui::{cmd, Console, ConsoleWriter, HistoryLine};
use loam_runtime::{
    Access, AppCommand, Command, Commands, Dispatch, Input, Phase, RequestId, Session, Stores,
};

use crate::command::CommandLine;

enum Queued<A> {
    App(Box<dyn AppCommand<A>>),
    Reset,
}

impl<A: 'static> Queued<A> {
    fn name(&self) -> &'static str {
        match self {
            Queued::App(command) => command.name(),
            Queued::Reset => "reset",
        }
    }
}

struct Inbox<A> {
    queued: Vec<Queued<A>>,
    submitted: Vec<(RequestId, &'static str)>,
    controls: Controls,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Controls {
    pub target_fps: Option<f32>,
    pub vsync: Option<bool>,
}

impl<A> Default for Inbox<A> {
    fn default() -> Self {
        Self {
            queued: Vec::new(),
            submitted: Vec::new(),
            controls: Controls::default(),
        }
    }
}

type Shared<A> = Arc<Mutex<Inbox<A>>>;

fn lock<A>(shared: &Shared<A>) -> std::sync::MutexGuard<'_, Inbox<A>> {
    shared.lock().unwrap_or_else(|error| error.into_inner())
}

/// What a console verb may ask of the session: an app command, a reset, a frame cap, or vsync.
pub struct Submit<A: Stores> {
    inbox: Shared<A>,
}

impl<A: Stores> Submit<A> {
    pub fn app(&mut self, command: impl AppCommand<A>) {
        lock(&self.inbox)
            .queued
            .push(Queued::App(Box::new(command)));
    }

    pub fn app_fn(
        &mut self,
        name: &'static str,
        apply: impl FnMut(&mut Dispatch<'_, A>) + Send + 'static,
    ) {
        self.app(FnCommand { name, apply });
    }

    pub fn reset(&mut self) {
        lock(&self.inbox).queued.push(Queued::Reset);
    }

    pub fn target_fps(&mut self, fps: f32) {
        lock(&self.inbox).controls.target_fps = Some(fps);
    }

    pub fn vsync(&mut self, enabled: bool) {
        lock(&self.inbox).controls.vsync = Some(enabled);
    }
}

struct FnCommand<F> {
    name: &'static str,
    apply: F,
}

impl<A, F> AppCommand<A> for FnCommand<F>
where
    F: FnMut(&mut Dispatch<'_, A>) + Send + 'static,
{
    fn name(&self) -> &'static str {
        self.name
    }

    fn apply(
        &mut self,
        dispatch: &mut Dispatch<'_, A>,
    ) -> Result<loam_runtime::Outcome, loam_runtime::Rejection> {
        (self.apply)(dispatch);
        Ok(loam_runtime::Outcome::Done)
    }
}

/// The egui console bound to a session: verbs queue through a Dispatch entry, and a command's result reaches the history at the next boundary.
pub struct SessionConsole<A: Stores> {
    console: Console<Submit<A>>,
    submit: Submit<A>,
    inbox: Shared<A>,
    lines: Vec<String>,
}

impl<A: Stores> Default for SessionConsole<A> {
    fn default() -> Self {
        let inbox = Shared::default();
        let mut console = Self {
            console: Console::new(),
            submit: Submit {
                inbox: inbox.clone(),
            },
            inbox,
            lines: Vec::new(),
        };
        crate::trace::register_command(&mut console.console);
        console.register(
            "fps",
            "show or set the frame cap; unlimited removes it",
            |args, submit: &mut Submit<A>, out| {
                match args.first().copied() {
                    Some("unlimited" | "off" | "0") => {
                        submit.target_fps(0.0);
                        out.line("fps: unlimited; applies on the next frame");
                    }
                    Some(other) => match other.parse::<f32>() {
                        Ok(fps) if fps > 0.0 && fps <= MAX_ACCEPTED_FPS => {
                            submit.target_fps(fps);
                            out.line(format!("fps: target {fps:.1}; applies on the next frame"));
                        }
                        _ => out.line(format!(
                            "usage: fps <n> | unlimited  (n in (0, {MAX_ACCEPTED_FPS:.0}])"
                        )),
                    },
                    None => out.line("usage: fps <n> | unlimited"),
                }
                Ok(())
            },
        );
        console.register(
            "vsync",
            "set the surface present mode (on = Fifo, off = Mailbox or Immediate)",
            |args, submit: &mut Submit<A>, out| {
                match args.first().copied() {
                    Some("on") => {
                        submit.vsync(true);
                        out.line("vsync: Fifo requested; applies on the next frame");
                    }
                    Some("off") => {
                        submit.vsync(false);
                        out.line(
                            "vsync: Mailbox or Immediate requested; applies on the next frame",
                        );
                    }
                    _ => out.line("usage: vsync on | off"),
                }
                Ok(())
            },
        );
        console
    }
}

const MAX_ACCEPTED_FPS: f32 = 1000.0;

impl<A: Stores> SessionConsole<A> {
    pub fn register(
        &mut self,
        name: &'static str,
        help: &'static str,
        mut handler: impl FnMut(&[&str], &mut Submit<A>, &mut ConsoleWriter) -> anyhow::Result<()>
            + 'static,
    ) {
        self.console
            .register(cmd(name, help, move |args, submit: &mut Submit<A>, out| {
                handler(args, submit, out)
            }));
    }

    pub fn has(&self, name: &str) -> bool {
        self.console.has_command(name)
    }

    pub fn ui(&self) -> &Console<Submit<A>> {
        &self.console
    }

    pub fn ui_mut(&mut self) -> &mut Console<Submit<A>> {
        &mut self.console
    }

    pub fn take_controls(&mut self) -> Controls {
        std::mem::take(&mut lock(&self.inbox).controls)
    }

    pub fn execute(&mut self, line: &str) {
        self.console.execute(line);
    }

    pub fn submit(&mut self, command: impl AppCommand<A>) {
        self.submit.app(command);
    }

    /// Registers the Dispatch entry that submits queued commands; once per session.
    pub fn install(&self, session: &mut Session<A>) {
        let inbox = self.inbox.clone();
        let mut scratch: Vec<Queued<A>> = Vec::new();
        session.system(
            Phase::Dispatch,
            "loam-app::console",
            Access::new().commands(),
            move |_input: &Input, commands: &mut Commands<A>| {
                {
                    let mut held = lock(&inbox);
                    if held.queued.is_empty() {
                        return;
                    }
                    std::mem::swap(&mut held.queued, &mut scratch);
                }
                for queued in scratch.drain(..) {
                    let name = queued.name();
                    let request = match queued {
                        Queued::App(command) => commands.submit(Command::App(command)),
                        Queued::Reset => commands.submit(Command::Reset),
                    };
                    lock(&inbox).submitted.push((request, name));
                }
            },
        );
    }

    /// Matches this boundary's results to the requests the console submitted and writes them to the history.
    pub fn collect(&mut self, session: &Session<A>) {
        let mut inbox = lock(&self.inbox);
        if inbox.submitted.is_empty() {
            return;
        }
        for result in session.results() {
            let Some(index) = inbox
                .submitted
                .iter()
                .position(|(request, _)| *request == result.request)
            else {
                continue;
            };
            let (_, name) = inbox.submitted.swap_remove(index);
            self.lines.push(match &result.outcome {
                Ok(outcome) => format!("{name}: {outcome:?}"),
                Err(rejection) => format!("{name}: rejected, {rejection:?}"),
            });
        }
        drop(inbox);
        for line in self.lines.drain(..) {
            self.console.write(HistoryLine::output(line));
        }
    }

    pub fn dispatch_pending(&mut self) {
        for line in self.console.drain_pending() {
            let Some(parsed) = CommandLine::parse(&line) else {
                continue;
            };
            self.console
                .dispatch(&parsed.name, &parsed.arg_refs(), &mut self.submit);
        }
    }
}

#[cfg(test)]
mod tests {
    use loam_runtime::SimConfig;

    use super::*;

    loam_runtime::stores! {
        #[derive(Default)]
        pub struct Counted {
            hits: Value<u32>,
        }
    }

    #[test]
    fn a_console_commands_session_result_reaches_the_console_at_the_next_boundary() {
        let mut session = Session::new(Counted::default(), SimConfig::default());
        let mut console = SessionConsole::<Counted>::default();
        console.register("bump", "raise the counter", |_args, submit, out| {
            submit.app_fn("bump", |d: &mut Dispatch<'_, Counted>| {
                *d.app.hits.get_mut() += 1;
            });
            out.line("queued");
            Ok(())
        });
        console.install(&mut session);

        console.execute("bump");
        console.dispatch_pending();
        console.collect(&session);
        assert!(
            !history(&console).iter().any(|line| line.contains("Done")),
            "the result cannot exist before the boundary that applies the command"
        );

        session.boundary(Input::default()).expect("boundary");
        console.collect(&session);

        assert_eq!(*session.app.hits.get(), 1);
        assert!(
            history(&console).iter().any(|line| line == "bump: Done"),
            "the console never matched its request to the session result: {:?}",
            history(&console)
        );
    }

    fn history(console: &SessionConsole<Counted>) -> Vec<String> {
        console
            .ui()
            .history()
            .iter()
            .map(|line| line.text.clone())
            .collect()
    }
}
