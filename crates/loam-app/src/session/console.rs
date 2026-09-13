use std::sync::{Arc, Mutex, MutexGuard};

use loam_egui::{cmd, Console, ConsoleWriter, HistoryLine};
use loam_runtime::{AppCommand, Dispatch, Stores};

use super::commands::{CommandSender, Reported};
use crate::command::CommandLine;

#[derive(Default)]
struct ConsoleState {
    controls: Controls,
    responses: Vec<HistoryLine>,
}

type Shared = Arc<Mutex<ConsoleState>>;

fn lock(shared: &Shared) -> MutexGuard<'_, ConsoleState> {
    shared.lock().unwrap_or_else(|error| error.into_inner())
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Controls {
    pub target_fps: Option<f32>,
    pub vsync: Option<bool>,
}

pub struct Submit<A: Stores> {
    commands: CommandSender<A>,
    console: Shared,
}

impl<A: Stores> Submit<A> {
    /// Reads committed state at the command boundary and returns its text to the console.
    pub fn inspect(
        &mut self,
        name: &'static str,
        read: impl FnMut(&Dispatch<'_, A>, &mut ConsoleWriter) + Send + 'static,
    ) {
        self.commands.app(Inspect {
            name,
            read,
            console: self.console.clone(),
        });
    }

    pub fn app(&mut self, command: impl AppCommand<A>) {
        self.commands.app(command);
    }

    pub fn app_fn(
        &mut self,
        name: &'static str,
        apply: impl FnMut(&mut Dispatch<'_, A>) + Send + 'static,
    ) {
        self.commands.app_fn(name, apply);
    }

    pub fn reset(&mut self) {
        self.commands.reset();
    }

    pub fn target_fps(&mut self, fps: f32) {
        lock(&self.console).controls.target_fps = Some(fps);
    }

    pub fn vsync(&mut self, enabled: bool) {
        lock(&self.console).controls.vsync = Some(enabled);
    }
}

struct Inspect<F> {
    name: &'static str,
    read: F,
    console: Shared,
}

impl<A: Stores, F> AppCommand<A> for Inspect<F>
where
    F: FnMut(&Dispatch<'_, A>, &mut ConsoleWriter) + Send + 'static,
{
    fn name(&self) -> &'static str {
        self.name
    }

    fn apply(
        &mut self,
        dispatch: &mut Dispatch<'_, A>,
    ) -> Result<loam_runtime::Outcome, loam_runtime::Rejection> {
        let mut output = ConsoleWriter::new();
        (self.read)(dispatch, &mut output);
        lock(&self.console).responses.extend(output.take_lines());
        Ok(loam_runtime::Outcome::Done)
    }
}

pub struct SessionConsole<A: Stores> {
    console: Console<Submit<A>>,
    submit: Submit<A>,
    state: Shared,
    completed: Vec<Reported>,
}

impl<A: Stores> SessionConsole<A> {
    pub(crate) fn new(sender: CommandSender<A>) -> Self {
        let state = Shared::default();
        let mut console = Self {
            console: Console::new(),
            submit: Submit {
                commands: sender.reported(),
                console: state.clone(),
            },
            state,
            completed: Vec::new(),
        };
        crate::trace::register_command(&mut console.console);
        console.register(
            "recover",
            "restore the initial session after a fault",
            |args, submit: &mut Submit<A>, out| {
                if args.is_empty() {
                    submit.reset();
                } else {
                    out.line("usage: recover");
                }
                Ok(())
            },
        );
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
        std::mem::take(&mut lock(&self.state).controls)
    }

    pub fn execute(&mut self, line: &str) {
        self.console.execute(line);
    }

    pub fn submit(&mut self, command: impl AppCommand<A>) {
        self.submit.app(command);
    }

    pub(crate) fn collect(&mut self) {
        let mut state = lock(&self.state);
        for line in state.responses.drain(..) {
            self.console.write(line);
        }
        drop(state);
        let dropped = self.submit.commands.take_reported(&mut self.completed);
        for reported in self.completed.drain(..) {
            let line = match reported.outcome {
                Ok(outcome) => format!("{}: {outcome:?}", reported.name),
                Err(rejection) => format!("{}: rejected, {rejection:?}", reported.name),
            };
            self.console.write(HistoryLine::output(line));
        }
        if dropped > 0 {
            self.console.write(HistoryLine::output(format!(
                "commands: {dropped} results omitted"
            )));
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

const MAX_ACCEPTED_FPS: f32 = 1000.0;

#[cfg(test)]
mod tests {
    use loam_runtime::{Bindings, HostConfig, Input, Session, SimConfig};

    use super::*;
    use crate::args::Args;
    use crate::session::SessionApp;

    loam_runtime::stores! {
        #[derive(Default)]
        pub struct Counted {
            hits: Value<u32>,
        }
    }

    #[test]
    fn a_console_commands_session_result_reaches_the_console_at_the_next_boundary() {
        let mut session = Session::new(Counted::default(), SimConfig::default());
        let mut app =
            SessionApp::with_args(HostConfig::new("console", Bindings::new()), Args::default());
        app.console
            .register("bump", "raise the counter", |_args, submit, out| {
                submit.app_fn("bump", |d: &mut Dispatch<'_, Counted>| {
                    *d.app.hits.get_mut() += 1;
                });
                out.line("queued");
                Ok(())
            });
        app.console
            .register("count", "read the counter", |_args, submit, _out| {
                submit.inspect("count", |dispatch, out| {
                    out.line(format!("count={}", dispatch.app.hits.get()));
                });
                Ok(())
            });
        app.console.execute("bump");
        app.console.execute("count");
        app.console.dispatch_pending();
        assert!(
            !history(&app.console)
                .iter()
                .any(|line| line.contains("Done")),
            "the result cannot exist before the boundary that applies the command"
        );

        app.boundary(&mut session, Input::default())
            .expect("boundary");

        assert_eq!(*session.app.hits.get(), 1);
        assert!(history(&app.console).iter().any(|line| line == "count=1"));
        assert!(
            history(&app.console)
                .iter()
                .any(|line| line == "bump: Done"),
            "the console never matched its request to the session result: {:?}",
            history(&app.console)
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
