use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::bail;
use loam_console::{cmd, CommandLine, Console, ConsoleWriter, HistoryLine, LineKind};
use loam_runtime::{
    AppCommand, Backdrop, CursorPolicy, Dispatch, FreeCamera, Outcome, Rejection, Rig, Setting,
    Stores, Views,
};

use super::commands::{CommandSender, Report, Reported};
use crate::environment::Environment;

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

    pub fn toggle(&mut self, name: &'static str, setting: Setting<A>) {
        self.commands.toggle(name, setting);
    }

    pub fn set(&mut self, name: &'static str, setting: Setting<A>, on: bool) {
        self.commands.set(name, setting, on);
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
    refused: Option<(usize, String)>,
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
            refused: None,
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
        register_ground(&mut console.console);
        register_floor(&mut console.console);
        register_camera(&mut console.console);
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

    /// Registers a verb that flips `setting` when bare and sets it on `on` or `off`, each as a command named `name`.
    pub fn toggle_command(&mut self, name: &'static str, help: &'static str, setting: Setting<A>) {
        self.console.register(
            cmd(name, help, move |args, submit: &mut Submit<A>, _out| {
                match args {
                    [] => submit.toggle(name, setting),
                    ["on"] => submit.set(name, setting, true),
                    ["off"] => submit.set(name, setting, false),
                    _ => anyhow::bail!("usage: {name} [on|off]"),
                }
                Ok(())
            })
            .with_args(&[&["on", "off"]]),
        );
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

    pub(crate) fn note(&mut self, line: String) {
        self.console.write(HistoryLine::output(line));
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
                Ok(outcome) => format!("{}: {outcome}", reported.name),
                Err(rejection) => {
                    let line = format!("{}: rejected, {rejection}", reported.name);
                    if let (Report::Script(number), None) = (reported.report, &self.refused) {
                        self.refused = Some((number, line.clone()));
                    }
                    line
                }
            };
            self.console.write(HistoryLine::output(line));
        }
        if dropped > 0 {
            self.console.write(HistoryLine::output(format!(
                "commands: {dropped} results omitted"
            )));
        }
    }

    // A refused dispatch ends on its error line.
    pub(crate) fn dispatch_script_line(&mut self, text: &str, number: usize) -> Option<String> {
        self.submit.commands.set_report(Report::Script(number));
        self.console.execute(text);
        self.dispatch_pending();
        self.submit.commands.set_report(Report::Console);
        self.console
            .history()
            .back()
            .filter(|line| line.kind == LineKind::Error)
            .map(|line| line.text.clone())
    }

    pub(crate) fn take_refused(&mut self) -> Option<(usize, String)> {
        self.refused.take()
    }

    pub fn dispatch_pending(&mut self) -> bool {
        let lines = self.console.drain_pending();
        let dispatched = !lines.is_empty();
        for line in lines {
            let Some(parsed) = CommandLine::parse(&line) else {
                continue;
            };
            self.console
                .dispatch(&parsed.name, &parsed.arg_refs(), &mut self.submit);
        }
        dispatched
    }
}

const MAX_ACCEPTED_FPS: f32 = 1000.0;

const NO_BACKDROP: &str = "this scene has no backdrop";
const NO_FREE_CAMERA: &str = "this scene has no free camera";
const NO_CAMERA_PAIR: &str = "this scene has no orbit and free camera pair";

fn root_backdrop(views: &Views) -> Option<Backdrop> {
    views.root().backdrop
}

fn root_free_camera(views: &Views) -> Option<&FreeCamera> {
    views.root().rig.free()
}

fn register_ground<A: Stores>(console: &mut Console<Submit<A>>) {
    console.register(
        cmd(
            "ground",
            "read or set checker colors and fog: dark | light | fog | reset",
            |args, submit: &mut Submit<A>, _out| {
                let field = match args {
                    [] => Some(None),
                    ["dark"] => Some(Some("dark")),
                    ["light"] => Some(Some("light")),
                    ["fog"] => Some(Some("fog")),
                    _ => None,
                };
                if let Some(field) = field {
                    submit.inspect("ground", move |dispatch, out| {
                        match root_backdrop(dispatch.views) {
                            Some(backdrop) => out.line(Environment::of(&backdrop).report(field)),
                            None => out.line(format!("ground: {NO_BACKDROP}")),
                        }
                    });
                    return Ok(());
                }
                Environment::default().apply(args)?;
                let owned = args
                    .iter()
                    .map(|arg| (*arg).to_string())
                    .collect::<Vec<_>>();
                let console = submit.console.clone();
                submit.commands.try_app_fn("ground", move |dispatch| {
                    let backdrop = dispatch
                        .views
                        .root_mut()
                        .backdrop
                        .as_mut()
                        .ok_or(Rejection::Unsupported(NO_BACKDROP))?;
                    let args = owned.iter().map(String::as_str).collect::<Vec<_>>();
                    let mut environment = Environment::of(backdrop);
                    let line = environment
                        .apply(&args)
                        .expect("the console accepted these arguments before queueing them");
                    *backdrop = environment.backdrop(backdrop.ground.y);
                    lock(&console).responses.push(HistoryLine::output(line));
                    Ok(Outcome::Done)
                });
                Ok(())
            },
        )
        .with_args(&[&["dark", "light", "fog", "reset"]])
        .with_long_help(
            "ground                     read all three\n\
             ground dark                read the dark checker color\n\
             ground dark <r> <g> <b>    set it, each channel in [0, 1]\n\
             ground light <r> <g> <b>   the light checker color\n\
             ground fog                 read the density\n\
             ground fog <density>       sky blended per world unit, in [0, 1];\n\
             \x20                          half the sky is mixed in at ln2/density\n\
             ground reset               back to the shipped defaults",
        ),
    );
}

fn register_floor<A: Stores>(console: &mut Console<Submit<A>>) {
    console.register(
        cmd(
            "floor",
            "toggle the ground plane; optional on | off",
            |args, submit: &mut Submit<A>, _out| {
                let shown = match args {
                    [] => None,
                    ["on"] => Some(true),
                    ["off"] => Some(false),
                    _ => bail!("usage: floor [on|off]"),
                };
                submit.commands.try_app_fn("floor", move |dispatch| {
                    let backdrop = dispatch
                        .views
                        .root_mut()
                        .backdrop
                        .as_mut()
                        .ok_or(Rejection::Unsupported(NO_BACKDROP))?;
                    let ground = &mut backdrop.ground;
                    ground.visible = shown.unwrap_or(!ground.visible);
                    Ok(Outcome::Done)
                });
                Ok(())
            },
        )
        .with_args(&[&["on", "off"]]),
    );
}

fn register_camera<A: Stores>(console: &mut Console<Submit<A>>) {
    console.register(
        cmd(
            "camera",
            "camera [orbit|freecam|speed <N>|cursor <hold|toggle>]; bare switches",
            |args, submit: &mut Submit<A>, _out| {
                match args {
                    [] | ["orbit"] | ["freecam"] => {
                        let flying = args.first().map(|arg| *arg == "freecam");
                        submit.commands.try_app_fn("camera", move |dispatch| {
                            let root = dispatch.views.root_mut();
                            if !matches!(root.rig, Rig::Pair(_)) {
                                return Err(Rejection::Unsupported(NO_CAMERA_PAIR));
                            }
                            let flying = flying.unwrap_or(!root.rig.flying());
                            root.set_flying(flying);
                            Ok(Outcome::Done)
                        });
                    }
                    ["speed"] => submit.inspect("camera speed", |dispatch, out| {
                        match root_free_camera(dispatch.views) {
                            Some(free) => {
                                out.line(format!("camera speed: {:.2} u/sec", free.speed))
                            }
                            None => out.line(format!("camera speed: {NO_FREE_CAMERA}")),
                        }
                    }),
                    ["speed", token] => {
                        let speed = token.parse::<f32>().map_err(|error| {
                            anyhow::anyhow!("camera speed: invalid `{token}`: {error}")
                        })?;
                        if !speed.is_finite() || speed <= 0.0 {
                            bail!("camera speed must be finite and positive");
                        }
                        submit.commands.try_app_fn("camera speed", move |dispatch| {
                            free_camera_mut(dispatch.views)?.speed = speed;
                            Ok(Outcome::Done)
                        });
                    }
                    ["cursor"] => submit.inspect("camera cursor", |dispatch, out| {
                        match root_free_camera(dispatch.views) {
                            Some(free) => {
                                out.line(format!("camera cursor: {}", cursor_name(free.cursor)))
                            }
                            None => out.line(format!("camera cursor: {NO_FREE_CAMERA}")),
                        }
                    }),
                    ["cursor", "hold"] | ["cursor", "toggle"] => {
                        let policy = if args[1] == "hold" {
                            CursorPolicy::Hold
                        } else {
                            CursorPolicy::Toggle
                        };
                        submit
                            .commands
                            .try_app_fn("camera cursor", move |dispatch| {
                                free_camera_mut(dispatch.views)?.cursor = policy;
                                Ok(Outcome::Done)
                            });
                    }
                    _ => bail!("camera: unknown argument (try orbit|freecam|speed|cursor)"),
                }
                Ok(())
            },
        )
        .with_args(&[
            &["orbit", "freecam", "speed", "cursor"],
            &["hold", "toggle"],
        ]),
    );
}

fn free_camera_mut(views: &mut Views) -> Result<&mut FreeCamera, Rejection> {
    views
        .root_mut()
        .rig
        .free_mut()
        .ok_or(Rejection::Unsupported(NO_FREE_CAMERA))
}

fn cursor_name(policy: CursorPolicy) -> &'static str {
    match policy {
        CursorPolicy::Hold => "hold",
        CursorPolicy::Toggle => "toggle",
    }
}

#[cfg(test)]
mod tests {
    use loam_runtime::{
        Ctx, DomainError, Input, Phase, Rejection, Session, SimConfig, SpawnBundle,
    };

    use super::*;
    use crate::args::Args;
    use crate::host::{Bindings, HostConfig};
    use crate::session::SessionApp;

    loam_runtime::stores! {
        #[derive(Default)]
        pub struct Counted {
            hits: Value<u32>,
        }
    }

    #[test]
    fn session_result() {
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
                .any(|line| line.contains("done")),
            "the result cannot exist before the boundary that applies the command"
        );

        app.boundary(&mut session, Input::default())
            .expect("boundary");

        assert_eq!(*session.app.hits.get(), 1);
        assert!(history(&app.console).iter().any(|line| line == "count=1"));
        assert!(
            history(&app.console)
                .iter()
                .any(|line| line == "bump: done"),
            "the console never matched its request to the session result: {:?}",
            history(&app.console)
        );
    }

    #[test]
    fn refused_command() {
        let mut session = Session::new(Counted::default(), SimConfig::default());
        session.system(Phase::Dispatch, "keys", |ctx: Ctx<'_, Counted>| {
            ctx.commands
                .try_app_fn("strip", |_d: &mut Dispatch<'_, Counted>| {
                    Err(Rejection::Unsupported("filmstrip belongs to Rotate"))
                });
            Ok(())
        });
        let mut app =
            SessionApp::with_args(HostConfig::new("console", Bindings::new()), Args::default());
        let entity = session
            .dispatch(|d| d.spawn(SpawnBundle::new()))
            .expect("a bare entity");
        app.sender()
            .try_app_fn("refused", |_d: &mut Dispatch<'_, Counted>| {
                Err(Rejection::Unsupported("not now"))
            });
        app.sender()
            .try_app_fn("nudge", move |_d: &mut Dispatch<'_, Counted>| {
                Err(Rejection::Domain(DomainError::Stale(entity)))
            });
        app.sender()
            .app_fn("accepted", |d: &mut Dispatch<'_, Counted>| {
                *d.app.hits.get_mut() += 1;
            });

        app.boundary(&mut session, Input::default())
            .expect("boundary");

        let lines = history(&app.console);
        for expected in [
            "refused: rejected, unsupported: not now".to_string(),
            format!("nudge: rejected, {entity} is stale"),
            "strip: rejected, unsupported: filmstrip belongs to Rotate".to_string(),
        ] {
            assert!(
                lines.contains(&expected),
                "`{expected}` never reached the console: {lines:?}"
            );
        }
        assert!(
            !lines.iter().any(|line| line.contains("Stale(")),
            "a refusal reached the console as Debug text: {lines:?}"
        );
        assert!(
            !lines.iter().any(|line| line.starts_with("accepted")),
            "an unreported success reached the console: {lines:?}"
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
