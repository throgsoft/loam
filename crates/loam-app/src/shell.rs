//! A registry may mix spaces; scene lifecycle does not constrain geometry.

use std::borrow::Cow;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use anyhow::{anyhow, bail, Result};
use loam_egui::{cmd, Console, ConsoleWriter};
use loam_render::device::RenderDevice;

use crate::args::Args;
use crate::command::{CommandCtx, CommandLine};
use crate::{egui, App, FrameCtx, RenderCtx, Runtime, SetupCtx, ShaderDb, TickCtx};

pub trait Scene {
    fn tick(&mut self, _dt: f32, _ctx: &mut TickCtx) {}

    /// Active and cached scenes share the runner's shader database.
    fn apply_shader_events(&mut self, _events: &[std::path::PathBuf], _shader_db: &mut ShaderDb) {}
    fn apply_command(
        &mut self,
        cmd: &CommandLine,
        _ctx: &mut CommandCtx<'_>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("no command target for `{}`", cmd.name)
    }
    /// Contributions to the shared menu bar, rendered after the Demo menu.
    fn menus(&mut self, _ui: &mut egui::Ui) {}
    fn update(&mut self, ctx: &mut FrameCtx<'_>);
    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>);
    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    );
    /// Must not submit; see `RenderCtx`.
    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> Result<()>;
    fn title(&self, fps: f32) -> Cow<'static, str>;
    /// `Some(reason)` makes an unforced restart ask first.
    fn unsaved_work(&self) -> Option<Cow<'static, str>> {
        None
    }
}

pub struct SceneEntry {
    pub slug: &'static str,
    pub label: &'static str,
    pub build: fn(&mut SetupCtx<'_>, &SceneControl) -> Result<Box<dyn Scene>>,
}

pub trait SceneRegistry: 'static {
    /// Non-empty; index 0 is the boot fallback.
    const SCENES: &'static [SceneEntry];
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Request {
    Switch(usize),
    Restart { forced: bool },
}

struct Switcher {
    active: usize,
    pending: Option<Request>,
}

#[derive(Clone)]
pub struct SceneControl(Rc<RefCell<Switcher>>);

impl SceneControl {
    fn new(active: usize) -> Self {
        Self(Rc::new(RefCell::new(Switcher {
            active,
            pending: None,
        })))
    }

    fn queue(&self, request: Request) {
        self.0.borrow_mut().pending = Some(request);
    }

    fn active(&self) -> usize {
        self.0.borrow().active
    }
}

fn scene_index(scenes: &[SceneEntry], slug: &str) -> Option<usize> {
    scenes.iter().position(|entry| entry.slug == slug)
}

// Applied after the current frame's scene `ui` returns.
fn request_scene(control: &SceneControl, scenes: &[SceneEntry], slug: &str) -> Result<()> {
    let index = scene_index(scenes, slug).ok_or_else(|| {
        let known = scenes
            .iter()
            .map(|entry| entry.slug)
            .collect::<Vec<_>>()
            .join("|");
        anyhow!("unknown scene `{slug}` (try {known})")
    })?;
    control.queue(Request::Switch(index));
    Ok(())
}

fn run_restart(control: &SceneControl, args: &[&str], out: &mut ConsoleWriter) -> Result<()> {
    let forced = match args {
        [] => false,
        ["force"] => true,
        _ => bail!("usage: restart [force]"),
    };
    control.queue(Request::Restart { forced });
    out.line(if forced {
        "restart: rebuilding the active scene"
    } else {
        "restart: rebuilding the active scene unless it reports unsaved work"
    });
    Ok(())
}

fn claims_restart(
    code: winit::keyboard::KeyCode,
    state: winit::event::ElementState,
    ui_captures_keyboard: bool,
) -> bool {
    code == winit::keyboard::KeyCode::KeyR
        && state == winit::event::ElementState::Pressed
        && !ui_captures_keyboard
}

/// Fill with [`crate::build_info`]: the `env!` must expand in the demo's crate.
pub struct BuildInfo {
    pub crate_name: &'static str,
    pub crate_version: &'static str,
    pub build_hash: &'static str,
    pub build_dirty: &'static str,
}

pub fn register_shell_commands<Ctx: 'static, R: SceneRegistry>(
    console: &mut Console<Ctx>,
    build: BuildInfo,
    runtime: &Runtime,
    control: &SceneControl,
) {
    register_scene_commands::<Ctx, R>(console, control);
    crate::capture::register_commands(console, runtime);
    crate::capture::bind_default_hotkeys(console);
    crate::log::register_command(console);
    crate::trace::register_command(console);
    crate::fps::register_command(console, runtime);
    crate::vsync::register_command(console, runtime);
    crate::version::register_command(
        console,
        build.crate_name,
        build.crate_version,
        build.build_hash,
        build.build_dirty,
    );
}

fn register_scene_commands<Ctx: 'static, R: SceneRegistry>(
    console: &mut Console<Ctx>,
    control: &SceneControl,
) {
    let slugs = R::SCENES.iter().map(|entry| entry.slug).collect::<Vec<_>>();
    let switch_control = control.clone();
    console.register(
        cmd::<Ctx, _>(
            "scene",
            "list scenes (active marked `*`); `scene <slug>` switches, same slugs as --scene= / ?scene=",
            move |args, _ctx: &mut Ctx, out| match args.first().copied() {
                None => {
                    let active = switch_control.active();
                    for (i, entry) in R::SCENES.iter().enumerate() {
                        let mark = if i == active { '*' } else { ' ' };
                        out.line(format!("{mark} {} - {}", entry.slug, entry.label));
                    }
                    Ok(())
                }
                Some(slug) => {
                    request_scene(&switch_control, R::SCENES, slug)?;
                    out.line(format!("scene: switching to `{slug}`"));
                    Ok(())
                }
            },
        )
        .with_args(&[&slugs])
        .with_long_help(
            "Use `scene <slug>` to switch. Bare `scene` lists scenes and marks the active one.\n\
             Select the initial scene with `--scene=<slug>` or `?scene=<slug>`.\n\
             `--embed=1` or `?embed=1` hides the shell menu bar.",
        ),
    );
    let restart_control = control.clone();
    console.register(
        cmd::<Ctx, _>(
            "restart",
            "rebuild the active scene at its boot state; `restart force` skips the confirmation",
            move |args, _ctx: &mut Ctx, out| run_restart(&restart_control, args, out),
        )
        .with_args(&[&["force"]])
        .with_long_help(
            "Rebuilds the active scene and clears its console history. R does the same.\n\
             Unsaved work requires confirmation unless `force` is given.\n\
             The shell keeps the frame-script playhead across restarts.",
        ),
    );
}

struct Scenes {
    active: Box<dyn Scene>,
    inactive: Vec<Option<Box<dyn Scene>>>,
    control: SceneControl,
}

impl Scenes {
    fn new(count: usize, active: Box<dyn Scene>, control: SceneControl) -> Self {
        Self {
            active,
            inactive: std::iter::repeat_with(|| None).take(count).collect(),
            control,
        }
    }

    fn drain_pending(
        &mut self,
        registry: &[SceneEntry],
        build: impl FnOnce(usize) -> Result<Box<dyn Scene>>,
    ) -> Drain {
        let request = self.control.0.borrow_mut().pending.take();
        let Some(request) = request else {
            return Drain::Nothing;
        };
        let current = self.control.active();
        match request {
            Request::Switch(next) if next != current => {
                let replacement = match self.inactive[next].take() {
                    Some(scene) => Ok(scene),
                    None => build(next),
                };
                match replacement {
                    Ok(scene) => {
                        self.inactive[current] = Some(std::mem::replace(&mut self.active, scene));
                        self.control.0.borrow_mut().active = next;
                    }
                    Err(err) => {
                        tracing::error!("scene '{}' failed to build: {err:#}", registry[next].slug)
                    }
                }
            }
            Request::Switch(_) => {}
            Request::Restart { forced } => {
                if !forced {
                    if let Some(reason) = self.active.unsaved_work() {
                        return Drain::Ask(reason);
                    }
                }
                match build(current) {
                    Ok(scene) => self.active = scene,
                    Err(err) => tracing::error!(
                        "scene '{}' failed to rebuild: {err:#}",
                        registry[current].slug
                    ),
                }
            }
        }
        Drain::Applied
    }
}

pub struct SceneShell<R: SceneRegistry> {
    scenes: Scenes,
    embed: bool,
    capture_panel: crate::capture::CapturePanel,
    perf: crate::trace::PerfOverlay,
    confirm: Option<Cow<'static, str>>,
    script: Option<crate::script::ScriptDriver>,
    registry: PhantomData<fn() -> R>,
}

impl<R: SceneRegistry> SceneShell<R> {
    fn active_scene(&mut self) -> &mut dyn Scene {
        self.scenes.active.as_mut()
    }

    // `watcher` is `None`: the runner lends it only for the duration of `setup`.
    fn apply_pending_switch(
        &mut self,
        rd: &RenderDevice,
        shader_db: &mut ShaderDb,
        runtime: &Runtime,
        time: f32,
    ) {
        let Self {
            scenes, confirm, ..
        } = self;
        let control = scenes.control.clone();
        let drained = scenes.drain_pending(R::SCENES, |next| {
            let mut setup = SetupCtx {
                rd,
                shader_db,
                runtime,
                watcher: None,
                time,
            };
            (R::SCENES[next].build)(&mut setup, &control)
        });
        match drained {
            Drain::Nothing => {}
            Drain::Applied => {
                *confirm = None;
            }
            Drain::Ask(reason) => *confirm = Some(reason),
        }
    }

    // Painted before the drain, so an answer applies in the frame it is given.
    fn show_restart_confirm(&mut self, ctx: &egui::Context) {
        let Some(reason) = self.confirm.as_deref() else {
            return;
        };
        let mut answer: Option<bool> = None;
        egui::Modal::new(egui::Id::new("shell-restart-confirm")).show(ctx, |ui| {
            ui.heading("Restart this scene?");
            ui.label(reason);
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Restart").clicked() {
                    answer = Some(true);
                }
                if ui.button("Cancel").clicked() {
                    answer = Some(false);
                }
            });
        });
        match answer {
            None => {}
            Some(true) => {
                self.scenes.control.queue(Request::Restart { forced: true });
                self.confirm = None;
            }
            Some(false) => self.confirm = None,
        }
    }
}

enum Drain {
    Nothing,
    Applied,
    Ask(Cow<'static, str>),
}

fn resolve_boot(scenes: &[SceneEntry], args: &Args) -> (usize, bool) {
    let active = match args.get("scene") {
        None => 0,
        Some(slug) => scene_index(scenes, slug).unwrap_or_else(|| {
            tracing::warn!("unknown scene '{slug}'; defaulting to '{}'", scenes[0].slug);
            0
        }),
    };
    let embed = args.get("embed").is_some_and(|v| v != "0" && v != "false");
    (active, embed)
}

impl<R: SceneRegistry> App for SceneShell<R> {
    fn setup(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        if R::SCENES.is_empty() {
            bail!("scene registry is empty");
        }
        let args = Args::current();
        let (active, embed) = resolve_boot(R::SCENES, &args);
        let script = crate::script::driver_from_args(&args)?;
        let control = SceneControl::new(active);
        let scene = (R::SCENES[active].build)(ctx, &control)?;
        let scenes = Scenes::new(R::SCENES.len(), scene, control);
        Ok(Self {
            scenes,
            embed,
            capture_panel: crate::capture::CapturePanel::new(),
            perf: crate::trace::PerfOverlay::new(),
            confirm: None,
            script,
            registry: PhantomData,
        })
    }

    fn apply_shader_events(&mut self, events: &[std::path::PathBuf], shader_db: &mut ShaderDb) {
        self.scenes.active.apply_shader_events(events, shader_db);
        for scene in self.scenes.inactive.iter_mut().flatten() {
            scene.apply_shader_events(events, shader_db);
        }
    }

    fn apply_command(&mut self, cmd: &CommandLine, ctx: &mut CommandCtx<'_>) -> Result<()> {
        self.active_scene().apply_command(cmd, ctx)?;
        // Resolve lifecycle commands before the next command selects its target scene.
        self.apply_pending_switch(ctx.rd, ctx.shader_db, ctx.runtime, ctx.time);
        Ok(())
    }

    fn tick(&mut self, dt: f32, ctx: &mut TickCtx) {
        self.active_scene().tick(dt, ctx);
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        if let Some(driver) = self.script.as_mut() {
            if driver.advance_with(|command| ctx.runtime.submit(command))
                == crate::script::ScriptStatus::Finished
            {
                self.script = None;
                ctx.runtime.request_exit();
            }
        }
        self.active_scene().update(ctx);
    }

    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        if !self.embed {
            // The bar renders first so the scene's windows see it in `available_rect()`.
            let control = self.scenes.control.clone();
            let active = control.active();
            let scene = self.active_scene();
            egui::TopBottomPanel::top("shell-menu-bar").show(ctx, |ui| {
                egui::MenuBar::new().ui(ui, |ui| {
                    ui.menu_button("Demo", |ui| {
                        for (i, entry) in R::SCENES.iter().enumerate() {
                            if ui.selectable_label(active == i, entry.label).clicked() {
                                control.queue(Request::Switch(i));
                                ui.close_kind(egui::UiKind::Menu);
                            }
                        }
                        ui.separator();
                        if ui.button("Restart scene (R)").clicked() {
                            control.queue(Request::Restart { forced: false });
                            ui.close_kind(egui::UiKind::Menu);
                        }
                    });
                    scene.menus(ui);
                });
            });
        }
        self.active_scene().ui(ctx, frame);
        self.capture_panel.show(ctx, frame.runtime);
        self.perf.show(ctx);
        self.show_restart_confirm(ctx);
        // Apply menu choices and confirmation responses before recording.
        self.apply_pending_switch(frame.rd, frame.shader_db, frame.runtime, frame.time);
    }

    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    ) {
        if claims_restart(code, state, ctx.ui_capture.keyboard) {
            self.scenes
                .control
                .queue(Request::Restart { forced: false });
            return;
        }
        self.active_scene().on_key(code, state, ctx);
    }

    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> Result<()> {
        self.active_scene().record(ctx)
    }

    fn title(&self, fps: f32) -> Cow<'static, str> {
        self.scenes.active.title(fps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct StubScene {
        state: Rc<Cell<u32>>,
        unsaved: bool,
    }

    impl Scene for StubScene {
        fn update(&mut self, _ctx: &mut FrameCtx<'_>) {}
        fn ui(&mut self, _ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {}
        fn on_key(
            &mut self,
            _code: winit::keyboard::KeyCode,
            _state: winit::event::ElementState,
            _ctx: &mut FrameCtx<'_>,
        ) {
        }
        fn record(&mut self, _ctx: &mut RenderCtx<'_>) -> Result<()> {
            Ok(())
        }
        fn title(&self, _fps: f32) -> Cow<'static, str> {
            self.state.get().to_string().into()
        }
        fn unsaved_work(&self) -> Option<Cow<'static, str>> {
            self.unsaved.then_some(Cow::Borrowed("unsaved edits"))
        }
    }

    fn scene(state: Rc<Cell<u32>>, unsaved: bool) -> Box<dyn Scene> {
        Box::new(StubScene { state, unsaved })
    }

    struct Fixture;
    impl SceneRegistry for Fixture {
        const SCENES: &'static [SceneEntry] = &[
            SceneEntry {
                slug: "first",
                label: "First",
                build: |_, _| unreachable!(),
            },
            SceneEntry {
                slug: "second",
                label: "Second",
                build: |_, _| unreachable!(),
            },
            SceneEntry {
                slug: "third",
                label: "Third",
                build: |_, _| unreachable!(),
            },
        ];
    }
    const REGISTRY: &[SceneEntry] = Fixture::SCENES;

    fn console(control: &SceneControl) -> Console<()> {
        let mut console = Console::new();
        register_scene_commands::<(), Fixture>(&mut console, control);
        console
    }

    fn marked_slug(console: &mut Console<()>) -> String {
        console.clear_history();
        crate::command::run_on_console(console, "scene", &mut ());
        console
            .history()
            .iter()
            .find_map(|line| line.text.strip_prefix("* "))
            .and_then(|marked| marked.split(' ').next())
            .expect("active scene label")
            .to_owned()
    }

    #[test]
    fn boot_selection_falls_back_and_respects_embed() {
        for (pairs, expected) in [
            (vec![], (0, false)),
            (vec![("scene", "missing")], (0, false)),
            (vec![("scene", "second"), ("embed", "1")], (1, true)),
            (vec![("embed", "true")], (0, true)),
            (vec![("embed", "0")], (0, false)),
            (vec![("embed", "false")], (0, false)),
        ] {
            assert_eq!(resolve_boot(REGISTRY, &Args::from_pairs(pairs)), expected);
        }
    }

    #[test]
    fn scene_commands_reuse_cached_state_without_crossing_shells() {
        let control = SceneControl::new(0);
        let other_control = SceneControl::new(1);
        let mut console = console(&control);
        let mut other_console = self::console(&other_control);
        let state = Rc::new(Cell::new(7));
        let mut scenes = Scenes::new(REGISTRY.len(), scene(state.clone(), false), control.clone());
        let builds = Cell::new(0);
        let build = |_| {
            builds.set(builds.get() + 1);
            Ok(scene(Rc::default(), false))
        };

        crate::command::run_on_console(&mut console, "scene second", &mut ());
        assert!(other_control.0.borrow().pending.is_none());
        assert!(matches!(
            scenes.drain_pending(REGISTRY, build),
            Drain::Applied
        ));
        assert_eq!(marked_slug(&mut console), "second");
        assert_eq!(builds.get(), 1);

        for target in ["second", "first", "second", "first"] {
            crate::command::run_on_console(&mut console, &format!("scene {target}"), &mut ());
            assert!(matches!(
                scenes.drain_pending(REGISTRY, build),
                Drain::Applied
            ));
            assert_eq!(marked_slug(&mut console), target);
        }
        assert_eq!(builds.get(), 1);
        assert_eq!(scenes.active.title(0.0), "7");
        assert_eq!(marked_slug(&mut other_console), "second");

        crate::command::run_on_console(&mut console, "scene third", &mut ());
        scenes.drain_pending(REGISTRY, |_| Err(anyhow!("build failed")));
        assert_eq!(marked_slug(&mut console), "first");
        assert_eq!(scenes.active.title(0.0), "7");
        assert!(matches!(
            scenes.drain_pending(REGISTRY, |_| unreachable!()),
            Drain::Nothing
        ));
    }

    #[test]
    fn restart_requires_consent_and_keeps_old_state_when_build_fails() {
        let control = SceneControl::new(0);
        let mut console = console(&control);
        let mut scenes = Scenes::new(
            REGISTRY.len(),
            scene(Rc::new(Cell::new(7)), true),
            control.clone(),
        );
        crate::command::run_on_console(&mut console, "restart", &mut ());
        assert!(
            matches!(scenes.drain_pending(REGISTRY, |_| unreachable!()), Drain::Ask(reason) if reason == "unsaved edits")
        );
        assert_eq!(scenes.active.title(0.0), "7");
        assert!(matches!(
            scenes.drain_pending(REGISTRY, |_| unreachable!()),
            Drain::Nothing
        ));

        crate::command::run_on_console(&mut console, "restart force", &mut ());
        scenes.drain_pending(REGISTRY, |_| Err(anyhow!("build failed")));
        assert_eq!(scenes.active.title(0.0), "7");

        crate::command::run_on_console(&mut console, "restart force", &mut ());
        assert!(matches!(
            scenes.drain_pending(REGISTRY, |index| {
                assert_eq!(index, 0);
                Ok(scene(Rc::default(), false))
            }),
            Drain::Applied
        ));
        assert_eq!(scenes.active.title(0.0), "0");
        assert_eq!(marked_slug(&mut console), "first");

        crate::command::run_on_console(&mut console, "restart", &mut ());
        assert!(matches!(
            scenes.drain_pending(REGISTRY, |_| Ok(scene(Rc::new(Cell::new(3)), false))),
            Drain::Applied
        ));
        assert_eq!(scenes.active.title(0.0), "3");
    }

    #[test]
    fn invalid_commands_leave_no_lifecycle_request() {
        let control = SceneControl::new(0);
        let mut out = ConsoleWriter::new();
        for args in [&["now"][..], &["force", "extra"][..]] {
            assert!(run_restart(&control, args, &mut out).is_err());
        }
        assert!(request_scene(&control, REGISTRY, "missing").is_err());
        assert!(control.0.borrow().pending.is_none());
    }

    #[test]
    fn restart_hotkey_ignores_release_and_keyboard_capture() {
        use winit::event::ElementState;
        use winit::keyboard::KeyCode;
        assert!(claims_restart(KeyCode::KeyR, ElementState::Pressed, false));
        assert!(!claims_restart(KeyCode::KeyR, ElementState::Pressed, true));
        assert!(!claims_restart(
            KeyCode::KeyR,
            ElementState::Released,
            false
        ));
        assert!(!claims_restart(KeyCode::KeyT, ElementState::Pressed, false));
    }
}
