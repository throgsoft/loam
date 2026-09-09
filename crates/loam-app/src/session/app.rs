use loam_egui::{egui, ConsoleWriter};
use loam_render::pass::{FramePass, Section};
use loam_runtime::host::HostConfig;
use loam_runtime::{Session, Stores};

use super::console::{SessionConsole, Submit};
use super::pacing::Pacer;
use super::WorkContext;
use crate::args::Args;
use crate::capture::CaptureRequest;

/// Runs each frame between publication and presentation; `sections` are the previous frame's, since the presenter clears them at upload.
pub struct FrameHook<'a, A: Stores> {
    pub session: &'a mut Session<A>,
    pub sections: &'a [Section],
    pub ui: Option<&'a egui::Context>,
    pub size: (u32, u32),
}

pub(crate) type FrameFn<A> = Box<dyn FnMut(&mut FrameHook<'_, A>)>;
pub(crate) type WorkFn = Box<dyn FnMut(WorkContext<'_>)>;

/// What the host runs on the application's behalf: pacing, vsync, passes, the frame hook, the console, the work recorder, captures, and the browser element ids.
pub struct SessionApp<A: Stores> {
    pub config: HostConfig,
    pub args: Args,
    pub(crate) pacer: Pacer,
    pub(crate) vsync: Option<bool>,
    pub(crate) passes: Vec<Box<dyn FramePass>>,
    pub(crate) frame: Option<FrameFn<A>>,
    pub(crate) console: SessionConsole<A>,
    pub(crate) work: WorkFn,
    pub(crate) captures: Vec<CaptureRequest>,
    pub(crate) debug_layer: bool,
    pub wasm: crate::WasmConfig,
}

impl<A: Stores> SessionApp<A> {
    pub fn new(config: HostConfig) -> Self {
        Self::with_args(config, Args::current())
    }

    pub fn with_args(config: HostConfig, args: Args) -> Self {
        let mut host = Self {
            config,
            args,
            pacer: Pacer::default(),
            vsync: None,
            passes: Vec::new(),
            frame: None,
            console: SessionConsole::default(),
            work: Box::new(|_| {}),
            captures: Vec::new(),
            debug_layer: true,
            wasm: crate::WasmConfig::default(),
        };
        host.apply_args();
        host
    }

    /// Reads `--fps` and `--vsync` from `args`.
    pub fn apply_args(&mut self) {
        if let Some(fps) = self.args.parse::<f32>("fps") {
            self.pacer.set_target_fps(fps);
        }
        if let Some(vsync) = match self.args.get("vsync") {
            Some("on") | Some("1") => Some(true),
            Some("off") | Some("0") => Some(false),
            _ => None,
        } {
            self.vsync = Some(vsync);
        }
    }

    pub fn pass(mut self, pass: Box<dyn FramePass>) -> Self {
        self.passes.push(pass);
        self
    }

    pub fn on_frame(mut self, hook: impl FnMut(&mut FrameHook<'_, A>) + 'static) -> Self {
        self.frame = Some(Box::new(hook));
        self
    }

    pub fn command(
        mut self,
        name: &'static str,
        help: &'static str,
        handler: impl FnMut(&[&str], &mut Submit<A>, &mut ConsoleWriter) -> anyhow::Result<()> + 'static,
    ) -> Self {
        self.console.register(name, help, handler);
        self
    }

    pub fn work(mut self, record: impl FnMut(WorkContext<'_>) + 'static) -> Self {
        self.work = Box::new(record);
        self
    }

    pub fn target_fps(mut self, fps: f32) -> Self {
        self.pacer.set_target_fps(fps);
        self
    }

    pub fn vsync(mut self, enabled: bool) -> Self {
        self.vsync = Some(enabled);
        self
    }

    pub fn capture(mut self, request: CaptureRequest) -> Self {
        self.captures.push(request);
        self
    }

    pub fn debug_layer(mut self, enabled: bool) -> Self {
        self.debug_layer = enabled;
        self
    }

    pub fn console_mut(&mut self) -> &mut SessionConsole<A> {
        &mut self.console
    }
}
