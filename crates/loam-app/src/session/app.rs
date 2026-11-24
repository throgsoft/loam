use loam_console::ConsoleWriter;
#[cfg(feature = "egui")]
use loam_egui::egui;
use loam_render::pass::{FramePass, Section};
use loam_runtime::host::HostConfig;
use loam_runtime::host::HostError;
use loam_runtime::{
    ActionId, CursorPolicy, Eye, Growth, Input, Publication, Session, Setting, Stores,
};

use super::commands::{CommandInbox, CommandSender};
use super::console::{SessionConsole, Submit};
use super::cursor::CursorCapture;
use super::pacing::Pacer;
use crate::args::Args;
use crate::capture::{CaptureRequest, CaptureUnavailable};
use crate::script::{driver_from_args, ScriptDriver};

/// Runs before the frame boundary with read-only session input.
#[non_exhaustive]
pub struct InputHook<'a, A: Stores> {
    pub session: &'a Session<A>,
    pub input: &'a Input,
    #[cfg(feature = "egui")]
    pub ui: Option<&'a egui::Context>,
    pub size: (u32, u32),
    pub sender: &'a CommandSender<A>,
}

/// Runs between publication and presentation.
pub struct FrameHook<'a, A: Stores> {
    pub session: &'a Session<A>,
    pub published: &'a Publication<A::Published>,
    /// The alpha `published` was drawn at; pass it to `drawn_pose` for anything drawn beside the instances.
    pub alpha: f32,
    pub sections: &'a [Section],
    #[cfg(feature = "egui")]
    pub ui: Option<&'a egui::Context>,
    pub size: (u32, u32),
    pub sender: &'a CommandSender<A>,
    pub capture: CaptureControl<'a>,
    pub(crate) cursor: &'a mut CursorCapture,
    pub(crate) background: &'a mut wgpu::Color,
    pub(crate) posts: &'a mut Posts,
    pub(crate) redraw: &'a mut bool,
}

impl<A: Stores> FrameHook<'_, A> {
    pub fn capture_cursor(&mut self, enabled: bool, policy: CursorPolicy) {
        self.cursor.capture(enabled, policy);
    }

    pub fn cursor_locked(&self) -> bool {
        self.cursor.locked()
    }

    /// Linear light; the present clear shows wherever no Background pass paints.
    pub fn set_background(&mut self, rgb: [f32; 3]) {
        *self.background = background_color(rgb);
    }

    /// Sent to the page as one `loam-post` event after this frame (native hosts discard it); positions belong in normalized canvas coordinates in [0, 1].
    pub fn post(&mut self, topic: &'static str, values: &[f32]) {
        self.posts.push(topic, values);
    }

    /// Draws this frame under `Redraw::OnChange`.
    pub fn request_redraw(&mut self) {
        *self.redraw = true;
    }
}

/// Runs between publication and presentation, reading only what the session published.
pub struct PublishedHook<'a, A: Stores> {
    pub published: &'a Publication<A::Published>,
    /// The root image's eye this frame is presented with.
    pub eye: Eye,
    pub sections: &'a [Section],
    #[cfg(feature = "egui")]
    pub ui: Option<&'a egui::Context>,
    pub size: (u32, u32),
    pub sender: &'a CommandSender<A>,
    pub capture: CaptureControl<'a>,
    pub(crate) cursor: &'a mut CursorCapture,
    pub(crate) background: &'a mut wgpu::Color,
    pub(crate) posts: &'a mut Posts,
    pub(crate) redraw: &'a mut bool,
}

impl<A: Stores> PublishedHook<'_, A> {
    pub fn capture_cursor(&mut self, enabled: bool, policy: CursorPolicy) {
        self.cursor.capture(enabled, policy);
    }

    pub fn cursor_locked(&self) -> bool {
        self.cursor.locked()
    }

    /// Linear light; the present clear shows wherever no Background pass paints.
    pub fn set_background(&mut self, rgb: [f32; 3]) {
        *self.background = background_color(rgb);
    }

    /// Sent to the page as one `loam-post` event after this frame (native hosts discard it); positions belong in normalized canvas coordinates in [0, 1].
    pub fn post(&mut self, topic: &'static str, values: &[f32]) {
        self.posts.push(topic, values);
    }

    /// Draws this frame under `Redraw::OnChange`.
    pub fn request_redraw(&mut self) {
        *self.redraw = true;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Redraw {
    #[default]
    EveryFrame,
    /// Draws only when the publication content, root eye, size or background changed since the last draw, or a hook called `request_redraw`.
    OnChange,
}

#[derive(Default)]
pub(crate) struct Posts {
    entries: Vec<(&'static str, std::ops::Range<usize>)>,
    values: Vec<f32>,
}

impl Posts {
    fn push(&mut self, topic: &'static str, values: &[f32]) {
        let range = self.values.len()..self.values.len() + values.len();
        self.values.extend_from_slice(values);
        self.entries.push((topic, range));
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.values.clear();
    }

    #[cfg(any(target_arch = "wasm32", test))]
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&'static str, &[f32])> {
        self.entries
            .iter()
            .map(|(topic, range)| (*topic, &self.values[range.clone()]))
    }
}

const DEFAULT_BACKGROUND: wgpu::Color = wgpu::Color {
    r: 0.02,
    g: 0.02,
    b: 0.03,
    a: 1.0,
};

fn background_color(rgb: [f32; 3]) -> wgpu::Color {
    wgpu::Color {
        r: f64::from(rgb[0]),
        g: f64::from(rgb[1]),
        b: f64::from(rgb[2]),
        a: 1.0,
    }
}

/// Queues capture starts and stops that the host drains at the end of the same frame.
pub struct CaptureControl<'a> {
    requests: &'a mut Vec<CaptureRequest>,
    supported: bool,
}

const CAPTURE_SUPPORTED: bool = cfg!(all(feature = "capture", not(target_arch = "wasm32")));

fn queue_capture(
    requests: &mut Vec<CaptureRequest>,
    request: CaptureRequest,
    supported: bool,
) -> Result<(), CaptureUnavailable> {
    if !supported {
        return Err(CaptureUnavailable);
    }
    requests.push(request);
    Ok(())
}

impl<'a> CaptureControl<'a> {
    pub(crate) fn new(requests: &'a mut Vec<CaptureRequest>) -> Self {
        Self::with_support(requests, CAPTURE_SUPPORTED)
    }

    fn with_support(requests: &'a mut Vec<CaptureRequest>, supported: bool) -> Self {
        Self {
            requests,
            supported,
        }
    }

    pub fn start(&mut self, request: CaptureRequest) -> Result<(), CaptureUnavailable> {
        queue_capture(self.requests, request, self.supported)
    }

    pub fn stop(&mut self) -> Result<(), CaptureUnavailable> {
        self.start(CaptureRequest::Stop)
    }
}

type SessionFn<A> = Box<dyn FnMut(&mut FrameHook<'_, A>)>;
type PublishedFn<A> = Box<dyn FnMut(&mut PublishedHook<'_, A>)>;

pub(crate) enum FrameFn<A: Stores> {
    Session(SessionFn<A>),
    Published(PublishedFn<A>),
}
pub(crate) type InputFn<A> = Box<dyn FnMut(&InputHook<'_, A>)>;

pub struct SessionApp<A: Stores> {
    pub config: HostConfig,
    pub args: Args,
    pub(crate) pacer: Pacer,
    pub(crate) vsync: Option<bool>,
    pub(crate) passes: Vec<Box<dyn FramePass>>,
    pub(crate) input: Option<InputFn<A>>,
    pub(crate) fault_recovery: Option<ActionId>,
    pub(crate) frame: Option<FrameFn<A>>,
    pub(crate) commands: CommandInbox<A>,
    pub(crate) console: SessionConsole<A>,
    pub(crate) captures: Vec<CaptureRequest>,
    pub(crate) background: wgpu::Color,
    pub(crate) redraw: Redraw,
    #[cfg(feature = "egui")]
    pub(crate) debug_layer: bool,
    #[cfg(feature = "text")]
    pub(crate) font: Option<Vec<u8>>,
    pub(crate) script: Option<ScriptDriver>,
}

impl<A: Stores> SessionApp<A> {
    pub fn new(config: HostConfig) -> Self {
        Self::with_args(config, Args::current())
    }

    pub fn with_args(config: HostConfig, args: Args) -> Self {
        let commands = CommandInbox::default();
        let console = SessionConsole::new(commands.sender());
        let mut host = Self {
            config,
            args,
            pacer: Pacer::default(),
            vsync: None,
            passes: Vec::new(),
            input: None,
            fault_recovery: None,
            frame: None,
            commands,
            console,
            captures: Vec::new(),
            background: DEFAULT_BACKGROUND,
            redraw: Redraw::default(),
            #[cfg(feature = "egui")]
            debug_layer: true,
            #[cfg(feature = "text")]
            font: None,
            script: None,
        };
        host.apply_args();
        host
    }

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
        match driver_from_args(&self.args) {
            Ok(driver) => self.script = driver,
            Err(error) => tracing::error!("--script ignored: {error:#}"),
        }
    }

    pub fn pass(mut self, pass: Box<dyn FramePass>) -> Self {
        self.passes.push(pass);
        self
    }

    /// Replaces any hook that `on_published` set.
    pub fn on_frame(mut self, hook: impl FnMut(&mut FrameHook<'_, A>) + 'static) -> Self {
        self.frame = Some(FrameFn::Session(Box::new(hook)));
        self
    }

    /// Replaces any hook that `on_frame` set.
    pub fn on_published(mut self, hook: impl FnMut(&mut PublishedHook<'_, A>) + 'static) -> Self {
        self.frame = Some(FrameFn::Published(Box::new(hook)));
        self
    }

    pub fn on_input(mut self, hook: impl FnMut(&InputHook<'_, A>) + 'static) -> Self {
        self.input = Some(Box::new(hook));
        self
    }

    pub fn recover_on_fault(mut self, action: ActionId) -> Self {
        self.fault_recovery = Some(action);
        self
    }

    pub fn sender(&self) -> CommandSender<A> {
        self.commands.sender()
    }

    pub fn boundary(
        &mut self,
        session: &mut Session<A>,
        input: Input,
    ) -> Result<Growth, HostError> {
        let fault = session.faulted_phase().and_then(|_| session.phase_error());
        let recovery = self.commands.recover(session);
        let recovered = fault.filter(|_| recovery.is_ok() && session.faulted_phase().is_none());
        let result = recovery.map_err(HostError::from).and_then(|()| {
            self.commands.drain(session);
            session.boundary(input).map_err(HostError::from)
        });
        if let Some(fault) = recovered {
            self.console.note(format!("recovered: {fault}"));
        }
        self.commands.collect(session);
        self.console.collect();
        result
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

    pub fn toggle_command(
        mut self,
        name: &'static str,
        help: &'static str,
        setting: Setting<A>,
    ) -> Self {
        self.console.toggle_command(name, help, setting);
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

    /// Linear light; the present clear shows wherever no Background pass paints.
    pub fn background(mut self, rgb: [f32; 3]) -> Self {
        self.background = background_color(rgb);
        self
    }

    /// TrueType or OpenType bytes for screen text; with none given an egui build uses egui's bundled font.
    #[cfg(feature = "text")]
    pub fn font(mut self, bytes: Vec<u8>) -> Self {
        self.font = Some(bytes);
        self
    }

    pub fn redraw(mut self, policy: Redraw) -> Self {
        self.redraw = policy;
        self
    }

    pub fn capture(mut self, request: CaptureRequest) -> Result<Self, CaptureUnavailable> {
        queue_capture(&mut self.captures, request, CAPTURE_SUPPORTED)?;
        Ok(self)
    }

    #[cfg_attr(not(feature = "egui"), allow(unused_mut))]
    pub fn debug_layer(mut self, enabled: bool) -> Self {
        #[cfg(feature = "egui")]
        {
            self.debug_layer = enabled;
        }
        #[cfg(not(feature = "egui"))]
        let _ = enabled;
        self
    }

    pub fn console_mut(&mut self) -> &mut SessionConsole<A> {
        &mut self.console
    }
}
