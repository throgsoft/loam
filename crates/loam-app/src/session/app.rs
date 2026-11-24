use std::num::NonZeroU32;

use loam_console::ConsoleWriter;
#[cfg(feature = "egui")]
use loam_egui::egui;
use loam_render::pass::{FramePass, Section};
use loam_runtime::{
    ActionId, CursorPolicy, Eye, Growth, Input, Publication, Session, Setting, Stores,
};

use super::commands::{CommandInbox, CommandSender};
use super::console::{SessionConsole, Submit};
use super::cursor::CursorCapture;
use super::pacing::Pacer;
use crate::args::Args;
use crate::capture::{CaptureRequest, CaptureUnavailable};
use crate::host::{HostConfig, HostError};

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

pub(crate) type PublishedFn<A> = Box<dyn FnMut(&mut PublishedHook<'_, A>)>;
pub(crate) type InputFn<A> = Box<dyn FnMut(&InputHook<'_, A>)>;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) type RecordFn<A> = Box<dyn FnMut(&Session<A>, &mut String) -> std::fmt::Result>;

// Every 10th step keeps a record dump short enough to diff.
#[cfg(not(target_arch = "wasm32"))]
const RECORD_STRIDE: NonZeroU32 = NonZeroU32::new(10).expect("the record stride is nonzero");

pub struct SessionApp<A: Stores> {
    pub config: HostConfig,
    pub args: Args,
    pub(crate) pacer: Pacer,
    pub(crate) vsync: Option<bool>,
    pub(crate) passes: Vec<Box<dyn FramePass>>,
    pub(crate) input: Option<InputFn<A>>,
    pub(crate) fault_recovery: Option<ActionId>,
    pub(crate) published: Option<PublishedFn<A>>,
    pub(crate) commands: CommandInbox<A>,
    pub(crate) console: SessionConsole<A>,
    pub(crate) captures: Vec<CaptureRequest>,
    pub(crate) background: wgpu::Color,
    pub(crate) redraw: Redraw,
    pub(crate) debug_layer: bool,
    #[cfg(feature = "text")]
    pub(crate) font: Option<Vec<u8>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) record: Option<RecordFn<A>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) record_stride: NonZeroU32,
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
            published: None,
            commands,
            console,
            captures: Vec::new(),
            background: DEFAULT_BACKGROUND,
            redraw: Redraw::default(),
            debug_layer: true,
            #[cfg(feature = "text")]
            font: None,
            #[cfg(not(target_arch = "wasm32"))]
            record: None,
            #[cfg(not(target_arch = "wasm32"))]
            record_stride: RECORD_STRIDE,
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
    }

    pub fn pass(mut self, pass: Box<dyn FramePass>) -> Self {
        self.passes.push(pass);
        self
    }

    pub fn on_published(mut self, hook: impl FnMut(&mut PublishedHook<'_, A>) + 'static) -> Self {
        self.published = Some(Box::new(hook));
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

    /// Writes lines into a headless record dump after the engine's own lines for each recorded step.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn on_record(
        mut self,
        hook: impl FnMut(&Session<A>, &mut String) -> std::fmt::Result + 'static,
    ) -> Self {
        self.record = Some(Box::new(hook));
        self
    }

    /// A headless record dump writes every `stride`th step; the default is 10.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn record_every(mut self, stride: NonZeroU32) -> Self {
        self.record_stride = stride;
        self
    }

    /// Does nothing on wasm32, which has no headless run.
    #[cfg(target_arch = "wasm32")]
    pub fn on_record(
        self,
        _hook: impl FnMut(&Session<A>, &mut String) -> std::fmt::Result + 'static,
    ) -> Self {
        self
    }

    /// Does nothing on wasm32, which has no headless run.
    #[cfg(target_arch = "wasm32")]
    pub fn record_every(self, _stride: NonZeroU32) -> Self {
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
