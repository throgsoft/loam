use loam_egui::{egui, ConsoleWriter};
use loam_render::pass::{FramePass, Section};
use loam_runtime::host::HostConfig;
use loam_runtime::host::HostError;
use loam_runtime::{ActionId, Growth, Input, Publication, Session, Stores};

use super::commands::{CommandInbox, CommandSender};
use super::console::{SessionConsole, Submit};
use super::cursor::{CursorCapture, CursorPolicy};
use super::pacing::Pacer;
use crate::args::Args;
use crate::capture::{CaptureRequest, CaptureUnavailable};
use crate::script::{driver_from_args, ScriptDriver};

/// Runs before the frame boundary with read-only session input.
pub struct InputHook<'a, A: Stores> {
    pub session: &'a Session<A>,
    pub input: &'a Input,
    pub ui: Option<&'a egui::Context>,
    pub size: (u32, u32),
    pub sender: &'a CommandSender<A>,
}

/// Runs between publication and presentation.
pub struct FrameHook<'a, A: Stores> {
    pub session: &'a Session<A>,
    pub published: &'a Publication,
    pub sections: &'a [Section],
    pub ui: Option<&'a egui::Context>,
    pub size: (u32, u32),
    pub sender: &'a CommandSender<A>,
    pub capture: CaptureControl<'a>,
    pub(crate) cursor: &'a mut CursorCapture,
}

impl<A: Stores> FrameHook<'_, A> {
    pub fn capture_cursor(&mut self, enabled: bool, policy: CursorPolicy) {
        self.cursor.capture(enabled, policy);
    }

    pub fn cursor_locked(&self) -> bool {
        self.cursor.locked()
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

pub(crate) type FrameFn<A> = Box<dyn FnMut(&mut FrameHook<'_, A>)>;
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
    pub(crate) debug_layer: bool,
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
            debug_layer: true,
            script: None,
        };
        host.apply_args();
        host
    }

    /// Reads `--fps`, `--vsync`, and `--script` from `args`; a script that fails to load is logged and dropped.
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

    pub fn on_frame(mut self, hook: impl FnMut(&mut FrameHook<'_, A>) + 'static) -> Self {
        self.frame = Some(Box::new(hook));
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

    pub fn target_fps(mut self, fps: f32) -> Self {
        self.pacer.set_target_fps(fps);
        self
    }

    pub fn vsync(mut self, enabled: bool) -> Self {
        self.vsync = Some(enabled);
        self
    }

    pub fn capture(mut self, request: CaptureRequest) -> Result<Self, CaptureUnavailable> {
        queue_capture(&mut self.captures, request, CAPTURE_SUPPORTED)?;
        Ok(self)
    }

    pub fn debug_layer(mut self, enabled: bool) -> Self {
        self.debug_layer = enabled;
        self
    }

    pub fn console_mut(&mut self) -> &mut SessionConsole<A> {
        &mut self.console
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    loam_runtime::stores! {
        #[derive(Default)]
        pub struct Bare {}
    }

    #[test]
    fn unsupported_capture_refuses_without_retaining_requests() {
        let mut requests = Vec::new();
        let mut control = CaptureControl::with_support(&mut requests, false);
        assert_eq!(
            control.start(CaptureRequest::OneShot {
                stage: crate::capture::CaptureStage::Post,
                dir: None,
                name: None,
            }),
            Err(CaptureUnavailable)
        );
        assert_eq!(control.stop(), Err(CaptureUnavailable));
        assert!(requests.is_empty());
        #[cfg(any(not(feature = "capture"), target_arch = "wasm32"))]
        {
            let app = SessionApp::<Bare>::with_args(
                HostConfig::new("capture", loam_runtime::Bindings::new()),
                Args::default(),
            );
            assert!(matches!(
                app.capture(CaptureRequest::Stop),
                Err(CaptureUnavailable)
            ));
        }
    }
}
