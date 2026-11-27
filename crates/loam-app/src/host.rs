use std::fmt;
use std::path::PathBuf;

use loam_render::device::{GpuSetupError, UncapturedGpuError};
use loam_render::pass::{PassError, PassExecutionError};
use loam_runtime::{ActionId, DomainError, PhaseError, Rejection, RestoreError, SimConfigError};

use crate::headless::script::ScriptError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    Space,
    Escape,
    Letter(char),
    Digit(u8),
}

#[derive(Clone, Debug, Default)]
pub struct Bindings {
    keys: Vec<(Key, ActionId)>,
}

impl Bindings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn key(mut self, key: Key, action: ActionId) -> Self {
        self.keys.push((key, action));
        self
    }

    pub fn action(&self, key: Key) -> Option<ActionId> {
        self.keys
            .iter()
            .find(|(bound, _)| *bound == key)
            .map(|(_, action)| *action)
    }
}

pub struct HostConfig {
    pub title: &'static str,
    pub bindings: Bindings,
}

impl HostConfig {
    pub fn new(title: &'static str, bindings: Bindings) -> Self {
        Self { title, bindings }
    }
}

/// Why a launch, a frame or a headless run stopped.
pub enum HostError {
    Setup(Rejection),
    Phase(PhaseError),
    SimConfig(SimConfigError),
    Gpu(GpuSetupError),
    Poll(wgpu::PollError),
    UncapturedGpu(UncapturedGpuError),
    CreateSurface(wgpu::CreateSurfaceError),
    NoSurfaceFormat,
    NoAlphaMode,
    SurfaceMemory,
    Pass(PassError),
    PassFailed(PassExecutionError),
    #[cfg(not(target_arch = "wasm32"))]
    EventLoop(winit::error::EventLoopError),
    #[cfg(not(target_arch = "wasm32"))]
    Window(winit::error::OsError),
    #[cfg(target_arch = "wasm32")]
    Web(crate::web::WebError),
    #[cfg(feature = "text")]
    Font(loam_text::ab_glyph::InvalidFont),
    #[cfg(feature = "text")]
    Glyph(loam_text::glyph::GlyphError),
    /// A command-line flag whose value is not what `expected` says.
    Flag {
        flag: &'static str,
        value: String,
        expected: &'static str,
    },
    /// A flag given with a space where its value must follow `=`.
    Detached {
        flag: &'static str,
        example: &'static str,
    },
    NoRecords,
    Script {
        path: PathBuf,
        error: ScriptError,
    },
    /// A script line the console cannot run, with the reason.
    ScriptLine {
        line: usize,
        text: String,
        reason: String,
    },
    /// A script line the session refused, with its result line.
    ScriptRefused {
        line: usize,
        result: String,
    },
    /// An error line the console printed during a headless frame.
    Console {
        frame: u64,
        line: String,
    },
    FramePastEnd(u64),
    RecordDump,
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Setup(rejection) => fmt::Display::fmt(rejection, f),
            Self::Phase(error) => fmt::Display::fmt(error, f),
            Self::SimConfig(error) => fmt::Display::fmt(error, f),
            Self::Gpu(error) => fmt::Display::fmt(error, f),
            Self::Poll(error) => write!(f, "the device poll failed: {error}"),
            Self::UncapturedGpu(error) => fmt::Display::fmt(error, f),
            Self::CreateSurface(error) => write!(f, "the surface was not created: {error}"),
            Self::NoSurfaceFormat => f.write_str("the surface advertises no texture format"),
            Self::NoAlphaMode => f.write_str("the surface advertises no alpha mode"),
            Self::SurfaceMemory => f.write_str("surface ran out of memory"),
            Self::Pass(error) => fmt::Display::fmt(error, f),
            Self::PassFailed(error) => fmt::Display::fmt(error, f),
            #[cfg(not(target_arch = "wasm32"))]
            Self::EventLoop(error) => write!(f, "the event loop failed: {error}"),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Window(error) => write!(f, "the window was not created: {error}"),
            #[cfg(target_arch = "wasm32")]
            Self::Web(error) => fmt::Display::fmt(error, f),
            #[cfg(feature = "text")]
            Self::Font(error) => write!(f, "the font did not parse: {error}"),
            #[cfg(feature = "text")]
            Self::Glyph(error) => fmt::Display::fmt(error, f),
            Self::Flag {
                flag,
                value,
                expected,
            } => write!(f, "--{flag} {value:?} is not {expected}"),
            Self::Detached { flag, example } => write!(
                f,
                "--{flag} needs its value attached with `=`, as in `{example}`"
            ),
            Self::NoRecords => f.write_str("a headless run needs --records"),
            Self::Script { path, error } => write!(f, "script {}: {error}", path.display()),
            Self::ScriptLine { line, text, reason } => {
                write!(f, "script line {line}: `{text}`: {reason}")
            }
            Self::ScriptRefused { line, result } => write!(f, "script line {line}: {result}"),
            Self::Console { frame, line } => write!(f, "frame {frame}: {line}"),
            Self::FramePastEnd(frame) => write!(f, "frame {frame} is past the last step"),
            Self::RecordDump => f.write_str("the record dump failed"),
        }
    }
}

// A main that returns this prints its Debug form, so Debug is the Display text.
impl fmt::Debug for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for HostError {}

impl From<Rejection> for HostError {
    fn from(rejection: Rejection) -> Self {
        Self::Setup(rejection)
    }
}

impl From<DomainError> for HostError {
    fn from(error: DomainError) -> Self {
        Self::Setup(Rejection::Domain(error))
    }
}

impl From<RestoreError> for HostError {
    fn from(error: RestoreError) -> Self {
        Self::Setup(Rejection::Restore(error))
    }
}

impl From<PhaseError> for HostError {
    fn from(error: PhaseError) -> Self {
        Self::Phase(error)
    }
}

impl From<SimConfigError> for HostError {
    fn from(error: SimConfigError) -> Self {
        Self::SimConfig(error)
    }
}

impl From<GpuSetupError> for HostError {
    fn from(error: GpuSetupError) -> Self {
        Self::Gpu(error)
    }
}

impl From<wgpu::PollError> for HostError {
    fn from(error: wgpu::PollError) -> Self {
        Self::Poll(error)
    }
}

impl From<UncapturedGpuError> for HostError {
    fn from(error: UncapturedGpuError) -> Self {
        Self::UncapturedGpu(error)
    }
}

impl From<wgpu::CreateSurfaceError> for HostError {
    fn from(error: wgpu::CreateSurfaceError) -> Self {
        Self::CreateSurface(error)
    }
}

impl From<PassError> for HostError {
    fn from(error: PassError) -> Self {
        Self::Pass(error)
    }
}

impl From<PassExecutionError> for HostError {
    fn from(error: PassExecutionError) -> Self {
        Self::PassFailed(error)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<winit::error::EventLoopError> for HostError {
    fn from(error: winit::error::EventLoopError) -> Self {
        Self::EventLoop(error)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<winit::error::OsError> for HostError {
    fn from(error: winit::error::OsError) -> Self {
        Self::Window(error)
    }
}

#[cfg(target_arch = "wasm32")]
impl From<crate::web::WebError> for HostError {
    fn from(error: crate::web::WebError) -> Self {
        Self::Web(error)
    }
}

#[cfg(feature = "text")]
impl From<loam_text::ab_glyph::InvalidFont> for HostError {
    fn from(error: loam_text::ab_glyph::InvalidFont) -> Self {
        Self::Font(error)
    }
}

#[cfg(feature = "text")]
impl From<loam_text::glyph::GlyphError> for HostError {
    fn from(error: loam_text::glyph::GlyphError) -> Self {
        Self::Glyph(error)
    }
}
