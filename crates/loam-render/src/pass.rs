use std::fmt;
use std::time::Duration;

use web_time::Instant;
use wgpu::{CommandEncoder, TextureFormat, TextureView};

use crate::device::{GpuContext, LossSignal, UncapturedGpuError};
use crate::gpu_timer::SectionTimer;
use crate::DepthConvention;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PassStage {
    Background,
    Scene,
    Overlay,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PassPhase {
    Attach,
    Record,
}

impl fmt::Display for PassPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Attach => f.write_str("attach"),
            Self::Record => f.write_str("record"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PassExecutionError {
    #[error("pass `{pass}` failed during {phase}: {source:#}")]
    Pass {
        pass: &'static str,
        phase: PassPhase,
        #[source]
        source: anyhow::Error,
    },
    #[error("GPU work outside a pass failed: {source:#}")]
    Backend {
        #[source]
        source: anyhow::Error,
    },
}

/// The color and depth formats the presenter negotiated, handed to every pass at attach.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FrameFormat {
    pub color: TextureFormat,
    pub depth: TextureFormat,
}

pub struct FrameTarget<'a> {
    pub color: &'a TextureView,
    pub depth: Option<&'a TextureView>,
    pub size: (u32, u32),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ColorLoad {
    Load,
    Clear,
}

pub trait FramePass {
    fn name(&self) -> &'static str;

    fn stage(&self) -> PassStage;

    fn color_load(&self) -> ColorLoad {
        ColorLoad::Load
    }

    /// `None` means the pass writes no depth; `Some` must match the frame's convention to register.
    fn depth_convention(&self) -> Option<DepthConvention> {
        None
    }

    fn depth_read(&self) -> Option<DepthConvention> {
        None
    }

    fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> anyhow::Result<()>;

    /// Builds device resources for startup or device replacement.
    fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> anyhow::Result<()>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PassError {
    DepthConvention {
        pass: &'static str,
        declared: DepthConvention,
        frame: DepthConvention,
    },
    DepthRead {
        pass: &'static str,
        declared: DepthConvention,
        frame: DepthConvention,
    },
    ColorClear {
        pass: &'static str,
        stage: PassStage,
    },
}

impl fmt::Display for PassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DepthConvention {
                pass,
                declared,
                frame,
            } => write!(
                f,
                "pass `{pass}` writes depth under {declared:?}; this frame compares under {frame:?}"
            ),
            Self::DepthRead {
                pass,
                declared,
                frame,
            } => write!(
                f,
                "pass `{pass}` reads depth under {declared:?}; this frame compares under {frame:?}"
            ),
            Self::ColorClear { pass, stage } => write!(
                f,
                "pass `{pass}` clears the shared color target in the {stage:?} stage; only {:?} clears it",
                PassStage::Background
            ),
        }
    }
}

impl std::error::Error for PassError {}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GpuTime {
    /// No timer, no free slot, or no result mapped yet; `Measured` can hold zero for a trivial pass.
    Unavailable,
    Measured(Duration),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Section {
    pub name: &'static str,
    /// Time spent recording the section into the encoder, not running it.
    pub cpu: Duration,
    pub gpu: GpuTime,
}

pub struct PassSchedule {
    convention: DepthConvention,
    passes: Vec<Box<dyn FramePass>>,
    sections: Vec<Section>,
    timer: Option<SectionTimer>,
    signal: Option<std::sync::Arc<LossSignal>>,
    unattached: Option<&'static str>,
}

impl PassSchedule {
    pub fn new(convention: DepthConvention) -> Self {
        Self {
            convention,
            passes: Vec::new(),
            sections: Vec::new(),
            timer: None,
            signal: None,
            unattached: None,
        }
    }

    pub fn convention(&self) -> DepthConvention {
        self.convention
    }

    pub fn unattached(&self) -> Option<&'static str> {
        self.unattached
    }

    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.passes.iter().map(|pass| pass.name())
    }

    pub fn sections(&self) -> &[Section] {
        &self.sections
    }

    pub fn register(&mut self, pass: Box<dyn FramePass>) -> Result<(), PassError> {
        if let Some(declared) = pass.depth_convention() {
            if declared != self.convention {
                return Err(PassError::DepthConvention {
                    pass: pass.name(),
                    declared,
                    frame: self.convention,
                });
            }
        }
        if let Some(declared) = pass.depth_read() {
            if declared != self.convention {
                return Err(PassError::DepthRead {
                    pass: pass.name(),
                    declared,
                    frame: self.convention,
                });
            }
        }
        let stage = pass.stage();
        if pass.color_load() == ColorLoad::Clear && stage != PassStage::Background {
            return Err(PassError::ColorClear {
                pass: pass.name(),
                stage,
            });
        }
        let at = self.passes.partition_point(|held| held.stage() <= stage);
        self.passes.insert(at, pass);
        Ok(())
    }

    pub fn attach(
        &mut self,
        gpu: &GpuContext,
        frame: FrameFormat,
    ) -> Result<(), PassExecutionError> {
        self.signal = Some(gpu.loss_signal());
        self.timer = SectionTimer::new(&gpu.device, &gpu.queue);
        let signal = self.signal.as_deref();
        for pass in self.passes.iter_mut() {
            let name = pass.name();
            if let Err(error) =
                run_pass(signal, name, PassPhase::Attach, || pass.attach(gpu, frame))
            {
                self.unattached = Some(name);
                return Err(error);
            }
        }
        self.unattached = None;
        Ok(())
    }

    pub fn begin_frame(&mut self) {
        self.sections.clear();
        if let Some(timer) = self.timer.as_mut() {
            timer.begin_frame();
        }
    }

    pub fn section(
        &mut self,
        name: &'static str,
        encoder: &mut CommandEncoder,
        body: impl FnOnce(&mut CommandEncoder),
    ) {
        time_section(&mut self.timer, &mut self.sections, name, encoder, body);
    }

    pub fn record(
        &mut self,
        stage: PassStage,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> Result<(), PassExecutionError> {
        if self.unattached.is_some() {
            return Ok(());
        }
        let timer = &mut self.timer;
        let sections = &mut self.sections;
        let signal = self.signal.as_deref();
        for pass in self.passes.iter_mut().filter(|pass| pass.stage() == stage) {
            let name = pass.name();
            run_pass(signal, name, PassPhase::Record, || {
                time_section(timer, sections, name, encoder, |encoder| {
                    pass.record(encoder, target)
                })
            })?;
        }
        Ok(())
    }

    pub fn end_frame(&mut self, encoder: &mut CommandEncoder) {
        if let Some(timer) = self.timer.as_mut() {
            timer.resolve(encoder);
        }
    }

    pub fn after_submit(&mut self) {
        if let Some(timer) = self.timer.as_mut() {
            timer.after_submit();
        }
    }
}

fn run_pass<T>(
    signal: Option<&LossSignal>,
    pass: &'static str,
    phase: PassPhase,
    body: impl FnOnce() -> anyhow::Result<T>,
) -> Result<T, PassExecutionError> {
    if let Some(source) = signal.and_then(LossSignal::take_error) {
        return Err(backend_error(source));
    }
    let outcome = body();
    if let Some(source) = signal.and_then(LossSignal::take_error) {
        return Err(PassExecutionError::Pass {
            pass,
            phase,
            source: source.into(),
        });
    }
    outcome.map_err(|source| PassExecutionError::Pass {
        pass,
        phase,
        source,
    })
}

fn backend_error(source: UncapturedGpuError) -> PassExecutionError {
    PassExecutionError::Backend {
        source: source.into(),
    }
}

fn time_section<T>(
    timer: &mut Option<SectionTimer>,
    sections: &mut Vec<Section>,
    name: &'static str,
    encoder: &mut CommandEncoder,
    body: impl FnOnce(&mut CommandEncoder) -> T,
) -> T {
    let _scope = loam_time::frame_trace::scope(name);
    let slot = timer.as_mut().and_then(|timer| timer.open(encoder, name));
    let started = Instant::now();
    let outcome = body(encoder);
    let cpu = started.elapsed();
    let gpu = match (timer.as_mut(), slot) {
        (Some(timer), Some(slot)) => {
            timer.close(encoder, slot);
            timer
                .elapsed(slot)
                .map_or(GpuTime::Unavailable, GpuTime::Measured)
        }
        _ => GpuTime::Unavailable,
    };
    sections.push(Section { name, cpu, gpu });
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    struct WrongVertexInput;

    impl FramePass for WrongVertexInput {
        fn name(&self) -> &'static str {
            "wrong vertex input"
        }

        fn stage(&self) -> PassStage {
            PassStage::Scene
        }

        fn record(
            &mut self,
            _encoder: &mut CommandEncoder,
            _target: &FrameTarget<'_>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> anyhow::Result<()> {
            let shader = gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("wrong vertex input shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vertex(@location(0) position: vec3<f32>) -> VertexOutput {
    return VertexOutput(vec4<f32>(position, 1.0));
}

@fragment
fn fragment() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#
                        .into(),
                    ),
                });
            let _pipeline = gpu
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("wrong vertex input pipeline"),
                    layout: None,
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vertex"),
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: 8,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32x2,
                                offset: 0,
                                shader_location: 0,
                            }],
                        }],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fragment"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: frame.color,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    multiview: None,
                    cache: None,
                });
            Ok(())
        }
    }

    struct Probe {
        name: &'static str,
        convention: Option<DepthConvention>,
        stage: PassStage,
    }

    impl FramePass for Probe {
        fn name(&self) -> &'static str {
            self.name
        }

        fn stage(&self) -> PassStage {
            self.stage
        }

        fn depth_convention(&self) -> Option<DepthConvention> {
            self.convention
        }

        fn record(
            &mut self,
            _encoder: &mut CommandEncoder,
            _target: &FrameTarget<'_>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        fn attach(&mut self, _gpu: &GpuContext, _frame: FrameFormat) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn an_uncaptured_pipeline_error_names_the_pass_and_validation_cause() {
        let gpu = crate::device::noop_context();
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        schedule.register(Box::new(WrongVertexInput)).unwrap();
        let error = schedule
            .attach(
                &gpu,
                FrameFormat {
                    color: TextureFormat::Rgba8Unorm,
                    depth: TextureFormat::Depth32Float,
                },
            )
            .unwrap_err();
        let PassExecutionError::Pass {
            pass,
            phase,
            source,
        } = error
        else {
            panic!("pipeline error was not attributed to its pass");
        };
        assert_eq!(pass, "wrong vertex input");
        assert_eq!(phase, PassPhase::Attach);
        let cause = format!("{source:#}");
        assert!(
            cause.contains("Input type is not compatible"),
            "unexpected validation cause: {cause}"
        );
    }

    fn probe(name: &'static str) -> Box<Probe> {
        Box::new(Probe {
            name,
            convention: None,
            stage: PassStage::Scene,
        })
    }

    #[test]
    fn a_pass_writing_depth_under_the_other_convention_is_refused_at_registration() {
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        let refused = schedule.register(Box::new(Probe {
            name: "standard-z overlay",
            convention: Some(DepthConvention::StandardZ),
            stage: PassStage::Scene,
        }));
        assert_eq!(
            refused,
            Err(PassError::DepthConvention {
                pass: "standard-z overlay",
                declared: DepthConvention::StandardZ,
                frame: DepthConvention::ReversedZ,
            })
        );
        assert_eq!(schedule.names().count(), 0);
    }

    #[test]
    fn a_stage_orders_a_pass_ahead_of_one_registered_before_it() {
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        schedule
            .register(Box::new(Probe {
                name: "overlay",
                convention: None,
                stage: PassStage::Overlay,
            }))
            .unwrap();
        schedule.register(probe("scene")).unwrap();
        schedule
            .register(Box::new(Probe {
                name: "background",
                convention: None,
                stage: PassStage::Background,
            }))
            .unwrap();
        assert_eq!(
            schedule.names().collect::<Vec<_>>(),
            ["background", "scene", "overlay"]
        );
    }
}
