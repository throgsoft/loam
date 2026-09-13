use std::fmt;
use std::time::Duration;

use web_time::Instant;
use wgpu::{CommandEncoder, TextureFormat, TextureView};

use crate::device::{GpuContext, LossSignal, UncapturedGpuError};
use crate::gpu_timer::SectionTimer;
use crate::DepthConvention;

pub type ResourceId = &'static str;

pub const SCENE_COLOR: ResourceId = "scene-color";
pub const SCENE_DEPTH: ResourceId = "scene-depth";
pub(crate) const SCENE_BASE: ResourceId = "scene-base";

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PassStage {
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

pub trait FramePass {
    fn name(&self) -> &'static str;

    /// Tokens whose producers precede this pass within its stage.
    fn reads(&self) -> &[ResourceId] {
        &[]
    }

    /// Tokens that place this pass before their consumers within its stage.
    fn writes(&self) -> &[ResourceId] {
        &[]
    }

    fn stage(&self) -> PassStage;

    /// `None` means the pass writes no depth; `Some` must match the frame's convention to register.
    fn depth_convention(&self) -> Option<DepthConvention> {
        None
    }

    fn record(&self, encoder: &mut CommandEncoder, target: &FrameTarget<'_>) -> anyhow::Result<()>;

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
    Cycle {
        pass: &'static str,
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
            Self::Cycle { pass } => write!(
                f,
                "pass `{pass}` both follows and precedes a registered pass"
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
}

impl PassSchedule {
    pub fn new(convention: DepthConvention) -> Self {
        Self {
            convention,
            passes: Vec::new(),
            sections: Vec::new(),
            timer: None,
            signal: None,
        }
    }

    pub fn convention(&self) -> DepthConvention {
        self.convention
    }

    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.passes.iter().map(|pass| pass.name())
    }

    pub fn sections(&self) -> &[Section] {
        &self.sections
    }

    /// Orders producers before consumers and serializes passes that both read and write the same resource in registration order.
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
        let name = pass.name();
        self.passes.push(pass);
        let Some(order) = topological(&self.passes) else {
            self.passes.pop();
            return Err(PassError::Cycle { pass: name });
        };
        let mut held: Vec<Option<Box<dyn FramePass>>> = self.passes.drain(..).map(Some).collect();
        for index in order {
            if let Some(pass) = held[index].take() {
                self.passes.push(pass);
            }
        }
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
            run_pass(signal, name, PassPhase::Attach, || pass.attach(gpu, frame))?;
        }
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
        let timer = &mut self.timer;
        let sections = &mut self.sections;
        let signal = self.signal.as_deref();
        for pass in self.passes.iter().filter(|pass| pass.stage() == stage) {
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

fn precedes(left: &dyn FramePass, right: &dyn FramePass, registered_first: bool) -> bool {
    left.stage() < right.stage()
        || (left.stage() == right.stage()
            && left.writes().iter().any(|id| {
                right.reads().contains(id)
                    && (registered_first
                        || !left.reads().contains(id)
                        || !right.writes().contains(id))
            }))
}

fn topological(passes: &[Box<dyn FramePass>]) -> Option<Vec<usize>> {
    let count = passes.len();
    let mut waiting = vec![0usize; count];
    for (consumer, degree) in waiting.iter_mut().enumerate() {
        *degree = (0..count)
            .filter(|&producer| {
                producer != consumer
                    && precedes(
                        passes[producer].as_ref(),
                        passes[consumer].as_ref(),
                        producer < consumer,
                    )
            })
            .count();
    }
    let mut order = Vec::with_capacity(count);
    let mut placed = vec![false; count];
    while order.len() < count {
        let next = (0..count).find(|&index| !placed[index] && waiting[index] == 0)?;
        placed[next] = true;
        order.push(next);
        for consumer in 0..count {
            if !placed[consumer]
                && precedes(
                    passes[next].as_ref(),
                    passes[consumer].as_ref(),
                    next < consumer,
                )
            {
                waiting[consumer] -= 1;
            }
        }
    }
    Some(order)
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
            &self,
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
        reads: &'static [ResourceId],
        writes: &'static [ResourceId],
        convention: Option<DepthConvention>,
        stage: PassStage,
    }

    impl FramePass for Probe {
        fn name(&self) -> &'static str {
            self.name
        }

        fn reads(&self) -> &[ResourceId] {
            self.reads
        }

        fn writes(&self) -> &[ResourceId] {
            self.writes
        }

        fn stage(&self) -> PassStage {
            self.stage
        }

        fn depth_convention(&self) -> Option<DepthConvention> {
            self.convention
        }

        fn record(
            &self,
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

    fn probe(
        name: &'static str,
        reads: &'static [ResourceId],
        writes: &'static [ResourceId],
    ) -> Box<Probe> {
        Box::new(Probe {
            name,
            reads,
            writes,
            convention: None,
            stage: PassStage::Scene,
        })
    }

    #[test]
    fn a_consumer_registered_first_still_runs_after_the_pass_it_reads() {
        const GLOW: ResourceId = "glow";
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        schedule.register(probe("consumer", &[GLOW], &[])).unwrap();
        schedule.register(probe("producer", &[], &[GLOW])).unwrap();
        assert_eq!(
            schedule.names().collect::<Vec<_>>(),
            ["producer", "consumer"]
        );
    }

    #[test]
    fn a_transform_registered_after_its_source_and_its_sink_is_not_a_cycle() {
        const RAW: ResourceId = "raw";
        const TONED: ResourceId = "toned";
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        schedule.register(probe("source", &[], &[RAW])).unwrap();
        schedule.register(probe("sink", &[TONED], &[])).unwrap();
        schedule
            .register(probe("transform", &[RAW], &[TONED]))
            .unwrap();
        assert_eq!(
            schedule.names().collect::<Vec<_>>(),
            ["source", "transform", "sink"]
        );
    }

    #[test]
    fn a_pass_writing_depth_under_the_other_convention_is_refused_at_registration() {
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        let refused = schedule.register(Box::new(Probe {
            name: "standard-z overlay",
            reads: &[],
            writes: &[SCENE_DEPTH],
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
    fn overlay_stage_follows_scene_with_reverse_registration() {
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        schedule
            .register(Box::new(Probe {
                name: "overlay",
                reads: &[SCENE_COLOR],
                writes: &[SCENE_COLOR],
                convention: None,
                stage: PassStage::Overlay,
            }))
            .unwrap();
        schedule
            .register(probe("scene", &[], &[SCENE_COLOR]))
            .unwrap();
        assert_eq!(schedule.names().collect::<Vec<_>>(), ["scene", "overlay"]);
    }

    #[test]
    fn two_passes_each_reading_the_others_output_refuse_the_second_as_a_cycle() {
        const LEFT: ResourceId = "left";
        const RIGHT: ResourceId = "right";
        const SHARED: ResourceId = "shared";
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        schedule
            .register(probe("left", &[RIGHT, SHARED], &[LEFT, SHARED]))
            .unwrap();
        let refused = schedule.register(probe("right", &[LEFT, SHARED], &[RIGHT, SHARED]));
        assert_eq!(refused, Err(PassError::Cycle { pass: "right" }));
        assert_eq!(schedule.names().collect::<Vec<_>>(), ["left"]);
    }
}
