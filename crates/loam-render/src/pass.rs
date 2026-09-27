use std::fmt;
use std::time::Duration;

use web_time::Instant;
use wgpu::{
    Color, CommandEncoder, LoadOp, Operations, RenderPass, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, StoreOp, TextureFormat, TextureView,
};

use crate::device::{GpuContext, LossSignal, UncapturedGpuError};
use crate::gpu_timer::SectionTimer;
use crate::view::DEPTH_CLEAR;
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

    /// True joins the stage's shared render pass, so `attach` must build pipelines against the frame's color and depth formats.
    fn shares_pass(&self) -> bool {
        false
    }

    /// Read before `prepare`, and only when `color_load` is `Clear`.
    fn clear_color(&self) -> Color {
        Color::BLACK
    }

    /// Encoder work for a sharing pass; runs just before its render pass opens, not at frame start.
    fn prepare(
        &mut self,
        _encoder: &mut CommandEncoder,
        _target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Runs inside the shared pass, which owns the load ops; viewport and scissor carry over from earlier draws.
    fn draw(
        &mut self,
        _pass: &mut RenderPass<'_>,
        _target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> anyhow::Result<()>;

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
            record_alone(signal, timer, sections, pass.as_mut(), encoder, target)?;
        }
        Ok(())
    }

    /// Replaces `record` for Background and Scene and includes the frame clear.
    pub fn record_scene(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
        background: Color,
    ) -> Result<(), PassExecutionError> {
        let depth_clear = depth_clear(self.convention);
        if self.unattached.is_some() {
            self.clear(encoder, target, background, depth_clear);
            return Ok(());
        }
        let mut plan = Plan::new(
            &self.passes,
            target.depth.is_some(),
            background,
            depth_clear,
        );
        while let Some(step) = plan.next(&self.passes) {
            match step {
                Step::Clear => self.clear(encoder, target, background, depth_clear),
                Step::Alone(index) => record_alone(
                    self.signal.as_deref(),
                    &mut self.timer,
                    &mut self.sections,
                    self.passes[index].as_mut(),
                    encoder,
                    target,
                )?,
                Step::Shared(run) => self.record_shared(run, encoder, target)?,
            }
        }
        Ok(())
    }

    fn clear(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
        background: Color,
        depth_clear: f32,
    ) {
        self.section("present-clear", encoder, |encoder| {
            encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("loam-render present clear"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target.color,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(background),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: target.depth.map(|view| {
                    RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(Operations {
                            load: LoadOp::Clear(depth_clear),
                            store: StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        });
    }

    fn record_shared(
        &mut self,
        run: Run,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> Result<(), PassExecutionError> {
        let Some(depth) = target.depth else {
            return Ok(());
        };
        let _scope = loam_time::frame_trace::scope(SCENE_PASS);
        let started = Instant::now();
        let signal = self.signal.as_deref();
        let timer = &mut self.timer;
        let sections = &mut self.sections;
        let passes = &mut self.passes[run.start..run.end];
        let first = sections.len();
        sections.push(Section {
            name: SCENE_PASS,
            cpu: Duration::ZERO,
            gpu: GpuTime::Unavailable,
        });
        for pass in passes.iter_mut() {
            let name = pass.name();
            let begun = Instant::now();
            let outcome = run_pass(signal, name, PassPhase::Record, || {
                pass.prepare(encoder, target)
            });
            sections.push(Section {
                name,
                cpu: begun.elapsed(),
                gpu: GpuTime::Unavailable,
            });
            outcome?;
        }
        // Timestamps resolve only at render pass boundaries, so members share one GPU slot.
        let slot = timer
            .as_mut()
            .and_then(|timer| timer.open(encoder, SCENE_PASS));
        let mut outcome = Ok(());
        {
            let mut render = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("loam-render scene pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target.color,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: run.color,
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(Operations {
                        load: run.depth,
                        store: run.depth_store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            for (offset, pass) in passes.iter_mut().enumerate() {
                let begun = Instant::now();
                outcome = run_pass(signal, pass.name(), PassPhase::Record, || {
                    pass.draw(&mut render, target)
                });
                sections[first + 1 + offset].cpu += begun.elapsed();
                if outcome.is_err() {
                    break;
                }
            }
        }
        let gpu = match (timer.as_mut(), slot) {
            (Some(timer), Some(slot)) => {
                timer.close(encoder, slot);
                timer
                    .elapsed(slot)
                    .map_or(GpuTime::Unavailable, GpuTime::Measured)
            }
            _ => GpuTime::Unavailable,
        };
        sections[first].cpu = started.elapsed();
        sections[first].gpu = gpu;
        outcome
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

fn record_alone(
    signal: Option<&LossSignal>,
    timer: &mut Option<SectionTimer>,
    sections: &mut Vec<Section>,
    pass: &mut dyn FramePass,
    encoder: &mut CommandEncoder,
    target: &FrameTarget<'_>,
) -> Result<(), PassExecutionError> {
    let name = pass.name();
    run_pass(signal, name, PassPhase::Record, || {
        time_section(timer, sections, name, encoder, |encoder| {
            pass.record(encoder, target)
        })
    })
}

const SCENE_PASS: &str = "scene-pass";

fn depth_clear(convention: DepthConvention) -> f32 {
    match convention {
        DepthConvention::StandardZ => 1.0,
        DepthConvention::ReversedZ => DEPTH_CLEAR,
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Step {
    Clear,
    Alone(usize),
    Shared(Run),
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct Run {
    start: usize,
    end: usize,
    color: LoadOp<Color>,
    depth: LoadOp<f32>,
    depth_store: StoreOp,
}

struct Plan {
    scene_end: usize,
    shareable: bool,
    background: Color,
    depth_clear: f32,
    at: usize,
    opened: bool,
    depth_written: bool,
}

impl Plan {
    fn new(
        passes: &[Box<dyn FramePass>],
        shareable: bool,
        background: Color,
        depth_clear: f32,
    ) -> Self {
        Self {
            scene_end: passes.partition_point(|pass| pass.stage() < PassStage::Overlay),
            shareable,
            background,
            depth_clear,
            at: 0,
            opened: false,
            depth_written: false,
        }
    }

    fn joins(&self, pass: &dyn FramePass) -> bool {
        self.shareable && pass.shares_pass()
    }

    fn next(&mut self, passes: &[Box<dyn FramePass>]) -> Option<Step> {
        let scene = &passes[..self.scene_end];
        if !self.opened {
            self.opened = true;
            if !scene.first().is_some_and(|pass| self.joins(pass.as_ref())) {
                return Some(Step::Clear);
            }
        }
        let start = self.at;
        let first = scene.get(start)?;
        if !self.joins(first.as_ref()) {
            self.at += 1;
            self.depth_written |= first.depth_convention().is_some();
            return Some(Step::Alone(start));
        }
        let end = start
            + 1
            + scene[start + 1..]
                .iter()
                .take_while(|pass| {
                    self.joins(pass.as_ref()) && pass.color_load() == ColorLoad::Load
                })
                .count();
        let color = match first.color_load() {
            ColorLoad::Clear => LoadOp::Clear(first.clear_color()),
            ColorLoad::Load if start == 0 => LoadOp::Clear(self.background),
            ColorLoad::Load => LoadOp::Load,
        };
        let depth = if self.depth_written {
            LoadOp::Load
        } else {
            LoadOp::Clear(self.depth_clear)
        };
        // Only a sharing pass is held to its depth_read declaration.
        let kept = passes[end..].iter().enumerate().any(|(offset, pass)| {
            end + offset >= self.scene_end
                || !self.joins(pass.as_ref())
                || pass.depth_convention().is_some()
                || pass.depth_read().is_some()
        });
        let writes = scene[start..end]
            .iter()
            .any(|pass| pass.depth_convention().is_some());
        self.depth_written = kept && (self.depth_written || writes);
        self.at = end;
        Some(Step::Shared(Run {
            start,
            end,
            color,
            depth,
            depth_store: if kept {
                StoreOp::Store
            } else {
                StoreOp::Discard
            },
        }))
    }
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

    struct Clearing {
        stage: PassStage,
    }

    impl FramePass for Clearing {
        fn name(&self) -> &'static str {
            "clearing"
        }

        fn stage(&self) -> PassStage {
            self.stage
        }

        fn color_load(&self) -> ColorLoad {
            ColorLoad::Clear
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

    struct Reader;

    impl FramePass for Reader {
        fn name(&self) -> &'static str {
            "reader"
        }

        fn stage(&self) -> PassStage {
            PassStage::Scene
        }

        fn depth_read(&self) -> Option<DepthConvention> {
            Some(DepthConvention::StandardZ)
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

    struct Flaky {
        fail: std::sync::Arc<std::sync::atomic::AtomicBool>,
        recorded: std::sync::Arc<std::sync::atomic::AtomicU32>,
    }

    impl FramePass for Flaky {
        fn name(&self) -> &'static str {
            "flaky"
        }

        fn stage(&self) -> PassStage {
            PassStage::Scene
        }

        fn record(
            &mut self,
            _encoder: &mut CommandEncoder,
            _target: &FrameTarget<'_>,
        ) -> anyhow::Result<()> {
            self.recorded
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }

        fn attach(&mut self, _gpu: &GpuContext, _frame: FrameFormat) -> anyhow::Result<()> {
            if self.fail.load(std::sync::atomic::Ordering::Relaxed) {
                anyhow::bail!("the pass could not build its resources");
            }
            Ok(())
        }
    }

    #[test]
    fn a_pass_that_clears_the_shared_color_target_is_refused_outside_the_background() {
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        assert!(matches!(
            schedule.register(Box::new(Clearing {
                stage: PassStage::Scene
            })),
            Err(PassError::ColorClear {
                pass: "clearing",
                stage: PassStage::Scene
            })
        ));
        assert!(matches!(
            schedule.register(Box::new(Clearing {
                stage: PassStage::Overlay
            })),
            Err(PassError::ColorClear { .. })
        ));
        schedule
            .register(Box::new(Clearing {
                stage: PassStage::Background,
            }))
            .expect("a background pass may clear");
    }

    #[test]
    fn a_depth_reader_under_the_other_convention_is_refused_at_registration() {
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        assert!(matches!(
            schedule.register(Box::new(Reader)),
            Err(PassError::DepthRead {
                pass: "reader",
                declared: DepthConvention::StandardZ,
                frame: DepthConvention::ReversedZ
            })
        ));
    }

    #[test]
    fn a_failed_attach_suppresses_recording_until_every_pass_attaches() {
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
        use std::sync::Arc;

        let gpu = crate::device::noop_context();
        let fail = Arc::new(AtomicBool::new(true));
        let recorded = Arc::new(AtomicU32::new(0));
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        schedule
            .register(Box::new(Flaky {
                fail: fail.clone(),
                recorded: recorded.clone(),
            }))
            .unwrap();
        let frame = FrameFormat {
            color: TextureFormat::Rgba8Unorm,
            depth: TextureFormat::Depth32Float,
        };
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("flaky target"),
            size: wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: frame.color,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let color = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let target = FrameTarget {
            color: &color,
            depth: None,
            size: (8, 8),
        };

        assert!(schedule.attach(&gpu, frame).is_err());
        assert_eq!(schedule.unattached(), Some("flaky"));
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        schedule
            .record(PassStage::Scene, &mut encoder, &target)
            .expect("recording is a no-op while a pass is unattached");
        assert_eq!(recorded.load(Ordering::Relaxed), 0);

        fail.store(false, Ordering::Relaxed);
        schedule.attach(&gpu, frame).expect("the retry attaches");
        assert_eq!(schedule.unattached(), None);
        schedule
            .record(PassStage::Scene, &mut encoder, &target)
            .expect("recorded");
        assert_eq!(recorded.load(Ordering::Relaxed), 1);
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

    struct Planned {
        stage: PassStage,
        shares: bool,
        load: ColorLoad,
        writes: bool,
        reads: bool,
    }

    impl FramePass for Planned {
        fn name(&self) -> &'static str {
            "planned"
        }

        fn stage(&self) -> PassStage {
            self.stage
        }

        fn color_load(&self) -> ColorLoad {
            self.load
        }

        fn depth_convention(&self) -> Option<DepthConvention> {
            self.writes.then_some(DepthConvention::ReversedZ)
        }

        fn depth_read(&self) -> Option<DepthConvention> {
            self.reads.then_some(DepthConvention::ReversedZ)
        }

        fn shares_pass(&self) -> bool {
            self.shares
        }

        fn clear_color(&self) -> Color {
            HORIZON
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

    const HORIZON: Color = Color::GREEN;
    const BACKGROUND: Color = Color::BLUE;

    fn sky() -> Box<dyn FramePass> {
        Box::new(Planned {
            stage: PassStage::Background,
            shares: true,
            load: ColorLoad::Clear,
            writes: true,
            reads: false,
        })
    }

    fn faces() -> Box<dyn FramePass> {
        Box::new(Planned {
            stage: PassStage::Scene,
            shares: true,
            load: ColorLoad::Load,
            writes: true,
            reads: true,
        })
    }

    fn unshared(stage: PassStage, reads: bool) -> Box<dyn FramePass> {
        Box::new(Planned {
            stage,
            shares: false,
            load: ColorLoad::Load,
            writes: false,
            reads,
        })
    }

    fn steps(passes: &[Box<dyn FramePass>]) -> Vec<Step> {
        let mut plan = Plan::new(passes, true, BACKGROUND, DEPTH_CLEAR);
        std::iter::from_fn(|| plan.next(passes)).collect()
    }

    fn shared(
        passes: std::ops::Range<usize>,
        color: LoadOp<Color>,
        depth: LoadOp<f32>,
        depth_store: StoreOp,
    ) -> Step {
        Step::Shared(Run {
            start: passes.start,
            end: passes.end,
            color,
            depth,
            depth_store,
        })
    }

    #[test]
    fn a_shared_pass_that_opens_the_frame_clears_the_depth_the_last_frame_left() {
        assert_eq!(
            steps(&[faces(), faces()]),
            [shared(
                0..2,
                LoadOp::Clear(BACKGROUND),
                LoadOp::Clear(DEPTH_CLEAR),
                StoreOp::Discard
            )]
        );
    }

    #[test]
    fn a_depth_reader_in_the_overlay_keeps_the_scene_depth_stored() {
        let mut passes = vec![sky(), faces(), faces()];
        let opened = |depth_store| {
            [shared(
                0..3,
                LoadOp::Clear(HORIZON),
                LoadOp::Clear(DEPTH_CLEAR),
                depth_store,
            )]
        };
        assert_eq!(steps(&passes), opened(StoreOp::Discard));
        passes.push(unshared(PassStage::Overlay, true));
        assert_eq!(steps(&passes), opened(StoreOp::Store));
    }

    #[test]
    fn a_pass_that_does_not_share_splits_the_run_and_loads_what_was_drawn_before_it() {
        assert_eq!(
            steps(&[sky(), unshared(PassStage::Scene, false), faces()]),
            [
                shared(
                    0..1,
                    LoadOp::Clear(HORIZON),
                    LoadOp::Clear(DEPTH_CLEAR),
                    StoreOp::Store
                ),
                Step::Alone(1),
                shared(2..3, LoadOp::Load, LoadOp::Load, StoreOp::Discard),
            ]
        );
    }
}
