use std::fmt;
use std::time::Duration;

use web_time::Instant;
use wgpu::{CommandEncoder, TextureFormat, TextureView};

use crate::device::{GpuContext, MissingGpuCapability};
use crate::gpu_timer::SectionTimer;
use crate::DepthConvention;

pub type ResourceId = &'static str;

pub const SCENE_COLOR: ResourceId = "scene-color";
pub const SCENE_DEPTH: ResourceId = "scene-depth";

const SCENE_OUTPUTS: [ResourceId; 2] = [SCENE_COLOR, SCENE_DEPTH];

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PassOrder {
    BeforeScene,
    AfterScene,
}

pub struct FrameTarget<'a> {
    pub color: &'a TextureView,
    pub depth: Option<&'a TextureView>,
    pub size: (u32, u32),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FrameFormat {
    pub color: TextureFormat,
    pub depth: TextureFormat,
    pub sample_count: u32,
}

pub trait FramePass {
    fn name(&self) -> &'static str;

    fn reads(&self) -> &[ResourceId] {
        &[]
    }

    fn writes(&self) -> &[ResourceId] {
        &[]
    }

    fn order(&self) -> PassOrder;

    /// `None` means the pass writes no depth; `Some` must match the frame's convention to register.
    fn depth_convention(&self) -> Option<DepthConvention> {
        None
    }

    fn record(&self, encoder: &mut CommandEncoder, target: &FrameTarget<'_>);

    fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> Result<(), MissingGpuCapability> {
        let _ = frame;
        self.rebuild(gpu)
    }

    fn rebuild(&mut self, gpu: &GpuContext) -> Result<(), MissingGpuCapability>;
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
    SceneOutput {
        pass: &'static str,
        resource: ResourceId,
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
            Self::SceneOutput { pass, resource } => write!(
                f,
                "pass `{pass}` runs before the scene draw, which writes `{resource}`"
            ),
        }
    }
}

impl std::error::Error for PassError {}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GpuTime {
    /// No timer, no free slot, or no result mapped yet; never a zero duration.
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
}

impl PassSchedule {
    pub fn new(convention: DepthConvention) -> Self {
        Self {
            convention,
            passes: Vec::new(),
            sections: Vec::new(),
            timer: None,
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

    /// Inserts after every pass it must follow and before every pass that reads its writes; a depth writer under another convention is refused.
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
        if pass.order() == PassOrder::BeforeScene {
            if let Some(resource) = pass
                .reads()
                .iter()
                .chain(pass.writes())
                .find(|id| SCENE_OUTPUTS.contains(id))
            {
                return Err(PassError::SceneOutput {
                    pass: pass.name(),
                    resource,
                });
            }
        }
        let shares =
            |left: &[ResourceId], right: &[ResourceId]| left.iter().any(|id| right.contains(id));
        let after = self
            .passes
            .iter()
            .enumerate()
            .filter(|(_, other)| {
                other.order() < pass.order()
                    || (other.order() == pass.order() && shares(other.writes(), pass.reads()))
            })
            .map(|(index, _)| index + 1)
            .max()
            .unwrap_or(0);
        let before = self
            .passes
            .iter()
            .enumerate()
            .filter(|(_, other)| {
                other.order() > pass.order()
                    || (other.order() == pass.order() && shares(other.reads(), pass.writes()))
            })
            .map(|(index, _)| index)
            .min()
            .unwrap_or(self.passes.len());
        if after > before {
            return Err(PassError::Cycle { pass: pass.name() });
        }
        self.passes.insert(after, pass);
        Ok(())
    }

    pub fn rebuild(
        &mut self,
        gpu: &GpuContext,
        frame: FrameFormat,
    ) -> Result<(), MissingGpuCapability> {
        self.timer = SectionTimer::new(&gpu.device, &gpu.queue);
        for pass in self.passes.iter_mut() {
            pass.attach(gpu, frame)?;
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
        order: PassOrder,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) {
        let timer = &mut self.timer;
        let sections = &mut self.sections;
        for pass in self.passes.iter().filter(|pass| pass.order() == order) {
            time_section(timer, sections, pass.name(), encoder, |encoder| {
                pass.record(encoder, target)
            });
        }
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

fn time_section(
    timer: &mut Option<SectionTimer>,
    sections: &mut Vec<Section>,
    name: &'static str,
    encoder: &mut CommandEncoder,
    body: impl FnOnce(&mut CommandEncoder),
) {
    let _scope = loam_time::frame_trace::scope(name);
    let slot = timer.as_mut().and_then(|timer| timer.open(encoder, name));
    let started = Instant::now();
    body(encoder);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Probe {
        name: &'static str,
        reads: &'static [ResourceId],
        writes: &'static [ResourceId],
        convention: Option<DepthConvention>,
        order: PassOrder,
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

        fn order(&self) -> PassOrder {
            self.order
        }

        fn depth_convention(&self) -> Option<DepthConvention> {
            self.convention
        }

        fn record(&self, _encoder: &mut CommandEncoder, _target: &FrameTarget<'_>) {}

        fn rebuild(&mut self, _gpu: &GpuContext) -> Result<(), MissingGpuCapability> {
            Ok(())
        }
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
            order: PassOrder::AfterScene,
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
    fn a_pass_writing_depth_under_the_other_convention_is_refused_at_registration() {
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        let refused = schedule.register(Box::new(Probe {
            name: "standard-z overlay",
            reads: &[],
            writes: &[SCENE_DEPTH],
            convention: Some(DepthConvention::StandardZ),
            order: PassOrder::AfterScene,
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
    fn a_pass_reading_the_scene_before_the_scene_draws_it_is_refused_at_registration() {
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        let refused = schedule.register(Box::new(Probe {
            name: "early readback",
            reads: &[SCENE_COLOR],
            writes: &[],
            convention: None,
            order: PassOrder::BeforeScene,
        }));
        assert_eq!(
            refused,
            Err(PassError::SceneOutput {
                pass: "early readback",
                resource: SCENE_COLOR,
            })
        );
        assert_eq!(schedule.names().count(), 0);
    }

    #[test]
    fn two_passes_each_reading_the_others_output_refuse_the_second_as_a_cycle() {
        const LEFT: ResourceId = "left";
        const RIGHT: ResourceId = "right";
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        schedule.register(probe("left", &[RIGHT], &[LEFT])).unwrap();
        let refused = schedule.register(probe("right", &[LEFT], &[RIGHT]));
        assert_eq!(refused, Err(PassError::Cycle { pass: "right" }));
        assert_eq!(schedule.names().collect::<Vec<_>>(), ["left"]);
    }
}
