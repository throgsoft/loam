use std::cell::RefCell;
use std::rc::Rc;

use loam_runtime::{Backdrop, Eye, Ground, Sky};
use wgpu::{Color, CommandEncoder, Queue, RenderPass};

use super::{horizon, SkyGroundNode, SkyGroundUniforms, DEFAULT_SKY};
use crate::device::GpuContext;
use crate::pass::{ColorLoad, FrameFormat, FramePass, FrameTarget, PassStage, PublishedFrame};
use crate::{DepthConvention, Viewport};

struct State {
    eye: Eye,
    backdrop: Option<Backdrop>,
    built: Option<Built>,
}

struct Built {
    node: SkyGroundNode,
    queue: Queue,
}

/// Without a backdrop it neither clears nor draws, so the frame keeps the host's background.
#[derive(Clone)]
pub struct SkyGroundPass {
    reads_backdrop: bool,
    shared: Rc<RefCell<State>>,
}

impl SkyGroundPass {
    pub fn new(ground: Ground) -> Self {
        Self::showing(
            Some(Backdrop {
                sky: DEFAULT_SKY,
                ground,
            }),
            false,
        )
    }

    /// The pass that draws the backdrop each published frame carries.
    pub(crate) fn backdrop() -> Self {
        Self::showing(None, true)
    }

    fn showing(backdrop: Option<Backdrop>, reads_backdrop: bool) -> Self {
        Self {
            reads_backdrop,
            shared: Rc::new(RefCell::new(State {
                eye: Eye::default(),
                backdrop,
                built: None,
            })),
        }
    }

    pub fn publish(&self, eye: &Eye, sky: Sky, ground: Ground) {
        let mut state = self.shared.borrow_mut();
        state.eye = *eye;
        state.backdrop = Some(Backdrop { sky, ground });
    }
}

impl FramePass for SkyGroundPass {
    fn name(&self) -> &'static str {
        "sky-ground"
    }

    fn stage(&self) -> PassStage {
        PassStage::Background
    }

    fn color_load(&self) -> ColorLoad {
        if self.shared.borrow().backdrop.is_some() {
            ColorLoad::Clear
        } else {
            ColorLoad::Load
        }
    }

    fn depth_convention(&self) -> Option<DepthConvention> {
        Some(DepthConvention::ReversedZ)
    }

    fn read(&mut self, frame: &PublishedFrame<'_>) {
        if self.reads_backdrop {
            let mut state = self.shared.borrow_mut();
            state.eye = frame.eye;
            state.backdrop = frame.backdrop;
        }
    }

    fn shares_pass(&self) -> bool {
        true
    }

    fn clear_color(&self) -> Color {
        horizon(self.shared.borrow().backdrop.map_or(DEFAULT_SKY, |b| b.sky))
    }

    fn prepare(
        &mut self,
        _encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
        let state = &mut *self.shared.borrow_mut();
        let (Some(built), Some(backdrop)) = (state.built.as_mut(), state.backdrop) else {
            return Ok(());
        };
        built.node.set_uniforms(
            &built.queue,
            &SkyGroundUniforms::new(
                crate::depth::root_view_projection(&state.eye),
                Viewport::full([target.size.0, target.size.1]),
                backdrop.sky,
                backdrop.ground,
            ),
        );
        Ok(())
    }

    fn draw(&mut self, pass: &mut RenderPass<'_>, target: &FrameTarget<'_>) -> anyhow::Result<()> {
        let state = self.shared.borrow();
        if let (Some(built), Some(_)) = (state.built.as_ref(), state.backdrop) {
            built
                .node
                .draw(pass, Some(&Viewport::full([target.size.0, target.size.1])));
        }
        Ok(())
    }

    fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
        let Some(depth) = target.depth else {
            return Ok(());
        };
        self.prepare(encoder, target)?;
        let state = self.shared.borrow();
        if let (Some(built), Some(_)) = (state.built.as_ref(), state.backdrop) {
            built.node.record(
                encoder,
                target.color,
                depth,
                Some(&Viewport::full([target.size.0, target.size.1])),
            );
        }
        Ok(())
    }

    fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> anyhow::Result<()> {
        let mut state = self.shared.borrow_mut();
        state.built = Some(Built {
            node: SkyGroundNode::new(
                &gpu.device,
                frame.color,
                frame.depth,
                DepthConvention::ReversedZ,
            ),
            queue: gpu.queue.clone(),
        });
        Ok(())
    }
}
