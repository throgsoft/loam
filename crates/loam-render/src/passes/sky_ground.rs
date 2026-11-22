use std::cell::RefCell;
use std::rc::Rc;

use loam_runtime::Eye;
use wgpu::{Color, CommandEncoder, Queue, RenderPass};

use crate::device::GpuContext;
use crate::pass::{ColorLoad, FrameFormat, FramePass, FrameTarget, PassStage};
use crate::sky_ground::{Ground, Sky, SkyGroundNode, SkyGroundUniforms, DEFAULT_SKY};
use crate::{DepthConvention, Viewport};

struct State {
    eye: Eye,
    sky: Sky,
    ground: Ground,
    node: Option<SkyGroundNode>,
    queue: Option<Queue>,
}

#[derive(Clone)]
pub struct SkyGroundPass {
    shared: Rc<RefCell<State>>,
}

impl SkyGroundPass {
    pub fn new(ground: Ground) -> Self {
        Self {
            shared: Rc::new(RefCell::new(State {
                eye: Eye::default(),
                sky: DEFAULT_SKY,
                ground,
                node: None,
                queue: None,
            })),
        }
    }

    pub fn publish(&self, eye: &Eye, sky: Sky, ground: Ground) {
        let mut state = self.shared.borrow_mut();
        state.eye = *eye;
        state.sky = sky;
        state.ground = ground;
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
        ColorLoad::Clear
    }

    fn depth_convention(&self) -> Option<DepthConvention> {
        Some(DepthConvention::ReversedZ)
    }

    fn shares_pass(&self) -> bool {
        true
    }

    fn clear_color(&self) -> Color {
        self.shared.borrow().sky.horizon()
    }

    fn prepare(
        &mut self,
        _encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
        let state = &mut *self.shared.borrow_mut();
        let (Some(node), Some(queue)) = (state.node.as_mut(), state.queue.as_ref()) else {
            return Ok(());
        };
        node.set_uniforms(
            queue,
            &SkyGroundUniforms::new(
                crate::view::root_view_projection(&state.eye),
                Viewport::full([target.size.0, target.size.1]),
                state.sky,
                state.ground,
            ),
        );
        Ok(())
    }

    fn draw(&mut self, pass: &mut RenderPass<'_>, target: &FrameTarget<'_>) -> anyhow::Result<()> {
        if let Some(node) = self.shared.borrow().node.as_ref() {
            node.draw(pass, Some(&Viewport::full([target.size.0, target.size.1])));
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
        if let Some(node) = self.shared.borrow().node.as_ref() {
            node.record(
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
        state.node = Some(SkyGroundNode::new(
            &gpu.device,
            frame.color,
            frame.depth,
            DepthConvention::ReversedZ,
        ));
        state.queue = Some(gpu.queue.clone());
        Ok(())
    }
}
