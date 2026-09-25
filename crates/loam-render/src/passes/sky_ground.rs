use std::cell::RefCell;
use std::rc::Rc;

use loam_runtime::Eye;
use wgpu::{CommandEncoder, Queue};

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

    fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
        let Some(depth) = target.depth else {
            return Ok(());
        };
        let state = &mut *self.shared.borrow_mut();
        let (Some(node), Some(queue)) = (state.node.as_mut(), state.queue.as_ref()) else {
            return Ok(());
        };
        let viewport = Viewport::full([target.size.0, target.size.1]);
        node.set_uniforms(
            queue,
            &SkyGroundUniforms::new(
                crate::view::root_view_projection(&state.eye),
                viewport,
                state.sky,
                state.ground,
            ),
        );
        node.record(encoder, target.color, depth, Some(&viewport));
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
