use std::cell::RefCell;
use std::rc::Rc;

use loam_runtime::Eye;
use wgpu::{CommandEncoder, Queue};

use crate::device::GpuContext;
use crate::pass::{FrameFormat, FramePass, FrameTarget, PassStage, ResourceId, SCENE_BASE};
use crate::sky_ground::{Ground, SkyGroundNode, SkyGroundUniforms};
use crate::{DepthConvention, Viewport};

struct State {
    eye: Eye,
    ground: Ground,
    node: Option<SkyGroundNode>,
    queue: Option<Queue>,
}

const BASE: [ResourceId; 1] = [SCENE_BASE];

/// A shared scene background pass with a sky and reversed Z floor.
#[derive(Clone)]
pub struct SkyGroundPass {
    shared: Rc<RefCell<State>>,
}

impl SkyGroundPass {
    pub fn new(ground: Ground) -> Self {
        Self {
            shared: Rc::new(RefCell::new(State {
                eye: Eye::default(),
                ground,
                node: None,
                queue: None,
            })),
        }
    }

    /// Stores the eye and the ground; the record builds the uniforms from them.
    pub fn publish(&self, eye: &Eye, ground: Ground) {
        let mut state = self.shared.borrow_mut();
        state.eye = *eye;
        state.ground = ground;
    }
}

impl FramePass for SkyGroundPass {
    fn name(&self) -> &'static str {
        "sky-ground"
    }

    fn reads(&self) -> &[ResourceId] {
        &BASE
    }

    fn writes(&self) -> &[ResourceId] {
        &BASE
    }

    fn stage(&self) -> PassStage {
        PassStage::Scene
    }

    fn depth_convention(&self) -> Option<DepthConvention> {
        Some(DepthConvention::ReversedZ)
    }

    fn record(&self, encoder: &mut CommandEncoder, target: &FrameTarget<'_>) -> anyhow::Result<()> {
        let Some(depth) = target.depth else {
            return Ok(());
        };
        let state = self.shared.borrow();
        let (Some(node), Some(queue)) = (state.node.as_ref(), state.queue.as_ref()) else {
            return Ok(());
        };
        let viewport = Viewport::full([target.size.0, target.size.1]);
        node.set_uniforms(
            queue,
            &SkyGroundUniforms::new(
                crate::view::root_view_projection(&state.eye),
                viewport,
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
            frame.sample_count,
        ));
        state.queue = Some(gpu.queue.clone());
        Ok(())
    }
}
