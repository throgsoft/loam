use std::cell::RefCell;
use std::rc::Rc;

use loam_runtime::Eye;
use wgpu::{CommandEncoder, Queue, TextureFormat};

use crate::device::{GpuContext, MissingGpuCapability};
use crate::pass::{FramePass, FrameTarget, PassOrder};
use crate::sky_ground::{Ground, SkyGroundNode, SkyGroundUniforms};
use crate::view::DEPTH_FORMAT;
use crate::{DepthConvention, Viewport};

struct State {
    format: TextureFormat,
    sample_count: u32,
    eye: Eye,
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
                format: TextureFormat::Rgba8UnormSrgb,
                sample_count: 1,
                eye: Eye::default(),
                ground,
                node: None,
                queue: None,
            })),
        }
    }

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

    fn order(&self) -> PassOrder {
        PassOrder::BeforeScene
    }

    fn depth_convention(&self) -> Option<DepthConvention> {
        Some(DepthConvention::ReversedZ)
    }

    fn target(&mut self, format: TextureFormat, sample_count: u32) {
        let mut state = self.shared.borrow_mut();
        state.format = format;
        state.sample_count = sample_count;
    }

    fn record(&self, encoder: &mut CommandEncoder, target: &FrameTarget<'_>) {
        let Some(depth) = target.depth else {
            return;
        };
        let state = self.shared.borrow();
        let (Some(node), Some(queue)) = (state.node.as_ref(), state.queue.as_ref()) else {
            return;
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
    }

    fn rebuild(&mut self, gpu: &GpuContext) -> Result<(), MissingGpuCapability> {
        let mut state = self.shared.borrow_mut();
        state.node = Some(SkyGroundNode::new(
            &gpu.device,
            state.format,
            DEPTH_FORMAT,
            DepthConvention::ReversedZ,
            state.sample_count,
        ));
        state.queue = Some(gpu.queue.clone());
        Ok(())
    }
}
