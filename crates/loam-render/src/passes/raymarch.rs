use std::cell::RefCell;
use std::rc::Rc;

use wgpu::{CommandEncoder, Queue, TextureFormat};

use crate::device::{GpuContext, MissingGpuCapability};
use crate::pass::{FrameFormat, FramePass, FrameTarget, PassOrder};
use crate::raymarch::{RayMarchNode, RayMarchUniforms};
use crate::Viewport;

struct State {
    source: String,
    format: TextureFormat,
    depth: TextureFormat,
    sample_count: u32,
    uniforms: RayMarchUniforms,
    node: Option<RayMarchNode>,
    queue: Option<Queue>,
}

#[derive(Clone)]
pub struct RaymarchPass {
    shared: Rc<RefCell<State>>,
}

impl RaymarchPass {
    pub fn new(source: String) -> Self {
        Self {
            shared: Rc::new(RefCell::new(State {
                source,
                format: TextureFormat::Rgba8UnormSrgb,
                depth: crate::view::DEPTH_FORMAT,
                sample_count: 1,
                uniforms: RayMarchUniforms::default(),
                node: None,
                queue: None,
            })),
        }
    }

    pub fn publish(&self, uniforms: RayMarchUniforms) {
        self.shared.borrow_mut().uniforms = uniforms;
    }
}

impl FramePass for RaymarchPass {
    fn name(&self) -> &'static str {
        "raymarch"
    }

    fn order(&self) -> PassOrder {
        PassOrder::BeforeScene
    }

    fn record(&self, encoder: &mut CommandEncoder, target: &FrameTarget<'_>) {
        let mut state = self.shared.borrow_mut();
        let State {
            uniforms,
            node,
            queue,
            ..
        } = &mut *state;
        let (Some(node), Some(queue)) = (node.as_mut(), queue.as_ref()) else {
            return;
        };
        uniforms.resolution = [target.size.0 as f32, target.size.1 as f32];
        *node.uniforms_mut() = *uniforms;
        node.flush_uniforms(queue);
        node.record(
            encoder,
            target.color,
            None,
            Viewport::full([target.size.0, target.size.1]),
        );
    }

    fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> Result<(), MissingGpuCapability> {
        {
            let mut state = self.shared.borrow_mut();
            state.format = frame.color;
            state.depth = frame.depth;
            state.sample_count = frame.sample_count;
        }
        self.rebuild(gpu)
    }

    fn rebuild(&mut self, gpu: &GpuContext) -> Result<(), MissingGpuCapability> {
        let mut state = self.shared.borrow_mut();
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("loam-render::passes::raymarch"),
                source: wgpu::ShaderSource::Wgsl(state.source.clone().into()),
            });
        state.node = Some(RayMarchNode::new(
            &gpu.device,
            state.format,
            &module,
            state.sample_count,
        ));
        state.queue = Some(gpu.queue.clone());
        Ok(())
    }
}
