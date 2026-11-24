use std::cell::RefCell;
use std::rc::Rc;

use loam_runtime::{Eye, PublishedView};
use wgpu::{CommandEncoder, Queue};

use crate::device::GpuContext;
use crate::pass::{ColorLoad, FrameFormat, FramePass, FrameTarget, PassStage};
use crate::raymarch::{RayMarchNode, RayMarchUniforms};
use crate::Viewport;

struct State {
    source: String,
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
                uniforms: RayMarchUniforms::default(),
                node: None,
                queue: None,
            })),
        }
    }

    pub fn set_params(&self, params: [f32; 4]) {
        self.shared.borrow_mut().uniforms.params = params;
    }
}

impl FramePass for RaymarchPass {
    fn name(&self) -> &'static str {
        "raymarch"
    }

    fn stage(&self) -> PassStage {
        PassStage::Background
    }

    fn color_load(&self) -> ColorLoad {
        ColorLoad::Clear
    }

    fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
        let mut state = self.shared.borrow_mut();
        let State {
            uniforms,
            node,
            queue,
            ..
        } = &mut *state;
        let (Some(node), Some(queue)) = (node.as_mut(), queue.as_ref()) else {
            return Ok(());
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
        Ok(())
    }

    fn read_publication(&mut self, eye: &Eye, _views: &[PublishedView]) {
        let uniforms = &mut self.shared.borrow_mut().uniforms;
        *uniforms = RayMarchUniforms {
            params: uniforms.params,
            time: uniforms.time,
            tick: uniforms.tick,
            ..RayMarchUniforms::from_eye(eye)
        };
    }

    fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> anyhow::Result<()> {
        let mut state = self.shared.borrow_mut();
        crate::shader::validate_raymarch_wgsl(&state.source)?;
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("loam-render::passes::raymarch"),
                source: wgpu::ShaderSource::Wgsl(state.source.clone().into()),
            });
        state.node = Some(RayMarchNode::new(&gpu.device, frame.color, &module));
        state.queue = Some(gpu.queue.clone());
        Ok(())
    }
}
