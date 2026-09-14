use std::cell::RefCell;
use std::rc::Rc;

use wgpu::{CommandEncoder, Device, Queue};

use crate::device::GpuContext;
use crate::pass::{ColorLoad, FrameFormat, FramePass, FrameTarget, PassStage};
use crate::raymarch::{BodyUniform, Hyperslice4DNode, Hyperslice4DUniforms};
use crate::{DepthConvention, DepthMode, Viewport};

struct State {
    enabled: bool,
    source: String,
    uniforms: Hyperslice4DUniforms,
    bodies: Vec<BodyUniform>,
    cells: Vec<(Viewport, f32, BodyUniform)>,
    node: Option<Hyperslice4DNode>,
    device: Option<Device>,
    queue: Option<Queue>,
}

/// A shared hyperslice pass that writes scene color and reversed Z depth.
#[derive(Clone)]
pub struct HyperslicePass {
    shared: Rc<RefCell<State>>,
}

impl HyperslicePass {
    pub fn new(source: String) -> Self {
        Self {
            shared: Rc::new(RefCell::new(State {
                enabled: true,
                source,
                uniforms: Hyperslice4DUniforms::default(),
                bodies: Vec::new(),
                cells: Vec::new(),
                node: None,
                device: None,
                queue: None,
            })),
        }
    }

    /// Stores the uniforms and copies the bodies; the record fills in the resolution and flushes both.
    pub fn publish(&self, uniforms: Hyperslice4DUniforms, bodies: &[BodyUniform]) {
        let mut state = self.shared.borrow_mut();
        state.uniforms = uniforms;
        state.bodies.clear();
        state.bodies.extend_from_slice(bodies);
    }

    /// One draw per cell within the frame, each with its own w and body; an empty strip returns the pass to its full-frame draw.
    pub fn publish_strip(&self, cells: &[(Viewport, f32, BodyUniform)]) {
        let mut state = self.shared.borrow_mut();
        state.cells.clear();
        state.cells.extend_from_slice(cells);
    }

    pub fn body_count(&self) -> usize {
        self.shared.borrow().bodies.len()
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.shared.borrow_mut().enabled = enabled;
    }

    pub fn strip_cells(&self) -> usize {
        self.shared.borrow().cells.len()
    }
}

impl FramePass for HyperslicePass {
    fn name(&self) -> &'static str {
        "hyperslice"
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

    fn depth_read(&self) -> Option<DepthConvention> {
        Some(DepthConvention::ReversedZ)
    }

    fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
        let mut state = self.shared.borrow_mut();
        if !state.enabled {
            return Ok(());
        }
        let State {
            uniforms,
            bodies,
            cells,
            node,
            device,
            queue,
            ..
        } = &mut *state;
        let (Some(node), Some(device), Some(queue)) =
            (node.as_mut(), device.as_ref(), queue.as_ref())
        else {
            return Ok(());
        };
        uniforms.resolution = [target.size.0 as f32, target.size.1 as f32];
        uniforms.viewport_origin = [0.0, 0.0];
        *node.uniforms_mut() = *uniforms;
        if !cells.is_empty() {
            node.record_strip(device, queue, encoder, target.color, cells)?;
            return Ok(());
        }
        let Some(depth) = target.depth else {
            return Ok(());
        };
        node.set_bodies(bodies);
        node.flush_uniforms(queue);
        node.record(
            encoder,
            target.color,
            Some(depth),
            Viewport::full([target.size.0, target.size.1]),
        );
        Ok(())
    }

    fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> anyhow::Result<()> {
        let mut state = self.shared.borrow_mut();
        crate::shader::validate_hyperslice_wgsl(&state.source)?;
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("loam-render::passes::hyperslice"),
                source: wgpu::ShaderSource::Wgsl(state.source.clone().into()),
            });
        state.node = Some(Hyperslice4DNode::with_depth(
            &gpu.device,
            frame.color,
            &module,
            DepthMode::ReadWrite {
                format: frame.depth,
            },
        ));
        state.device = Some(gpu.device.clone());
        state.queue = Some(gpu.queue.clone());
        Ok(())
    }
}
