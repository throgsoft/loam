use std::cell::RefCell;
use std::rc::Rc;

use wgpu::{CommandEncoder, Device, Queue, TextureFormat};

use crate::device::{GpuContext, MissingGpuCapability};
use crate::pass::{
    FrameFormat, FramePass, FrameTarget, PassOrder, ResourceId, SCENE_COLOR, SCENE_DEPTH,
};
use crate::raymarch::{BodyUniform, Hyperslice4DNode, Hyperslice4DUniforms};
use crate::{DepthConvention, DepthMode, Viewport};

const WRITES: [ResourceId; 2] = [SCENE_COLOR, SCENE_DEPTH];

struct State {
    source: String,
    format: TextureFormat,
    depth: TextureFormat,
    sample_count: u32,
    uniforms: Hyperslice4DUniforms,
    bodies: Vec<BodyUniform>,
    cells: Vec<(Viewport, f32, BodyUniform)>,
    node: Option<Hyperslice4DNode>,
    device: Option<Device>,
    queue: Option<Queue>,
}

/// A `Hyperslice4DNode` after the scene that writes scene colour and depth under reversed Z, or colour only per cell while a strip is published; clones share one state.
#[derive(Clone)]
pub struct HyperslicePass {
    shared: Rc<RefCell<State>>,
}

impl HyperslicePass {
    pub fn new(source: String) -> Self {
        Self {
            shared: Rc::new(RefCell::new(State {
                source,
                format: TextureFormat::Rgba8UnormSrgb,
                depth: crate::view::DEPTH_FORMAT,
                sample_count: 1,
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

    pub fn strip_cells(&self) -> usize {
        self.shared.borrow().cells.len()
    }
}

impl FramePass for HyperslicePass {
    fn name(&self) -> &'static str {
        "hyperslice"
    }

    fn writes(&self) -> &[ResourceId] {
        &WRITES
    }

    fn order(&self) -> PassOrder {
        PassOrder::AfterScene
    }

    fn depth_convention(&self) -> Option<DepthConvention> {
        Some(DepthConvention::ReversedZ)
    }

    fn record(&self, encoder: &mut CommandEncoder, target: &FrameTarget<'_>) {
        let mut state = self.shared.borrow_mut();
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
            return;
        };
        uniforms.resolution = [target.size.0 as f32, target.size.1 as f32];
        uniforms.viewport_origin = [0.0, 0.0];
        *node.uniforms_mut() = *uniforms;
        if !cells.is_empty() {
            let _ = node.record_strip(device, queue, encoder, target.color, cells);
            return;
        }
        let Some(depth) = target.depth else {
            return;
        };
        node.set_bodies(bodies);
        node.flush_uniforms(queue);
        node.record(
            encoder,
            target.color,
            Some(depth),
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
                label: Some("loam-render::passes::hyperslice"),
                source: wgpu::ShaderSource::Wgsl(state.source.clone().into()),
            });
        state.node = Some(Hyperslice4DNode::with_depth(
            &gpu.device,
            state.format,
            &module,
            DepthMode::ReadWrite {
                format: state.depth,
            },
            state.sample_count,
        ));
        state.device = Some(gpu.device.clone());
        state.queue = Some(gpu.queue.clone());
        Ok(())
    }
}
