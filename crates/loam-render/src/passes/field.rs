use std::cell::RefCell;
use std::rc::Rc;

use loam_runtime::FieldProgram;
use wgpu::{CommandEncoder, Queue, TextureFormat};

use crate::device::{GpuContext, MissingGpuCapability};
use crate::pass::{FramePass, FrameTarget, PassOrder};
use crate::raymarch::{FieldMarchNode, FieldMarchUniforms};
use crate::{DepthMode, Viewport};

struct State {
    format: TextureFormat,
    sample_count: u32,
    uniforms: FieldMarchUniforms,
    program: FieldProgram,
    node: Option<FieldMarchNode>,
    queue: Option<Queue>,
}

/// A [`FieldMarchNode`] recorded before the scene draw from the uniforms and the [`FieldProgram`] an application publishes.
#[derive(Clone)]
pub struct FieldPass {
    shared: Rc<RefCell<State>>,
}

impl Default for FieldPass {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldPass {
    pub fn new() -> Self {
        Self {
            shared: Rc::new(RefCell::new(State {
                format: TextureFormat::Rgba8UnormSrgb,
                sample_count: 1,
                uniforms: FieldMarchUniforms::default(),
                program: FieldProgram::default(),
                node: None,
                queue: None,
            })),
        }
    }

    /// `resolution`, `viewport_origin`, and the program lengths come from the frame target and the program, not from here.
    pub fn publish(&self, uniforms: FieldMarchUniforms, program: &FieldProgram) {
        let mut state = self.shared.borrow_mut();
        state.uniforms = uniforms;
        let held = &mut state.program;
        held.primitives.clear();
        held.primitives.extend_from_slice(&program.primitives);
        held.program.clear();
        held.program.extend_from_slice(&program.program);
        held.nodes.clear();
        held.nodes.extend_from_slice(&program.nodes);
        held.stack = program.stack;
        held.kind = program.kind;
    }

    /// Call once per boundary so the node counts stable structures toward its specialization.
    pub fn boundary(&self) {
        if let Some(node) = self.shared.borrow_mut().node.as_mut() {
            node.boundary();
        }
    }

    pub fn is_specialized(&self) -> bool {
        self.shared
            .borrow()
            .node
            .as_ref()
            .is_some_and(FieldMarchNode::is_specialized)
    }
}

impl FramePass for FieldPass {
    fn name(&self) -> &'static str {
        "field-march"
    }

    fn order(&self) -> PassOrder {
        PassOrder::BeforeScene
    }

    fn target(&mut self, format: TextureFormat, sample_count: u32) {
        let mut state = self.shared.borrow_mut();
        state.format = format;
        state.sample_count = sample_count;
    }

    fn record(&self, encoder: &mut CommandEncoder, target: &FrameTarget<'_>) {
        let mut state = self.shared.borrow_mut();
        let State {
            uniforms,
            program,
            node,
            queue,
            ..
        } = &mut *state;
        let (Some(node), Some(queue)) = (node.as_mut(), queue.as_ref()) else {
            return;
        };
        uniforms.resolution = [target.size.0 as f32, target.size.1 as f32];
        uniforms.viewport_origin = [0.0, 0.0];
        *node.uniforms_mut() = *uniforms;
        node.set_program(queue, program);
        node.record(
            encoder,
            target.color,
            None,
            Viewport::full([target.size.0, target.size.1]),
        );
    }

    fn rebuild(&mut self, gpu: &GpuContext) -> Result<(), MissingGpuCapability> {
        let mut state = self.shared.borrow_mut();
        state.node = Some(FieldMarchNode::new(
            &gpu.device,
            state.format,
            DepthMode::Off,
            state.sample_count,
        ));
        state.queue = Some(gpu.queue.clone());
        Ok(())
    }
}
