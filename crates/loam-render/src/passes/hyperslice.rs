use std::cell::RefCell;
use std::rc::Rc;

use loam_runtime::{Eye, FieldRecord, PublishedView, Rigid};
use wgpu::{CommandEncoder, Device, Queue};

use crate::device::GpuContext;
use crate::pass::{ColorLoad, FrameFormat, FramePass, FrameTarget, PassStage};
use crate::raymarch::{shape_id, BodyUniform, Hyperslice4DNode, Hyperslice4DUniforms};
use crate::sky_ground::Sky;
use crate::{DepthConvention, DepthMode, Viewport};

struct State {
    source: String,
    uniforms: Hyperslice4DUniforms,
    bodies: Vec<BodyUniform>,
    cells: Vec<(Viewport, f32, BodyUniform)>,
    node: Option<Hyperslice4DNode>,
    device: Option<Device>,
    queue: Option<Queue>,
}

/// Draws the field records of the first published view that has any, cut at that view's field cut.
#[derive(Clone)]
pub struct HyperslicePass {
    shared: Rc<RefCell<State>>,
    load: ColorLoad,
}

impl HyperslicePass {
    pub fn new(source: String) -> Self {
        Self {
            load: ColorLoad::Load,
            shared: Rc::new(RefCell::new(State {
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

    /// `params` reaches the shader as `u.params`.
    pub fn set_environment(&self, sky: Sky, params: [f32; 4]) {
        let uniforms = &mut self.shared.borrow_mut().uniforms;
        uniforms.sky_below = sky.below;
        uniforms.sky_above = sky.above;
        uniforms.params = params;
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

fn body_of(field: &FieldRecord) -> BodyUniform {
    let [r, g, b, _] = field.color;
    BodyUniform::polytope_with_rotor(
        field.position,
        shape_id(field.shape),
        field.size,
        field.frame,
        [r, g, b],
    )
}

impl FramePass for HyperslicePass {
    fn name(&self) -> &'static str {
        "hyperslice"
    }

    fn stage(&self) -> PassStage {
        PassStage::Background
    }

    fn color_load(&self) -> ColorLoad {
        self.load
    }

    fn depth_convention(&self) -> Option<DepthConvention> {
        Some(DepthConvention::ReversedZ)
    }

    fn depth_read(&self) -> Option<DepthConvention> {
        Some(DepthConvention::ReversedZ)
    }

    fn read_publication(&mut self, eye: &Eye, views: &[PublishedView]) {
        let mut state = self.shared.borrow_mut();
        let State {
            uniforms, bodies, ..
        } = &mut *state;
        let drawn = views.iter().find_map(|view| {
            let cut = view.records.field_cut()?;
            (!view.records.fields().is_empty()).then_some((view, cut))
        });
        let to_image = drawn.map_or(Rigid::IDENTITY, |(view, _)| view.placement.inverse());
        uniforms.camera_pos = to_image.apply(eye.position);
        uniforms.camera_forward = to_image.direction(eye.forward);
        uniforms.camera_right = to_image.direction(eye.right);
        uniforms.camera_up = to_image.direction(eye.up);
        uniforms.fov_y_tan = (eye.fov_y * 0.5).tan();
        uniforms.near = eye.near * to_image.scale;
        bodies.clear();
        if let Some((view, cut)) = drawn {
            uniforms.w_slice = cut;
            bodies.extend(view.records.fields().iter().map(body_of));
        }
    }

    fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
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
            return Ok(());
        };
        uniforms.resolution = [target.size.0 as f32, target.size.1 as f32];
        uniforms.viewport_origin = [0.0, 0.0];
        *node.uniforms_mut() = *uniforms;
        if !cells.is_empty() {
            node.record_strip(device, queue, encoder, target.color, cells, self.load)?;
            return Ok(());
        }
        // Without bodies the kernel would still draw the shader's own scene, such as a floor.
        let (Some(depth), false) = (target.depth, bodies.is_empty()) else {
            return Ok(());
        };
        node.set_bodies(bodies);
        node.flush_uniforms(queue);
        node.record(
            encoder,
            target.color,
            Some(depth),
            Viewport::full([target.size.0, target.size.1]),
            self.load,
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
