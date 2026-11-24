use std::cell::RefCell;
use std::rc::Rc;

use loam_runtime::FieldRecord;
use wgpu::{CommandEncoder, Device, Queue};

use crate::device::GpuContext;
use crate::pass::{ColorLoad, FrameFormat, FramePass, FrameTarget, PassStage, PublishedFrame};
use crate::raymarch::{
    hyperslice_ground_wgsl, shape_id, BodyUniform, Hyperslice4DNode, Hyperslice4DUniforms,
};
use crate::sky_ground::DEFAULT_SKY;
use crate::{DepthConvention, DepthMode, Viewport};

struct State {
    source: String,
    uniforms: Hyperslice4DUniforms,
    bodies: Vec<BodyUniform>,
    cells: Vec<(Viewport, f32, BodyUniform)>,
    built: Option<Built>,
}

struct Built {
    node: Hyperslice4DNode,
    device: Device,
    queue: Queue,
}

/// Draws the field records of the first published view that has any, cut at that view's field cut; `u.params.x` is 1 while the backdrop's ground is visible.
#[derive(Clone)]
pub struct HyperslicePass {
    shared: Rc<RefCell<State>>,
}

impl HyperslicePass {
    pub fn new(source: String) -> Self {
        Self {
            shared: Rc::new(RefCell::new(State {
                source,
                uniforms: Hyperslice4DUniforms::default(),
                bodies: Vec::new(),
                cells: Vec::new(),
                built: None,
            })),
        }
    }

    /// The kernel over the backdrop's ground plane, which bounds the march and hides what lies below it.
    pub fn with_ground() -> Self {
        Self::new(hyperslice_ground_wgsl())
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

    fn depth_convention(&self) -> Option<DepthConvention> {
        Some(DepthConvention::ReversedZ)
    }

    fn depth_read(&self) -> Option<DepthConvention> {
        Some(DepthConvention::ReversedZ)
    }

    fn read(&mut self, frame: &PublishedFrame<'_>) {
        let mut state = self.shared.borrow_mut();
        let State {
            uniforms, bodies, ..
        } = &mut *state;
        let eye = &frame.eye;
        let drawn = frame.views.iter().find_map(|view| {
            let cut = view.records.field_cut()?;
            (!view.records.fields().is_empty()).then_some((view, cut))
        });
        uniforms.camera_pos = eye.position;
        uniforms.camera_forward = eye.forward;
        uniforms.camera_right = eye.right;
        uniforms.camera_up = eye.up;
        uniforms.fov_y_tan = (eye.fov_y * 0.5).tan();
        uniforms.near = eye.near;
        bodies.clear();
        if let Some((view, cut)) = drawn {
            uniforms.w_slice = cut;
            bodies.extend(view.records.fields().iter().map(body_of));
        }
        let backdrop = frame.backdrop;
        let sky = backdrop.map_or(DEFAULT_SKY, |backdrop| backdrop.sky);
        let floor = backdrop.is_some_and(|backdrop| backdrop.ground.visible);
        uniforms.sky_below = sky.below;
        uniforms.sky_above = sky.above;
        uniforms.params = [if floor { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0];
        if let Some(backdrop) = backdrop {
            uniforms.ground_plane = [0.0, 1.0, 0.0, backdrop.ground.y];
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
            built,
            ..
        } = &mut *state;
        let Some(Built {
            node,
            device,
            queue,
        }) = built.as_mut()
        else {
            return Ok(());
        };
        uniforms.resolution = [target.size.0 as f32, target.size.1 as f32];
        uniforms.viewport_origin = [0.0, 0.0];
        *node.uniforms_mut() = *uniforms;
        if !cells.is_empty() {
            node.record_strip(device, queue, encoder, target.color, cells, ColorLoad::Load)?;
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
            ColorLoad::Load,
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
        state.built = Some(Built {
            node: Hyperslice4DNode::with_depth(
                &gpu.device,
                frame.color,
                &module,
                DepthMode::ReadWrite {
                    format: frame.depth,
                },
            ),
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_source() {
        crate::shader::validate_hyperslice_wgsl(&hyperslice_ground_wgsl())
            .unwrap_or_else(|error| panic!("ground scene under the kernel: {error}"));
    }
}
