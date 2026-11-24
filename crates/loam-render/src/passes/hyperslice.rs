use std::cell::RefCell;
use std::rc::Rc;

use glam::Vec3;

use loam_runtime::{FieldRecord, Rigid};
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
    node: Option<Hyperslice4DNode>,
    device: Option<Device>,
    queue: Option<Queue>,
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
                node: None,
                device: None,
                queue: None,
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
        let backdrop = frame.backdrop;
        let sky = backdrop.map_or(DEFAULT_SKY, |backdrop| backdrop.sky);
        let floor = backdrop.is_some_and(|backdrop| backdrop.ground.visible);
        uniforms.sky_below = sky.below;
        uniforms.sky_above = sky.above;
        uniforms.params = [if floor { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0];
        if let Some(backdrop) = backdrop {
            let normal = Vec3::from(to_image.direction([0.0, 1.0, 0.0]));
            let on_plane = Vec3::from(to_image.apply([0.0, backdrop.ground.y, 0.0]));
            uniforms.ground_plane = normal.extend(normal.dot(on_plane)).to_array();
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

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec4};
    use loam_math::{EuclideanR4, Iso3};
    use loam_runtime::{
        Backdrop, BridgeSpec, DomainBuilder, Eye, Ground, Instance, LogCapacity, Material, Overlay,
        Placement, Pose, PreparedGeometry, Publication, Section4, Session, SimConfig, SpawnBundle,
        ViewSpec,
    };
    use loam_shape::FieldShape;

    use super::*;

    loam_runtime::stores! {
        #[derive(Default)]
        pub struct Probe {
            tags: Store<u8>,
        }
    }

    const GROUND_Y: f32 = -0.75;

    fn placed_field() -> Session<Probe> {
        let mut session = Session::new(Probe::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let sphere = session.prepare(PreparedGeometry::Field {
            shape: FieldShape::ThreeSphere,
            size: 0.5,
        });
        let material = session.add_material(Material::flat([1.0; 4]));
        let root = session.views().root_id();
        let (view, anchor) = session.dispatch(|d| {
            let eye = d
                .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
                .unwrap();
            let anchor = d
                .spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::new(0.0, 0.0, -4.0, 0.0)))
                        .instance(Instance::new(sphere, material)),
                )
                .unwrap();
            let view = d
                .domains
                .typed(r4)
                .unwrap()
                .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))
                .unwrap();
            (view, anchor)
        });
        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .view_mut(view)
            .unwrap()
            .fields = true;
        session
            .bridge(BridgeSpec {
                anchor,
                into: root,
                source: r4.id(),
                view,
                placement: Placement::Rigid(Rigid {
                    pose: Iso3 {
                        rotation: Quat::from_euler(glam::EulerRot::XYZ, 0.4, -0.7, 1.1),
                        translation: Vec3::new(0.3, -1.2, 2.0),
                    },
                    scale: 0.25,
                }),
            })
            .unwrap();
        session
    }

    #[test]
    fn the_ground_plane_is_the_backdrops_floor_carried_into_the_placed_views_image() {
        let mut session = placed_field();
        let mut publication = Publication::default();
        session.publish(&mut publication, 1.0).unwrap();
        let placement = publication.views[0].placement;
        assert_ne!(placement, Rigid::IDENTITY, "the view was not placed");
        assert!(!publication.views[0].records.fields().is_empty());

        let mut pass = HyperslicePass::with_ground();
        pass.read(&PublishedFrame {
            eye: Eye::default(),
            viewport_px: glam::Vec2::splat(64.0),
            pixels_per_point: 1.0,
            text_origin_px: [0.0; 2],
            views: &publication.views,
            overlay: &Overlay::default(),
            backdrop: Some(Backdrop {
                sky: DEFAULT_SKY,
                ground: Ground {
                    y: GROUND_Y,
                    dark: [0.0; 3],
                    light: [1.0; 3],
                    fog_per_unit: 0.0,
                    fog_start: 0.0,
                    visible: true,
                },
            }),
        });

        let uniforms = pass.shared.borrow().uniforms;
        assert_eq!(uniforms.params[0], 1.0, "a visible floor left the gate off");
        let [nx, ny, nz, offset] = uniforms.ground_plane;
        let normal = Vec3::new(nx, ny, nz);
        assert!((normal.length() - 1.0).abs() < 1e-4, "normal {normal}");
        let to_image = placement.inverse();
        let image = |world: [f32; 3]| Vec3::from(to_image.apply(world));
        for (x, z) in [(0.0, 0.0), (3.0, -2.0), (-5.0, 7.0)] {
            let on_floor = image([x, GROUND_Y, z]);
            assert!(
                (normal.dot(on_floor) - offset).abs() < 1e-3,
                "the floor point ({x}, {GROUND_Y}, {z}) is {} off the plane",
                normal.dot(on_floor) - offset
            );
        }
        let above = image([0.0, GROUND_Y + 1.0, 0.0]);
        assert!(
            normal.dot(above) > offset,
            "the plane's normal points below the floor"
        );
    }

    #[test]
    fn the_ground_source_passes_the_validation_attach_runs() {
        crate::shader::validate_hyperslice_wgsl(&hyperslice_ground_wgsl())
            .unwrap_or_else(|error| panic!("ground scene under the kernel: {error}"));
    }
}
