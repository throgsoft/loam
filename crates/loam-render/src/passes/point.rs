use std::cell::RefCell;
use std::rc::Rc;

use glam::Vec2;
use loam_math::{EuclideanR3, Projection};
use loam_runtime::{Eye, PointRecord};
use loam_shape::PointMesh;
use wgpu::{CommandEncoder, Device, Queue, TextureFormat};

use crate::device::{GpuContext, MissingGpuCapability};
use crate::pass::{FramePass, FrameTarget, PassOrder, ResourceId, SCENE_COLOR};
use crate::{DepthConvention, DepthMode, PointRasterNode};

const READS: [ResourceId; 1] = [SCENE_COLOR];
const WRITES: [ResourceId; 1] = [SCENE_COLOR];

struct State {
    format: TextureFormat,
    sample_count: u32,
    eye: Eye,
    mesh: PointMesh<3>,
    uploaded: bool,
    node: Option<PointRasterNode>,
    device: Option<Device>,
    queue: Option<Queue>,
}

#[derive(Clone)]
pub struct PointPass {
    name: &'static str,
    shared: Rc<RefCell<State>>,
}

impl PointPass {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            shared: Rc::new(RefCell::new(State {
                format: TextureFormat::Rgba8UnormSrgb,
                sample_count: 1,
                eye: Eye::default(),
                mesh: PointMesh::default(),
                uploaded: false,
                node: None,
                device: None,
                queue: None,
            })),
        }
    }

    pub fn publish(&self, eye: &Eye, points: &[PointRecord]) {
        let mut state = self.shared.borrow_mut();
        state.eye = *eye;
        let same = state.mesh.positions.len() == points.len()
            && state
                .mesh
                .positions
                .iter()
                .zip(points)
                .all(|(held, point)| *held == point.position);
        state.uploaded &= same;
        let mesh = &mut state.mesh;
        mesh.positions.clear();
        mesh.colors.clear();
        mesh.sizes.clear();
        for point in points {
            mesh.positions.push(point.position);
            mesh.colors.push(point.color);
            mesh.sizes.push(point.radius_px);
        }
    }

    pub fn point_count(&self) -> usize {
        self.shared.borrow().mesh.positions.len()
    }
}

impl FramePass for PointPass {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reads(&self) -> &[ResourceId] {
        &READS
    }

    fn writes(&self) -> &[ResourceId] {
        &WRITES
    }

    fn order(&self) -> PassOrder {
        PassOrder::AfterScene
    }

    fn target(&mut self, format: TextureFormat, sample_count: u32) {
        let mut state = self.shared.borrow_mut();
        state.format = format;
        state.sample_count = sample_count;
    }

    fn record(&self, encoder: &mut CommandEncoder, target: &FrameTarget<'_>) {
        let mut state = self.shared.borrow_mut();
        let State {
            eye,
            mesh,
            uploaded,
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
        node.set_camera(
            queue,
            crate::view::root_view_projection(eye),
            Vec2::new(target.size.0 as f32, target.size.1 as f32),
        );
        if !*uploaded {
            node.upload::<EuclideanR3, 3>(device, queue, mesh, &Projection::Identity);
            *uploaded = true;
        }
        node.record(encoder, target.color, None, None);
    }

    fn rebuild(&mut self, gpu: &GpuContext) -> Result<(), MissingGpuCapability> {
        let mut state = self.shared.borrow_mut();
        state.node = Some(PointRasterNode::new(
            &gpu.device,
            state.format,
            DepthMode::Off,
            DepthConvention::ReversedZ,
            state.sample_count,
        ));
        state.device = Some(gpu.device.clone());
        state.queue = Some(gpu.queue.clone());
        state.uploaded = false;
        Ok(())
    }
}
