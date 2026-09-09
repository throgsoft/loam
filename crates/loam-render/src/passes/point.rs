use std::cell::RefCell;
use std::rc::Rc;

use glam::Vec2;
use loam_math::{EuclideanR3, Projection};
use loam_runtime::{Eye, PointRecord};
use loam_shape::PointMesh;
use wgpu::{CommandEncoder, Device, Queue, TextureFormat};

use crate::device::{GpuContext, MissingGpuCapability};
use crate::pass::{FrameFormat, FramePass, FrameTarget, PassOrder, ResourceId, SCENE_COLOR};
use crate::{DepthConvention, DepthMode, PointRasterNode};

const READS: [ResourceId; 1] = [SCENE_COLOR];
const WRITES: [ResourceId; 1] = [SCENE_COLOR];

struct State {
    format: TextureFormat,
    depth: TextureFormat,
    sample_count: u32,
    eye: Eye,
    mesh: PointMesh<3>,
    uploaded: bool,
    uploads: u64,
    node: Option<PointRasterNode>,
    device: Option<Device>,
    queue: Option<Queue>,
}

/// A `PointRasterNode` after the scene that reads and writes scene colour with no depth test; clones share one state.
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
                depth: crate::view::DEPTH_FORMAT,
                sample_count: 1,
                eye: Eye::default(),
                mesh: PointMesh::default(),
                uploaded: false,
                uploads: 0,
                node: None,
                device: None,
                queue: None,
            })),
        }
    }

    /// Stores the eye and rebuilds the point mesh; the record uploads it again only when a position differs from the last publish or the device was rebuilt.
    pub fn publish(&self, eye: &Eye, points: &[PointRecord]) {
        let mut state = self.shared.borrow_mut();
        state.eye = *eye;
        let mesh = &state.mesh;
        let same = mesh.positions.len() == points.len()
            && mesh.colors.len() == points.len()
            && mesh.sizes.len() == points.len()
            && points.iter().enumerate().all(|(index, point)| {
                mesh.positions[index] == point.position
                    && mesh.colors[index] == point.color
                    && mesh.sizes[index] == point.radius_px
            });
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

    pub fn uploads(&self) -> u64 {
        self.shared.borrow().uploads
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

    fn record(&self, encoder: &mut CommandEncoder, target: &FrameTarget<'_>) {
        let mut state = self.shared.borrow_mut();
        let State {
            eye,
            mesh,
            uploaded,
            uploads,
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
            *uploads += 1;
        }
        node.record(encoder, target.color, None, None);
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
