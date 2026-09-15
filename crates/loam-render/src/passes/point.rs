use std::cell::RefCell;
use std::rc::Rc;

use glam::Vec2;
use loam_math::{EuclideanR3, Projection};
use loam_runtime::{Eye, PointRecord};
use loam_shape::PointMesh;
use wgpu::{CommandEncoder, Device, Queue};

use crate::device::GpuContext;
use crate::pass::{FrameFormat, FramePass, FrameTarget, PassStage};
use crate::{DepthConvention, DepthMode, PointRasterNode};

struct State {
    eye: Eye,
    mesh: PointMesh<3>,
    uploaded: bool,
    uploads: u64,
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

    fn stage(&self) -> PassStage {
        PassStage::Scene
    }

    fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
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
            return Ok(());
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
        Ok(())
    }

    fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> anyhow::Result<()> {
        let mut state = self.shared.borrow_mut();
        state.node = Some(PointRasterNode::new(
            &gpu.device,
            frame.color,
            DepthMode::Off,
            DepthConvention::ReversedZ,
        ));
        state.device = Some(gpu.device.clone());
        state.queue = Some(gpu.queue.clone());
        state.uploaded = false;
        Ok(())
    }
}
