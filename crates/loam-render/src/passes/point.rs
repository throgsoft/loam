use std::cell::RefCell;
use std::rc::Rc;

use glam::Vec2;
use loam_math::{EuclideanR3, Projection};
use loam_runtime::{Eye, PointRecord};
use loam_shape::PointMesh;
use wgpu::{CommandEncoder, Device, Queue};

use crate::device::GpuContext;
use crate::pass::{FrameFormat, FramePass, FrameTarget, PassStage, PublishedFrame};
use crate::{DepthConvention, DepthMode, PointRasterNode};

struct State {
    eye: Eye,
    mesh: PointMesh<3>,
    uploaded: bool,
    uploads: u64,
    built: Option<Built>,
}

struct Built {
    node: PointRasterNode,
    device: Device,
    queue: Queue,
}

#[derive(Clone)]
pub struct PointPass {
    name: &'static str,
    reads_overlay: bool,
    shared: Rc<RefCell<State>>,
}

impl PointPass {
    pub fn new(name: &'static str) -> Self {
        Self::reading(name, false)
    }

    pub(crate) fn overlay(name: &'static str) -> Self {
        Self::reading(name, true)
    }

    fn reading(name: &'static str, reads_overlay: bool) -> Self {
        Self {
            name,
            reads_overlay,
            shared: Rc::new(RefCell::new(State {
                eye: Eye::default(),
                mesh: PointMesh::default(),
                uploaded: false,
                uploads: 0,
                built: None,
            })),
        }
    }

    pub fn publish(&self, eye: &Eye, points: &[PointRecord]) {
        self.shared.borrow_mut().set(eye, points);
    }

    pub fn uploads(&self) -> u64 {
        self.shared.borrow().uploads
    }
}

impl State {
    fn set(&mut self, eye: &Eye, points: &[PointRecord]) {
        self.eye = *eye;
        let mesh = &self.mesh;
        let same = mesh.positions.len() == points.len()
            && mesh.colors.len() == points.len()
            && mesh.sizes.len() == points.len()
            && points.iter().enumerate().all(|(index, point)| {
                mesh.positions[index] == point.position
                    && mesh.colors[index] == point.color
                    && mesh.sizes[index] == point.radius_px
            });
        self.uploaded &= same;
        let mesh = &mut self.mesh;
        mesh.positions.clear();
        mesh.colors.clear();
        mesh.sizes.clear();
        for point in points {
            mesh.positions.push(point.position);
            mesh.colors.push(point.color);
            mesh.sizes.push(point.radius_px);
        }
    }
}

impl FramePass for PointPass {
    fn name(&self) -> &'static str {
        self.name
    }

    fn stage(&self) -> PassStage {
        PassStage::Scene
    }

    fn read(&mut self, frame: &PublishedFrame<'_>) {
        if self.reads_overlay {
            self.shared
                .borrow_mut()
                .set(&frame.eye, &frame.overlay.points);
        }
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
            built,
        } = &mut *state;
        let Some(Built {
            node,
            device,
            queue,
        }) = built.as_mut()
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
        state.built = Some(Built {
            node: PointRasterNode::new(
                &gpu.device,
                frame.color,
                DepthMode::Off,
                DepthConvention::ReversedZ,
            ),
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
        });
        state.uploaded = false;
        Ok(())
    }
}
