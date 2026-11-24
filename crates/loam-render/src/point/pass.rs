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
            crate::depth::root_view_projection(eye),
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

#[cfg(test)]
mod tests {
    use loam_runtime::{Eye, PointRecord};
    use wgpu::{
        BackendOptions, Backends, Extent3d, Instance, InstanceDescriptor, NoopBackendOptions,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
        TextureViewDescriptor,
    };

    use super::*;
    use crate::depth::DEPTH_FORMAT;
    use crate::device::{FeatureRequest, GpuContext};
    use crate::pass::{FrameFormat, FramePass, FrameTarget, PassSchedule};
    use crate::DepthConvention;

    const FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
    const SIZE: (u32, u32) = (32, 24);

    fn noop_gpu() -> GpuContext {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::NOOP,
            backend_options: BackendOptions {
                noop: NoopBackendOptions { enable: true },
                ..Default::default()
            },
            ..Default::default()
        });
        pollster::block_on(GpuContext::new(instance, FeatureRequest::default(), None))
            .expect("the noop backend always yields a context")
    }

    fn frame() -> FrameFormat {
        FrameFormat {
            color: FORMAT,
            depth: DEPTH_FORMAT,
        }
    }

    fn attachment(gpu: &GpuContext, format: TextureFormat) -> TextureView {
        gpu.device
            .create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: SIZE.0,
                    height: SIZE.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&TextureViewDescriptor::default())
    }

    fn record_once(gpu: &GpuContext, schedule: &mut PassSchedule) {
        let color = attachment(gpu, FORMAT);
        let depth = attachment(gpu, DEPTH_FORMAT);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let target = FrameTarget {
            color: &color,
            depth: Some(&depth),
            size: SIZE,
        };
        schedule.begin_frame();
        schedule
            .record(crate::PassStage::Background, &mut encoder, &target)
            .expect("background recorded");
        schedule
            .record(crate::PassStage::Scene, &mut encoder, &target)
            .expect("scene recorded");
        schedule
            .record(crate::PassStage::Overlay, &mut encoder, &target)
            .expect("overlays recorded");
        schedule.end_frame(&mut encoder);
        gpu.queue.submit(Some(encoder.finish()));
    }

    #[test]
    fn color_only_publish() {
        let gpu = noop_gpu();
        let mut point = PointPass::new("vertices");
        let record = PointRecord {
            position: [0.0, 0.0, -2.0],
            radius_px: 4.0,
            color: [1.0, 0.0, 0.0, 1.0],
        };
        point.publish(&Eye::default(), &[record]);
        point.attach(&gpu, frame()).expect("attach");

        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        schedule
            .register(Box::new(point.clone()))
            .expect("registered");
        record_once(&gpu, &mut schedule);
        let after_first = point.uploads();

        point.publish(
            &Eye::default(),
            &[PointRecord {
                color: [0.0, 1.0, 0.0, 1.0],
                ..record
            }],
        );
        record_once(&gpu, &mut schedule);
        assert_eq!(
            point.uploads(),
            after_first + 1,
            "the color change never left the CPU: the pass uploaded {} times across both frames",
            point.uploads()
        );
    }
}
