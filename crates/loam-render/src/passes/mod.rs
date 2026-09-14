mod hyperslice;
mod line;
mod point;
mod raymarch;
mod sky_ground;

pub use hyperslice::HyperslicePass;
pub use line::LinePass;
pub use point::PointPass;
pub use raymarch::RaymarchPass;
pub use sky_ground::SkyGroundPass;

#[cfg(test)]
mod tests {
    use loam_runtime::{Eye, PointRecord};
    use wgpu::{
        BackendOptions, Backends, Extent3d, Instance, InstanceDescriptor, NoopBackendOptions,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
        TextureViewDescriptor,
    };

    use super::*;
    use crate::device::{FeatureRequest, GpuContext};
    use crate::pass::{FrameFormat, FramePass, FrameTarget, PassSchedule};
    use crate::view::DEPTH_FORMAT;
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
    fn a_publish_that_changes_only_a_color_still_reaches_the_gpu() {
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
