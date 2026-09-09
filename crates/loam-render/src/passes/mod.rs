mod field;
mod hyperslice;
mod line;
mod point;
mod raymarch;
mod sky_ground;

pub use field::FieldPass;
pub use hyperslice::HyperslicePass;
pub use line::LinePass;
pub use point::PointPass;
pub use raymarch::RaymarchPass;
pub use sky_ground::SkyGroundPass;

#[cfg(test)]
mod tests {
    use loam_runtime::{Eye, PointRecord, SegmentRecord};
    use wgpu::{
        BackendOptions, Backends, Extent3d, Instance, InstanceDescriptor, NoopBackendOptions,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
        TextureViewDescriptor,
    };

    use super::*;
    use crate::device::{FeatureRequest, GpuContext};
    use crate::pass::{FrameFormat, FramePass, FrameTarget, PassSchedule};
    use crate::raymarch::{
        polytope_stub_sdfs_wgsl, BodyUniform, Hyperslice4DUniforms, HYPERSLICE_KERNEL_WGSL,
    };
    use crate::sky_ground::{Ground, DEFAULT_FOG_PER_UNIT, GROUND_DARK_GREY, GROUND_LIGHT_GREY};
    use crate::view::DEPTH_FORMAT;
    use crate::DepthConvention;

    const FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
    const SIZE: (u32, u32) = (32, 24);
    const SAMPLES: u32 = 4;

    const EMPTY_SCENE: &str = r#"
const LOAM_PRIM_HYPERSPHERE4D: u32 = 0u;
const LOAM_PRIM_HALFSPACE4D: u32 = 1u;
const LOAM_PRIM_OTHER: u32 = 255u;
struct LoamSceneHit { dist: f32, kind: u32 }
fn loam_scene_at(p: vec3<f32>) -> LoamSceneHit {
    return LoamSceneHit(1.0e9, LOAM_PRIM_OTHER);
}
fn loam_scene_sdf(p: vec3<f32>) -> f32 {
    return loam_scene_at(p).dist;
}
fn loam_scene_max_t(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    return 1.0e9;
}
"#;

    fn kernel() -> String {
        format!(
            "{HYPERSLICE_KERNEL_WGSL}\n{stubs}\n{EMPTY_SCENE}",
            stubs = polytope_stub_sdfs_wgsl()
        )
    }

    fn ground() -> Ground {
        Ground {
            y: 0.0,
            dark: GROUND_DARK_GREY,
            light: GROUND_LIGHT_GREY,
            fog_per_unit: DEFAULT_FOG_PER_UNIT,
            visible: true,
        }
    }

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

    fn frame(sample_count: u32) -> FrameFormat {
        FrameFormat {
            color: FORMAT,
            depth: DEPTH_FORMAT,
            sample_count,
        }
    }

    fn attachment(gpu: &GpuContext, format: TextureFormat, sample_count: u32) -> TextureView {
        gpu.device
            .create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: SIZE.0,
                    height: SIZE.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&TextureViewDescriptor::default())
    }

    fn wrappers() -> Vec<Box<dyn FramePass>> {
        let sky = SkyGroundPass::new(ground());
        let hyperslice = HyperslicePass::new(kernel());
        let line = LinePass::new("edges");
        let point = PointPass::new("vertices");
        sky.publish(&Eye::default(), ground());
        hyperslice.publish(
            Hyperslice4DUniforms::default(),
            &[BodyUniform::sphere([0.0; 4], 0.5, [1.0; 3])],
        );
        line.publish(&Eye::default(), &[SegmentRecord::default()]);
        point.publish(&Eye::default(), &[PointRecord::default()]);
        vec![
            Box::new(sky),
            Box::new(hyperslice),
            Box::new(line),
            Box::new(point),
        ]
    }

    fn record_once(gpu: &GpuContext, schedule: &mut PassSchedule, sample_count: u32) {
        let color = attachment(gpu, FORMAT, sample_count);
        let depth = attachment(gpu, DEPTH_FORMAT, sample_count);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let target = FrameTarget {
            color: &color,
            depth: Some(&depth),
            size: SIZE,
        };
        schedule.begin_frame();
        schedule.record(crate::PassOrder::BeforeScene, &mut encoder, &target);
        schedule.record(crate::PassOrder::AfterScene, &mut encoder, &target);
        schedule.end_frame(&mut encoder);
        gpu.queue.submit(Some(encoder.finish()));
    }

    #[test]
    fn the_background_wrapper_precedes_the_scene_and_the_overlays_follow_the_depth_writer() {
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        for pass in wrappers() {
            schedule.register(pass).expect("registered");
        }
        let names: Vec<&'static str> = schedule.names().collect();
        let index = |name: &str| {
            names
                .iter()
                .position(|held| *held == name)
                .unwrap_or_else(|| panic!("{name} never registered: {names:?}"))
        };
        assert!(
            index("sky-ground") < index("hyperslice"),
            "the background clears after the marcher shaded into it: {names:?}"
        );
        assert!(
            index("hyperslice") < index("edges") && index("hyperslice") < index("vertices"),
            "an overlay records before the pass that writes the depth it tests: {names:?}"
        );
    }

    #[test]
    fn a_wrapper_rebuilt_on_a_second_device_records_that_devices_resources_at_the_frames_sample_count(
    ) {
        let first = noop_gpu();
        let second = noop_gpu();
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        for pass in wrappers() {
            schedule.register(pass).expect("registered");
        }

        schedule
            .rebuild(&first, frame(SAMPLES))
            .expect("first attach");
        record_once(&first, &mut schedule, SAMPLES);

        schedule.rebuild(&second, frame(SAMPLES)).expect("recovery");
        second
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        record_once(&second, &mut schedule, SAMPLES);
        let error = pollster::block_on(second.device.pop_error_scope());
        assert!(
            error.is_none(),
            "a wrapper recorded the lost device's resources, or built at a sample count the frame does not use: {error:?}"
        );
    }
}
