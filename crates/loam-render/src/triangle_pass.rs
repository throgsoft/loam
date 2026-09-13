use std::cell::RefCell;
use std::rc::Rc;

use glam::Mat4;
use loam_math::{EuclideanR3, Projection};
use loam_runtime::{Eye, Rigid, Version};
use loam_shape::TriangleMesh;
use wgpu::{CommandEncoder, Device, Queue};

use crate::device::GpuContext;
use crate::pass::{
    FrameFormat, FramePass, FrameTarget, PassStage, ResourceId, SCENE_BASE, SCENE_COLOR,
    SCENE_DEPTH,
};
use crate::view::placed_view_projection;
use crate::{
    DepthConvention, DepthMode, FragmentShading, Ground, SkyGroundNode, SkyGroundUniforms,
    TriangleRasterNode, Viewport,
};

const WRITES: [ResourceId; 2] = [SCENE_COLOR, SCENE_DEPTH];
const READS: [ResourceId; 1] = [SCENE_BASE];

#[derive(Default)]
struct Input {
    mesh: TriangleMesh<3>,
    source_version: Option<Version>,
    revision: u64,
    view_projection: Mat4,
    ground: Option<Ground>,
    uploads: u64,
}

/// The mesh, view, and optional ground drawn after the scene by the pass `pass` builds; the mesh uploads once per `edit` and again after a device loss.
#[derive(Clone, Default)]
pub struct TriangleFeed {
    input: Rc<RefCell<Input>>,
}

impl TriangleFeed {
    pub fn pass(&self, shading: FragmentShading) -> Box<dyn FramePass> {
        Box::new(TrianglePass {
            input: self.input.clone(),
            shading,
            built: None,
        })
    }

    pub fn set_view(&self, eye: &Eye, placement: Rigid) {
        self.input.borrow_mut().view_projection = placed_view_projection(eye, placement);
    }

    pub fn set_ground(&self, ground: Option<Ground>) {
        self.input.borrow_mut().ground = ground;
    }

    /// Bumps the revision, so the next record uploads the mesh.
    pub fn edit(&self, build: impl FnOnce(&mut TriangleMesh<3>)) {
        let mut input = self.input.borrow_mut();
        build(&mut input.mesh);
        input.source_version = None;
        input.revision += 1;
    }

    /// Tracks one source by version; switching sources requires a distinct version.
    pub fn set_mesh(&self, version: Version, mesh: &TriangleMesh<3>) {
        let mut input = self.input.borrow_mut();
        if input.source_version == Some(version) {
            return;
        }
        input.source_version = Some(version);
        input.mesh.vertices.clone_from(&mesh.vertices);
        input.mesh.indices.clone_from(&mesh.indices);
        input.mesh.colors.clone_from(&mesh.colors);
        input.revision += 1;
    }

    pub fn uploads(&self) -> u64 {
        self.input.borrow().uploads
    }

    pub fn triangles(&self) -> usize {
        self.input.borrow().mesh.indices.len()
    }
}

struct Built {
    device: Device,
    queue: Queue,
    triangles: TriangleRasterNode,
    sky_ground: SkyGroundNode,
    uploaded: Option<u64>,
}

struct TrianglePass {
    input: Rc<RefCell<Input>>,
    shading: FragmentShading,
    built: Option<Built>,
}

impl FramePass for TrianglePass {
    fn name(&self) -> &'static str {
        "triangles"
    }

    fn writes(&self) -> &[ResourceId] {
        &WRITES
    }

    fn reads(&self) -> &[ResourceId] {
        &READS
    }

    fn stage(&self) -> PassStage {
        PassStage::Scene
    }

    fn depth_convention(&self) -> Option<DepthConvention> {
        Some(DepthConvention::ReversedZ)
    }

    fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
        let Some(depth) = target.depth else {
            return Ok(());
        };
        let Some(built) = self.built.as_mut() else {
            return Ok(());
        };
        let mut input = self.input.borrow_mut();
        if built.uploaded != Some(input.revision) {
            built.uploaded = Some(input.revision);
            input.uploads += 1;
            built.triangles.upload::<EuclideanR3, 3>(
                &built.device,
                &built.queue,
                &input.mesh,
                &Projection::Identity,
            );
        }
        if let Some(ground) = input.ground {
            built.sky_ground.set_uniforms(
                &built.queue,
                &SkyGroundUniforms::new(
                    input.view_projection,
                    Viewport::full([target.size.0, target.size.1]),
                    ground,
                ),
            );
            built.sky_ground.record(encoder, target.color, depth, None);
        }
        built
            .triangles
            .set_camera(&built.queue, input.view_projection);
        built
            .triangles
            .record(encoder, target.color, Some(depth), None);
        Ok(())
    }

    fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> anyhow::Result<()> {
        let triangles = TriangleRasterNode::new(
            gpu,
            frame.color,
            DepthMode::ReadWrite {
                format: frame.depth,
            },
            DepthConvention::ReversedZ,
            self.shading,
        )?;
        let sky_ground = SkyGroundNode::new(
            &gpu.device,
            frame.color,
            frame.depth,
            DepthConvention::ReversedZ,
        );
        self.built = Some(Built {
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
            triangles,
            sky_ground,
            uploaded: None,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use wgpu::{
        BackendOptions, Backends, BufferDescriptor, BufferUsages, Color, CommandEncoderDescriptor,
        Extent3d, Instance, InstanceDescriptor, LoadOp, NoopBackendOptions, Operations, PollType,
        RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor, StoreOp,
        TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
        TextureViewDescriptor,
    };

    use super::*;
    use crate::device::FeatureRequest;
    use crate::view::{projective_depth, DEPTH_CLEAR, DEPTH_FORMAT};

    const SIZE: (u32, u32) = (64, 64);
    const COLOR: TextureFormat = TextureFormat::Rgba8Unorm;

    fn frame_format() -> FrameFormat {
        FrameFormat {
            color: COLOR,
            depth: DEPTH_FORMAT,
        }
    }

    fn noop_context() -> GpuContext {
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

    fn attachment(gpu: &GpuContext, format: TextureFormat, usage: TextureUsages) -> wgpu::Texture {
        gpu.device.create_texture(&TextureDescriptor {
            label: Some("triangle pass probe"),
            size: Extent3d {
                width: SIZE.0,
                height: SIZE.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | usage,
            view_formats: &[],
        })
    }

    fn one_frame(
        gpu: &GpuContext,
        pass: &mut dyn FramePass,
        color: &TextureView,
        depth: &TextureView,
    ) {
        let mut encoder = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("clear"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: color,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(DEPTH_CLEAR),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.record(
            &mut encoder,
            &FrameTarget {
                color,
                depth: Some(depth),
                size: SIZE,
            },
        )
        .expect("record");
        gpu.queue.submit(Some(encoder.finish()));
    }

    fn triangle(feed: &TriangleFeed, z: f32, color: [f32; 4]) {
        feed.edit(|mesh| {
            mesh.vertices = vec![[-0.4, -0.4, z], [0.4, -0.4, z], [0.0, 0.4, z]];
            mesh.colors = vec![color; 3];
            mesh.indices = vec![[0, 1, 2]];
        });
    }

    #[test]
    fn an_unchanged_mesh_is_not_uploaded_again_on_the_next_frame() {
        let gpu = noop_context();
        let feed = TriangleFeed::default();
        let mut source = loam_runtime::Value::new(TriangleMesh {
            vertices: vec![[-0.4, -0.4, -2.0], [0.4, -0.4, -2.0], [0.0, 0.4, -2.0]],
            colors: vec![[1.0; 4]; 3],
            indices: vec![[0, 1, 2]],
        });
        feed.set_mesh(source.version(), source.get());
        let mut pass = feed.pass(FragmentShading::FaceNormalLambert);
        pass.attach(&gpu, frame_format()).expect("attach");
        let color =
            attachment(&gpu, COLOR, TextureUsages::empty()).create_view(&Default::default());
        let depth =
            attachment(&gpu, DEPTH_FORMAT, TextureUsages::empty()).create_view(&Default::default());

        one_frame(&gpu, pass.as_mut(), &color, &depth);
        feed.set_mesh(source.version(), source.get());
        one_frame(&gpu, pass.as_mut(), &color, &depth);
        source.get_mut().vertices[2][2] = -3.0;
        feed.set_mesh(source.version(), source.get());
        one_frame(&gpu, pass.as_mut(), &color, &depth);

        assert_eq!(
            feed.uploads(),
            2,
            "three frames with one source change uploaded {} times",
            feed.uploads()
        );
    }

    #[test]
    fn a_new_device_uploads_the_mesh_again_without_an_edit() {
        let gpu = noop_context();
        let feed = TriangleFeed::default();
        triangle(&feed, -2.0, [1.0; 4]);
        let mut pass = feed.pass(FragmentShading::FaceNormalLambert);
        pass.attach(&gpu, frame_format()).expect("attach");
        let color =
            attachment(&gpu, COLOR, TextureUsages::empty()).create_view(&Default::default());
        let depth =
            attachment(&gpu, DEPTH_FORMAT, TextureUsages::empty()).create_view(&Default::default());
        one_frame(&gpu, pass.as_mut(), &color, &depth);

        let lost = noop_context();
        pass.attach(&lost, frame_format()).expect("reattach");
        let color =
            attachment(&lost, COLOR, TextureUsages::empty()).create_view(&Default::default());
        let depth = attachment(&lost, DEPTH_FORMAT, TextureUsages::empty())
            .create_view(&Default::default());
        one_frame(&lost, pass.as_mut(), &color, &depth);

        assert_eq!(
            feed.uploads(),
            2,
            "the recovered pass drew from the lost device's buffers"
        );
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run with --include-ignored"]
    fn a_lit_triangle_writes_its_image_pixel_and_projective_depth_gpu_probe() {
        let gpu = pollster::block_on(GpuContext::new(
            Instance::default(),
            FeatureRequest::default(),
            None,
        ))
        .expect("gpu context");
        let feed = TriangleFeed::default();
        const Z: f32 = -2.0;
        triangle(&feed, Z, [1.0, 1.0, 1.0, 1.0]);
        let eye = Eye {
            aspect: SIZE.0 as f32 / SIZE.1 as f32,
            ..Eye::default()
        };
        feed.set_view(&eye, Rigid::IDENTITY);

        let mut pass = feed.pass(FragmentShading::FaceNormalLambert);
        pass.attach(&gpu, frame_format()).expect("attach");
        let color_texture = attachment(&gpu, COLOR, TextureUsages::COPY_SRC);
        let depth_texture = attachment(&gpu, DEPTH_FORMAT, TextureUsages::COPY_SRC);
        one_frame(
            &gpu,
            pass.as_mut(),
            &color_texture.create_view(&TextureViewDescriptor::default()),
            &depth_texture.create_view(&TextureViewDescriptor::default()),
        );

        let centre = (SIZE.0 / 2, SIZE.1 / 2);
        let outside = (1, 1);
        let colors = read_back::<4>(&gpu, &color_texture, COLOR);
        let depths = read_back::<4>(&gpu, &depth_texture, DEPTH_FORMAT);
        let at = |(x, y): (u32, u32), rows: &Vec<[u8; 4]>| rows[(y * SIZE.0 + x) as usize];
        let depth_at = |p: (u32, u32)| f32::from_le_bytes(at(p, &depths));

        let key = glam::Vec3::new(0.55, 0.85, 0.45).normalize();
        let lambert = 0.3 + 0.7 * key.z.abs();
        let lit = (lambert * 255.0).round() as i32;
        let drawn = at(centre, &colors);
        assert!(
            (i32::from(drawn[0]) - lit).abs() <= 2,
            "the lit triangle wrote {drawn:?} where the shader's lambert term gives {lit}"
        );
        assert_eq!(
            at(outside, &colors),
            [0, 0, 0, 255],
            "the triangle covered a pixel outside its image position"
        );
        assert!(
            (depth_at(centre) - projective_depth(glam::Vec3::new(0.0, 0.0, Z), eye.near)).abs()
                < 1e-6,
            "the triangle wrote depth {} for the image point at z = {Z}",
            depth_at(centre)
        );
        assert_eq!(depth_at(outside), DEPTH_CLEAR);
    }

    fn read_back<const N: usize>(
        gpu: &GpuContext,
        texture: &wgpu::Texture,
        format: TextureFormat,
    ) -> Vec<[u8; N]> {
        let row = SIZE.0 as usize * N;
        let padded = row.next_multiple_of(256);
        let staging = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("triangle pass readback"),
            size: (padded * SIZE.1 as usize) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: match format {
                    TextureFormat::Depth32Float => TextureAspect::DepthOnly,
                    _ => TextureAspect::All,
                },
            },
            TexelCopyBufferInfo {
                buffer: &staging,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: Some(SIZE.1),
                },
            },
            Extent3d {
                width: SIZE.0,
                height: SIZE.1,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(Some(encoder.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        gpu.device
            .poll(PollType::wait_indefinitely())
            .expect("poll");
        let view = staging.slice(..).get_mapped_range();
        let mut rows = Vec::with_capacity((SIZE.0 * SIZE.1) as usize);
        for y in 0..SIZE.1 as usize {
            for x in 0..SIZE.0 as usize {
                let at = y * padded + x * N;
                let mut texel = [0u8; N];
                texel.copy_from_slice(&view[at..at + N]);
                rows.push(texel);
            }
        }
        rows
    }
}
