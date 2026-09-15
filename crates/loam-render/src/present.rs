use std::cell::RefCell;
use std::rc::Rc;

use glam::Vec2;
use loam_runtime::{DomainId, Eye, PublishedView, Rigid, SegmentRecord, Stamp, ViewTarget};
use wgpu::{
    Color, CommandEncoder, Device, LoadOp, Operations, Queue, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, StoreOp, TextureFormat, TextureView,
};

use crate::depth::DepthBuffer;
use crate::device::GpuContext;
use crate::pass::{
    FrameFormat, FramePass, FrameTarget, PassError, PassExecutionError, PassSchedule, PassStage,
    Section,
};
use crate::triangle_pass::TriangleFeed;
use crate::view::{DEPTH_CLEAR, DEPTH_FORMAT};
use crate::{DepthConvention, DepthMode, FragmentShading, LineRasterNode};

struct ViewLines {
    opaque: LineRasterNode,
    translucent: LineRasterNode,
    scratch: Vec<SegmentRecord>,
    uploaded: Option<(DomainId, ViewTarget, Stamp)>,
    camera: Option<(glam::Mat4, Vec2)>,
}

impl ViewLines {
    fn new(device: &Device, format: TextureFormat) -> Self {
        Self {
            opaque: LineRasterNode::new(
                device,
                format,
                DepthMode::ReadWrite {
                    format: DEPTH_FORMAT,
                },
                DepthConvention::ReversedZ,
            ),
            translucent: LineRasterNode::new(
                device,
                format,
                DepthMode::ReadOnly {
                    format: DEPTH_FORMAT,
                },
                DepthConvention::ReversedZ,
            ),
            scratch: Vec::new(),
            uploaded: None,
            camera: None,
        }
    }

    fn set_camera(&mut self, queue: &Queue, camera: glam::Mat4, viewport: Vec2) {
        if self.camera == Some((camera, viewport)) {
            return;
        }
        self.camera = Some((camera, viewport));
        self.opaque.set_camera(queue, camera, viewport);
        self.translucent.set_camera(queue, camera, viewport);
    }

    fn upload_segments(&mut self, device: &Device, queue: &Queue, segments: &[SegmentRecord]) {
        self.scratch.clear();
        self.scratch
            .extend(segments.iter().copied().filter(segment_is_opaque));
        self.opaque.upload_segments(device, queue, &self.scratch);
        self.scratch.clear();
        self.scratch.extend(
            segments
                .iter()
                .copied()
                .filter(|segment| !segment_is_opaque(segment)),
        );
        self.translucent
            .upload_segments(device, queue, &self.scratch);
    }
}

struct PublishedLines {
    views: Rc<RefCell<Vec<ViewLines>>>,
}

impl FramePass for PublishedLines {
    fn name(&self) -> &'static str {
        "present-draw"
    }

    fn stage(&self) -> PassStage {
        PassStage::Scene
    }

    fn depth_convention(&self) -> Option<DepthConvention> {
        Some(DepthConvention::ReversedZ)
    }

    fn depth_read(&self) -> Option<DepthConvention> {
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
        let views = self.views.borrow();
        for slot in views.iter() {
            slot.opaque.record(encoder, target.color, Some(depth), None);
        }
        for slot in views.iter() {
            slot.translucent
                .record(encoder, target.color, Some(depth), None);
        }
        Ok(())
    }

    fn attach(&mut self, _gpu: &GpuContext, _frame: FrameFormat) -> anyhow::Result<()> {
        Ok(())
    }
}

pub struct Presenter {
    uploads: u64,
    format: TextureFormat,
    depth: Option<DepthBuffer>,
    views: Rc<RefCell<Vec<ViewLines>>>,
    fills: TriangleFeed,
    filled: Vec<(DomainId, ViewTarget)>,
    schedule: PassSchedule,
}

impl Presenter {
    pub fn new(format: TextureFormat) -> Result<Self, PassError> {
        let fills = TriangleFeed::default();
        let views = Rc::new(RefCell::new(Vec::new()));
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        schedule.register(fills.pass(FragmentShading::FaceNormalLambert))?;
        schedule.register(Box::new(PublishedLines {
            views: views.clone(),
        }))?;
        Ok(Self {
            format,
            uploads: 0,
            depth: None,
            views,
            fills,
            filled: Vec::new(),
            schedule,
        })
    }

    pub fn attach(&mut self, gpu: &GpuContext) -> Result<(), PassExecutionError> {
        self.depth = None;
        self.views.borrow_mut().clear();
        self.schedule.attach(
            gpu,
            FrameFormat {
                color: self.format,
                depth: DEPTH_FORMAT,
            },
        )
    }

    pub fn register_pass(&mut self, pass: Box<dyn FramePass>) -> Result<(), PassError> {
        self.schedule.register(pass)
    }

    pub fn uploads(&self) -> u64 {
        self.uploads
    }

    pub fn sections(&self) -> &[Section] {
        self.schedule.sections()
    }

    pub fn after_submit(&mut self) {
        self.schedule.after_submit();
    }

    pub fn upload(
        &mut self,
        device: &Device,
        queue: &Queue,
        eye: &Eye,
        viewport: Vec2,
        views: &[PublishedView],
    ) {
        let _scope = loam_time::frame_trace::scope("present-upload");
        let mut line_views = self.views.borrow_mut();
        while line_views.len() < views.len() {
            line_views.push(ViewLines::new(device, self.format));
        }
        line_views.truncate(views.len());
        let mut rebuilt = false;
        for (slot, view) in line_views.iter_mut().zip(views) {
            let camera = crate::view::placed_view_projection(eye, view.placement);
            slot.set_camera(queue, camera, viewport);
            let published = (view.domain, view.target, view.records.built());
            if slot.uploaded == Some(published) {
                continue;
            }
            slot.uploaded = Some(published);
            rebuilt = true;
            self.uploads += 1;
            slot.upload_segments(device, queue, view.records.segments());
        }
        drop(line_views);
        self.fills.set_view(eye, Rigid::IDENTITY);
        let listed = self.filled.len() == views.len()
            && (self.filled.iter().zip(views))
                .all(|(held, view)| *held == (view.domain, view.target));
        if rebuilt || !listed {
            self.filled.clear();
            self.filled
                .extend(views.iter().map(|view| (view.domain, view.target)));
            self.fills.edit(|mesh| {
                mesh.vertices.clear();
                mesh.colors.clear();
                mesh.indices.clear();
                for view in views {
                    for triangle in view.records.triangles() {
                        let base = mesh.vertices.len() as u32;
                        for corner in triangle.vertices {
                            mesh.vertices.push(view.placement.apply(corner));
                            mesh.colors.push(triangle.color);
                        }
                        mesh.indices.push([base, base + 1, base + 2]);
                    }
                }
            });
        }
    }

    pub fn record_scene(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        target: &TextureView,
        size: (u32, u32),
        background: Color,
    ) -> Result<(), PassExecutionError> {
        self.schedule.begin_frame();
        DepthBuffer::ensure(&mut self.depth, device, DEPTH_FORMAT, size);
        let Some(depth) = self.depth.as_ref() else {
            return Ok(());
        };
        let frame = FrameTarget {
            color: target,
            depth: Some(&depth.view),
            size,
        };
        self.schedule.section("present-clear", encoder, |encoder| {
            encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("loam-render present clear"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(background),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth.view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(DEPTH_CLEAR),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        });
        self.schedule
            .record(PassStage::Background, encoder, &frame)?;
        self.schedule.record(PassStage::Scene, encoder, &frame)
    }

    /// Call after `record_scene` with the same target and encoder.
    pub fn record_overlays(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &TextureView,
        size: (u32, u32),
    ) -> Result<(), PassExecutionError> {
        let Some(depth) = self.depth.as_ref() else {
            return Ok(());
        };
        let frame = FrameTarget {
            color: target,
            depth: Some(&depth.view),
            size,
        };
        self.schedule.record(PassStage::Overlay, encoder, &frame)?;
        self.schedule.end_frame(encoder);
        Ok(())
    }
}

fn segment_is_opaque(segment: &SegmentRecord) -> bool {
    segment.start_color[3] >= 1.0 && segment.end_color[3] >= 1.0
}

#[cfg(test)]
mod tests {
    use crate::pass::ColorLoad;
    use loam_math::EuclideanR4;
    use loam_runtime::{
        DomainBuilder, Instance, LogCapacity, Material, Pose, PreparedGeometry, Projection4,
        Records, Session, SimConfig, SpawnBundle, ViewSpec,
    };
    use wgpu::{
        BackendOptions, Backends, InstanceDescriptor, NoopBackendOptions, TextureDescriptor,
        TextureDimension, TextureFormat, TextureUsages,
    };

    use super::*;

    #[global_allocator]
    static COUNTING_ALLOCATOR: loam_time::alloc::CountingAllocator<std::alloc::System> =
        loam_time::alloc::CountingAllocator::new(std::alloc::System);

    loam_runtime::stores! {
        #[derive(Default)]
        pub struct Spun {
            spin: Store<f32>,
        }
    }

    fn noop_device() -> (Device, Queue) {
        let instance = wgpu::Instance::new(&InstanceDescriptor {
            backends: Backends::NOOP,
            backend_options: BackendOptions {
                noop: NoopBackendOptions { enable: true },
                ..Default::default()
            },
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("noop backend always yields an adapter");
        pollster::block_on(adapter.request_device(&Default::default()))
            .expect("noop adapter always yields a device")
    }

    #[test]
    fn warmed_publish_allocates_or_the_presenter_reuploads_unchanged_records() {
        let mut session = Session::new(Spun::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let edges = session.prepare(PreparedGeometry::Lines4 {
            segments: (0..64)
                .map(|i| {
                    let t = i as f32 * 0.01;
                    [[t, 0.0, 0.0, 0.0], [t, 1.0, 0.0, 0.5]]
                })
                .collect(),
        });
        let white = session.add_material(Material::lines([1.0; 4], 2.0));
        let root = session.views().root();
        let spun = session.dispatch(|d| {
            let eye = d
                .spawn(SpawnBundle::new().at(r4, Pose::at(glam::Vec4::ZERO)))
                .unwrap();
            d.domains
                .typed(r4)
                .unwrap()
                .add_view(ViewSpec::new(root, eye, Projection4 { focal: 2.0 }))
                .unwrap();
            d.spawn(
                SpawnBundle::new()
                    .at(r4, Pose::at(glam::Vec4::NEG_Z * 4.0))
                    .instance(Instance::new(edges, white))
                    .row(0.0_f32),
            )
            .unwrap()
        });

        let (device, queue) = noop_device();
        let target = device
            .create_texture(&TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let mut presenter = Presenter::new(TextureFormat::Rgba8Unorm).expect("presenter");
        let mut records = Records::default();
        let eye = Eye::default();
        let spin = |session: &mut Session<Spun>| {
            if let Ok(domain) = session.domains_mut().typed(r4) {
                if let Some(pose) = domain.poses().get(spun) {
                    let mut point = pose.point;
                    point.x += 0.01;
                    domain.set_point(spun, point).unwrap();
                }
            }
        };
        let present = |session: &mut Session<Spun>,
                       records: &mut Records,
                       presenter: &mut Presenter,
                       encoder: &mut CommandEncoder| {
            records.publish(session).unwrap();
            let published = records.lend().unwrap();
            presenter.upload(&device, &queue, &eye, Vec2::splat(64.0), &published.views);
            records.release(published);
            presenter
                .record_scene(&device, encoder, &target, (64, 64), Color::BLACK)
                .expect("recorded the scene");
            presenter
                .record_overlays(encoder, &target, (64, 64))
                .expect("recorded the overlays");
        };

        let mut encoder = device.create_command_encoder(&Default::default());
        for _ in 0..8 {
            spin(&mut session);
            present(&mut session, &mut records, &mut presenter, &mut encoder);
        }

        let published = loam_time::alloc::bytes_allocated_by(|| {
            for _ in 0..16 {
                spin(&mut session);
                records.publish(&mut session).unwrap();
            }
        })
        .expect("the counting allocator is installed");
        assert_eq!(
            published, 0,
            "16 warmed publications asked the allocator for {published} bytes"
        );

        present(&mut session, &mut records, &mut presenter, &mut encoder);
        let before = presenter.uploads();
        let idle = loam_time::alloc::bytes_allocated_by(|| {
            for _ in 0..16 {
                present(&mut session, &mut records, &mut presenter, &mut encoder);
            }
            assert_eq!(
                presenter.uploads(),
                before,
                "an idle presenter uploaded a view"
            );
        })
        .expect("the counting allocator is installed");
        let busy = loam_time::alloc::bytes_allocated_by(|| {
            for _ in 0..16 {
                spin(&mut session);
                present(&mut session, &mut records, &mut presenter, &mut encoder);
            }
            assert_eq!(
                presenter.uploads(),
                before + 16,
                "sixteen changed publications uploaded the view {} times",
                presenter.uploads() - before
            );
        })
        .expect("the counting allocator is installed");
        assert!(
            idle < busy,
            "the presenter uploads records it already holds: {idle} bytes idle against {busy} busy"
        );
    }

    #[test]
    fn a_view_dropped_from_the_publication_takes_its_section_fills_with_it() {
        use loam_runtime::{Section4, ViewId};

        let mut session = Session::new(Spun::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let geometry = session.prepare(PreparedGeometry::Polytope4 {
            polytope: loam_shape::polytope::Polytope4::Tesseract,
            scale: 0.7,
        });
        let body = session.add_material(Material::lines([1.0; 4], 1.0));
        let cut = session.add_material(Material::lines([1.0, 0.85, 0.35, 1.0], 2.0));
        let root = session.views().root();
        session.dispatch(|d| {
            let eye = d
                .spawn(SpawnBundle::new().at(r4, Pose::at(glam::Vec4::ZERO)))
                .expect("the eye spawned");
            d.spawn(
                SpawnBundle::new()
                    .at(r4, Pose::at(glam::Vec4::NEG_Z * 4.0))
                    .instance(Instance::new(geometry, body).sectioned(cut))
                    .row(0.0_f32),
            )
            .expect("the body spawned");
            let domain = d.domains.typed(r4).expect("the r4 domain");
            let _: ViewId = domain
                .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))
                .expect("the first view");
            domain
                .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))
                .expect("the second view")
        });

        let (device, queue) = noop_device();
        let mut presenter = Presenter::new(TextureFormat::Rgba8Unorm).expect("presenter");
        let mut records = Records::default();
        let eye = Eye::default();
        records.publish(&mut session).expect("published");
        let published = records.lend().expect("the buffer is free");

        presenter.upload(&device, &queue, &eye, Vec2::splat(64.0), &published.views);
        let both = presenter.fills.triangles();
        assert_eq!(
            both,
            24 * 2,
            "each of the two cut layers fans the tesseract's six straddling cells into four triangles, not {both}"
        );

        presenter.upload(
            &device,
            &queue,
            &eye,
            Vec2::splat(64.0),
            &published.views[..1],
        );
        assert_eq!(
            presenter.fills.triangles(),
            24,
            "the dropped view left its section fills on screen"
        );
        records.release(published);
    }

    #[test]
    fn dropping_the_first_of_two_equally_stamped_views_skips_the_survivors_lines() {
        let mut session = Session::new(Spun::default(), SimConfig::default());
        let white = session.add_material(Material::lines([1.0; 4], 1.0));
        let root = session.views().root();
        for (name, polytope) in [
            ("wide", loam_shape::polytope::Polytope4::Tesseract),
            ("narrow", loam_shape::polytope::Polytope4::Pentatope),
        ] {
            let domain = session.register_domain(
                DomainBuilder::new(name, EuclideanR4).tracked(LogCapacity::default()),
            );
            let geometry = session.prepare(PreparedGeometry::Polytope4 {
                polytope,
                scale: 0.7,
            });
            session.dispatch(|d| {
                let eye = d
                    .spawn(SpawnBundle::new().at(domain, Pose::at(glam::Vec4::ZERO)))
                    .expect("the eye spawned");
                d.spawn(
                    SpawnBundle::new()
                        .at(domain, Pose::at(glam::Vec4::NEG_Z * 4.0))
                        .instance(Instance::new(geometry, white))
                        .row(0.0_f32),
                )
                .expect("the body spawned");
                d.domains
                    .typed(domain)
                    .expect("the domain")
                    .add_view(ViewSpec::new(root, eye, Projection4 { focal: 2.0 }))
                    .expect("the view")
            });
        }

        let (device, queue) = noop_device();
        let mut presenter = Presenter::new(TextureFormat::Rgba8Unorm).expect("presenter");
        let mut records = Records::default();
        let eye = Eye::default();
        records.publish(&mut session).expect("published");
        let published = records.lend().expect("the buffer is free");
        let dropped = &published.views[0].records;
        let kept = &published.views[1].records;
        assert!(
            dropped.built() == kept.built() && dropped.segments().len() != kept.segments().len(),
            "the probe needs two equally stamped views whose lines differ"
        );
        let survivor = kept.segments().len() as u32;

        presenter.upload(&device, &queue, &eye, Vec2::splat(64.0), &published.views);
        presenter.upload(
            &device,
            &queue,
            &eye,
            Vec2::splat(64.0),
            &published.views[1..],
        );
        let line_views = presenter.views.borrow();
        assert_eq!(
            line_views[0].opaque.segment_count() + line_views[0].translucent.segment_count(),
            survivor,
            "the first slot still holds the dropped view's lines"
        );
        records.release(published);
    }

    const PROBE_SIZE: u32 = 64;
    const PAINT: Color = Color {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };

    struct Paint;

    impl FramePass for Paint {
        fn name(&self) -> &'static str {
            "paint"
        }

        fn stage(&self) -> PassStage {
            PassStage::Background
        }

        fn color_load(&self) -> ColorLoad {
            ColorLoad::Clear
        }

        fn record(
            &mut self,
            encoder: &mut CommandEncoder,
            target: &FrameTarget<'_>,
        ) -> anyhow::Result<()> {
            encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("paint"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target.color,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(PAINT),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            Ok(())
        }

        fn attach(&mut self, _gpu: &GpuContext, _frame: FrameFormat) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn pixel_at(gpu: &GpuContext, texture: &wgpu::Texture, at: [u32; 2]) -> [u8; 4] {
        let row = PROBE_SIZE * 4;
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row) * u64::from(PROBE_SIZE),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: PROBE_SIZE,
                height: PROBE_SIZE,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("the readback polled");
        receiver
            .recv()
            .expect("the map callback ran")
            .expect("the staging buffer mapped");
        let data = slice.get_mapped_range();
        let offset = ((at[1] * PROBE_SIZE + at[0]) * 4) as usize;
        let pixel = [
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ];
        drop(data);
        staging.unmap();
        pixel
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run with --include-ignored"]
    fn a_pass_before_the_scene_reaches_the_presented_frame_gpu_probe() {
        let gpu = pollster::block_on(GpuContext::new(
            wgpu::Instance::default(),
            crate::device::FeatureRequest::default(),
            None,
        ))
        .expect("a wgpu adapter");
        let texture = gpu.device.create_texture(&TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: PROBE_SIZE,
                height: PROBE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let mut presenter = Presenter::new(TextureFormat::Rgba8Unorm).expect("presenter");
        presenter
            .register_pass(Box::new(Paint))
            .expect("registered");
        presenter.attach(&gpu).expect("attached");
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        presenter
            .record_scene(
                &gpu.device,
                &mut encoder,
                &view,
                (PROBE_SIZE, PROBE_SIZE),
                Color::BLACK,
            )
            .expect("recorded the scene");
        presenter
            .record_overlays(&mut encoder, &view, (PROBE_SIZE, PROBE_SIZE))
            .expect("recorded the overlays");
        gpu.queue.submit(Some(encoder.finish()));

        assert_eq!(
            pixel_at(&gpu, &texture, [0, 0]),
            [255, 0, 0, 255],
            "the presenter cleared the frame after the pass that runs before the scene"
        );
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run with --include-ignored"]
    fn published_translucency_follows_all_opaque_geometry_gpu_probe() {
        let gpu = pollster::block_on(GpuContext::new(
            wgpu::Instance::default(),
            crate::device::FeatureRequest::default(),
            None,
        ))
        .expect("a wgpu adapter");
        let texture = gpu.device.create_texture(&TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: PROBE_SIZE,
                height: PROBE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target = texture.create_view(&Default::default());
        let mut presenter = Presenter::new(TextureFormat::Rgba8Unorm).expect("presenter");
        presenter.attach(&gpu).expect("attached");
        let eye = Eye::default();
        let camera = crate::view::placed_view_projection(&eye, Rigid::IDENTITY);
        presenter.fills.set_view(&eye, Rigid::IDENTITY);
        presenter.fills.edit(|mesh| {
            mesh.vertices
                .extend([[-2.0, -1.5, -4.0], [2.0, -1.5, -4.0], [0.0, 2.0, -4.0]]);
            mesh.colors.extend([[0.0, 1.0, 0.0, 1.0]; 3]);
            mesh.indices.push([0, 1, 2]);
        });

        let mut translucent = ViewLines::new(&gpu.device, TextureFormat::Rgba8Unorm);
        translucent.set_camera(&gpu.queue, camera, Vec2::splat(PROBE_SIZE as f32));
        translucent.upload_segments(
            &gpu.device,
            &gpu.queue,
            &[SegmentRecord {
                start: [-1.0, 0.0, -3.0],
                end: [1.0, 0.0, -3.0],
                start_color: [1.0, 0.0, 0.0, 0.01],
                end_color: [1.0, 0.0, 0.0, 0.01],
                width_px: 12.0,
                ..Default::default()
            }],
        );
        let mut opaque = ViewLines::new(&gpu.device, TextureFormat::Rgba8Unorm);
        opaque.set_camera(&gpu.queue, camera, Vec2::splat(PROBE_SIZE as f32));
        opaque.upload_segments(
            &gpu.device,
            &gpu.queue,
            &[SegmentRecord {
                start: [0.0, -0.75, -3.5],
                end: [0.0, 0.75, -3.5],
                start_color: [0.0, 0.0, 1.0, 1.0],
                end_color: [0.0, 0.0, 1.0, 1.0],
                width_px: 8.0,
                ..Default::default()
            }],
        );
        presenter.views.borrow_mut().extend([translucent, opaque]);

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        presenter
            .record_scene(
                &gpu.device,
                &mut encoder,
                &target,
                (PROBE_SIZE, PROBE_SIZE),
                Color::BLACK,
            )
            .expect("recorded the scene");
        presenter
            .record_overlays(&mut encoder, &target, (PROBE_SIZE, PROBE_SIZE))
            .expect("recorded the overlays");
        gpu.queue.submit(Some(encoder.finish()));

        let surface = pixel_at(&gpu, &texture, [42, 32]);
        assert!(
            surface[0] > 0 && surface[1] > 80,
            "the translucent line exposed the background through the fill: {surface:?}"
        );
        let crossing = pixel_at(&gpu, &texture, [32, 32]);
        assert!(
            crossing[0] > 0 && crossing[2] > 200,
            "a later view drew its opaque line after the translucent batch: {crossing:?}"
        );
    }
}
