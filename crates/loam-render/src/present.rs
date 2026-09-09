use glam::Vec2;
use loam_runtime::{Eye, PublishedView, Stamp};
use wgpu::{
    Color, CommandEncoder, Device, LoadOp, Operations, Queue, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, StoreOp, TextureFormat, TextureView,
};

use crate::depth::DepthBuffer;
use crate::view::{DEPTH_CLEAR, DEPTH_FORMAT};
use crate::{DepthConvention, DepthMode, LineRasterNode};

struct ViewLines {
    node: LineRasterNode,
    uploaded: Stamp,
}

pub struct Presenter {
    format: TextureFormat,
    sample_count: u32,
    depth: Option<DepthBuffer>,
    views: Vec<ViewLines>,
}

impl Presenter {
    pub fn new(format: TextureFormat, sample_count: u32) -> Self {
        Self {
            format,
            sample_count,
            depth: None,
            views: Vec::new(),
        }
    }

    /// Skips a view whose records were built by a publication it already uploaded.
    pub fn upload(
        &mut self,
        device: &Device,
        queue: &Queue,
        eye: &Eye,
        viewport: Vec2,
        views: &[PublishedView],
    ) {
        while self.views.len() < views.len() {
            self.views.push(ViewLines {
                node: LineRasterNode::new(
                    device,
                    self.format,
                    DepthMode::ReadWrite {
                        format: DEPTH_FORMAT,
                    },
                    DepthConvention::ReversedZ,
                    self.sample_count,
                ),
                uploaded: Stamp::default(),
            });
        }
        self.views.truncate(views.len());
        for (slot, view) in self.views.iter_mut().zip(views) {
            slot.node.set_root_camera(queue, eye, viewport);
            let built = view.records.built();
            if slot.uploaded == built {
                continue;
            }
            slot.uploaded = built;
            slot.node
                .upload_segments(device, queue, view.records.segments());
        }
    }

    pub fn record(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        target: &TextureView,
        size: (u32, u32),
        background: Color,
    ) {
        DepthBuffer::ensure(
            &mut self.depth,
            device,
            DEPTH_FORMAT,
            size,
            self.sample_count,
        );
        let Some(depth) = self.depth.as_ref() else {
            return;
        };
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
        for slot in &self.views {
            slot.node.record(encoder, target, Some(&depth.view), None);
        }
    }
}

#[cfg(test)]
mod tests {
    use loam_math::{EuclideanR4, Iso4Flat};
    use loam_runtime::{
        DomainBuilder, Instance, LogCapacity, Material, Pose, PreparedGeometry, Projection4,
        Records, Session, SimConfig, SpawnBundle, ViewSpec,
    };
    use wgpu::{
        BackendOptions, Backends, InstanceDescriptor, NoopBackendOptions, TextureDescriptor,
        TextureDimension, TextureFormat, TextureUsages,
    };

    use super::*;

    mod alloc_probe {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        thread_local! {
            static BYTES: Cell<usize> = const { Cell::new(0) };
        }

        pub struct Counting;

        // SAFETY: Methods preserve System contracts; const TLS and wrapping Cell updates cannot unwind.
        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                let _ = BYTES.try_with(|bytes| bytes.set(bytes.get().wrapping_add(layout.size())));
                // SAFETY: The caller supplies a valid nonzero allocation layout.
                unsafe { System.alloc(layout) }
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                // SAFETY: The caller supplies a live System allocation and its original layout.
                unsafe { System.dealloc(ptr, layout) }
            }

            unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                let _ = BYTES.try_with(|bytes| bytes.set(bytes.get().wrapping_add(new_size)));
                // SAFETY: The caller supplies a live System allocation, its layout, and a valid new size.
                unsafe { System.realloc(ptr, layout, new_size) }
            }
        }

        pub fn bytes_allocated_by(body: impl FnOnce()) -> usize {
            let before = BYTES.with(Cell::get);
            body();
            BYTES.with(Cell::get).wrapping_sub(before)
        }
    }

    #[global_allocator]
    static COUNTING_ALLOCATOR: alloc_probe::Counting = alloc_probe::Counting;

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
                .spawn(SpawnBundle::new().at(r4, Pose(Iso4Flat::IDENTITY)))
                .unwrap();
            d.domains.typed(r4).unwrap().add_view(ViewSpec::new(
                root,
                eye,
                Projection4 { focal: 2.0 },
            ));
            d.spawn(
                SpawnBundle::new()
                    .at(
                        r4,
                        Pose(Iso4Flat::from_translation(glam::Vec4::NEG_Z * 4.0)),
                    )
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
        let mut presenter = Presenter::new(TextureFormat::Rgba8Unorm, 1);
        let mut records = Records::<Spun>::default();
        let eye = Eye::default();
        let spin = |session: &mut Session<Spun>| {
            if let Ok(domain) = session.domains_mut().typed(r4) {
                if let Some(pose) = domain.poses.get_mut(spun) {
                    pose.0.translation.x += 0.01;
                }
            }
        };
        let present = |session: &mut Session<Spun>,
                       records: &mut Records<Spun>,
                       presenter: &mut Presenter,
                       encoder: &mut CommandEncoder| {
            records.publish(session).unwrap();
            let published = records.lend().unwrap();
            presenter.upload(&device, &queue, &eye, Vec2::splat(64.0), &published.views);
            records.release(published);
            presenter.record(&device, encoder, &target, (64, 64), Color::BLACK);
        };

        let mut encoder = device.create_command_encoder(&Default::default());
        for _ in 0..8 {
            spin(&mut session);
            present(&mut session, &mut records, &mut presenter, &mut encoder);
        }

        let published = alloc_probe::bytes_allocated_by(|| {
            for _ in 0..16 {
                spin(&mut session);
                records.publish(&mut session).unwrap();
            }
        });
        assert_eq!(
            published, 0,
            "16 warmed publications asked the allocator for {published} bytes"
        );

        let idle = alloc_probe::bytes_allocated_by(|| {
            for _ in 0..16 {
                present(&mut session, &mut records, &mut presenter, &mut encoder);
            }
        });
        let busy = alloc_probe::bytes_allocated_by(|| {
            for _ in 0..16 {
                spin(&mut session);
                present(&mut session, &mut records, &mut presenter, &mut encoder);
            }
        });
        assert!(
            idle < busy,
            "the presenter uploads records it already holds: {idle} bytes idle against {busy} busy"
        );
    }
}
