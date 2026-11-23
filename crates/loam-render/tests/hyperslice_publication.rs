use glam::{Vec2, Vec4};
use loam_math::EuclideanR4;
use loam_render::present::Presenter;
use loam_render::HyperslicePass;
use loam_runtime::{
    DomainBuilder, Eye, Instance, LogCapacity, Material, Pose, PreparedGeometry, Publication,
    Section4, Session, SimConfig, SpawnBundle, ViewSpec,
};
use loam_shape::polytope::Polytope4;
use loam_shape::FieldShape;
use wgpu::*;

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Probe {
        tags: Store<u8>,
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
fn a_registered_hyperslice_pass_holds_one_body_per_published_field_record() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let tesseract = session.prepare(PreparedGeometry::Polytope4 {
        polytope: Polytope4::Tesseract,
        scale: 0.5,
    });
    let sphere = session.prepare(PreparedGeometry::Field {
        shape: FieldShape::ThreeSphere,
        size: 0.5,
    });
    let material = session.add_material(Material::flat([1.0; 4]));
    let root = session.views().root();
    let view = session.dispatch(|d| {
        let eye = d
            .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
            .unwrap();
        for (x, geometry) in [(-1.0, tesseract), (1.0, sphere)] {
            d.spawn(
                SpawnBundle::new()
                    .at(r4, Pose::at(Vec4::new(x, 0.0, -4.0, 0.0)))
                    .instance(Instance::new(geometry, material)),
            )
            .unwrap();
        }
        d.domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))
            .unwrap()
    });

    let (device, queue) = noop_device();
    let pass = HyperslicePass::new(String::new());
    let mut presenter = Presenter::new(TextureFormat::Rgba8Unorm).expect("presenter");
    presenter
        .register_pass(Box::new(pass.clone()))
        .expect("registered");
    let mut publication = Publication::default();
    let mut present = |session: &mut Session<Probe>| {
        session.publish(&mut publication).unwrap();
        presenter.upload(
            &device,
            &queue,
            &Eye::default(),
            Vec2::splat(64.0),
            &publication.views,
        );
        publication.views[0].records.fields().len()
    };

    let fields = |session: &mut Session<Probe>, on: bool| {
        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .view_mut(view)
            .unwrap()
            .fields = on;
    };

    fields(&mut session, true);
    assert_eq!(present(&mut session), 2);
    assert_eq!(
        pass.body_count(),
        2,
        "the pass did not read the frame's field records"
    );

    fields(&mut session, false);
    assert_eq!(present(&mut session), 0);
    assert_eq!(
        pass.body_count(),
        0,
        "the pass kept drawing bodies the publication dropped"
    );
}
