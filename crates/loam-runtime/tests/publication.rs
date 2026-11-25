use loam_math::{EuclideanR4, Space};
use loam_runtime::{
    DomainBuilder, DomainError, Instance, LogCapacity, Material, Phase, PhaseError, Pose,
    PreparedGeometry, Publication, Section4, Session, SimConfig, SpawnBundle, ViewSpec,
};
use loam_shape::polytope::Polytope4;

type Vec4 = <EuclideanR4 as Space>::Point;

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Scored {
        frames: Value<u32>,
    }
    publishes Vec<u32>;
}

#[test]
fn latest_value_system() {
    let mut session = Session::new(Scored::default(), SimConfig::default());
    session.system(Phase::Publication, "count", |ctx| {
        *ctx.app.frames.get_mut() += 1;
        Ok(())
    });
    session.publish_value("replaced", |_, value: &mut Vec<u32>| {
        value.push(u32::MAX);
        Ok(())
    });
    session.publish_value("frames", |ctx, value: &mut Vec<u32>| {
        let frames = *ctx.app.frames.get();
        if frames == 3 {
            return Err(DomainError::InvalidFrame);
        }
        value.clear();
        value.push(frames);
        Ok(())
    });
    let mut publication = Publication::default();

    session.publish(&mut publication, 1.0).unwrap();
    assert_eq!(
        publication.value,
        [1],
        "the value missed this publish's publication systems or kept a replaced system's write"
    );
    session.publish(&mut publication, 1.0).unwrap();
    assert_eq!(publication.value, [2]);

    let failure = PhaseError {
        phase: Phase::Publication,
        system: Some("frames"),
        cause: DomainError::InvalidFrame,
    };
    assert_eq!(session.publish(&mut publication, 1.0), Err(failure));
    assert_eq!(
        session.phase_error(),
        Some(failure),
        "a failed value system left the session unfaulted"
    );
}

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Bare {}
}

#[test]
fn still_after_record_count_change() {
    let mut session = Session::new(Bare::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let tesseract = session.prepare(PreparedGeometry::Polytope4 {
        polytope: Polytope4::Tesseract,
        scale: 0.5,
    });
    let edge = session.add_material(Material::lines([1.0; 4], 1.0));
    let root = session.views().root_id();
    let body = session.dispatch(|d| {
        let eye = d
            .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
            .unwrap();
        let body = d
            .spawn(
                SpawnBundle::new()
                    .at(r4, Pose::at(Vec4::new(0.0, 0.0, -4.0, 0.0)))
                    .instance(Instance::new(tesseract, edge).sectioned(edge)),
            )
            .unwrap();
        d.domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))
            .unwrap();
        body
    });
    let mut publication = Publication::default();
    let mut shown = |session: &mut Session<Bare>| {
        session.publish(&mut publication, 1.0).unwrap();
        let records = &publication.views[0].records;
        (
            publication.content(),
            records.built(),
            records.segments().len(),
        )
    };
    let move_to = |session: &mut Session<Bare>, at: Vec4| {
        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .set_pose(body, Pose::at(at))
            .unwrap();
    };

    let first = shown(&mut session);
    move_to(&mut session, Vec4::new(0.5, 0.0, -4.0, 0.0));
    let slid = shown(&mut session);
    assert_ne!(slid.0, first.0, "a slide inside the slice drew nothing new");
    assert_eq!(slid.2, first.2);
    assert_eq!(
        shown(&mut session),
        slid,
        "a still scene changed after a move that kept its record count"
    );

    move_to(&mut session, Vec4::new(0.5, 0.0, -4.0, 2.0));
    let lifted = shown(&mut session);
    assert_ne!(lifted.0, slid.0, "a lift out of the slice drew nothing new");
    assert!(
        lifted.2 < slid.2,
        "a body lifted out of the slice kept its {} cut segments",
        slid.2
    );
    assert_eq!(
        shown(&mut session),
        lifted,
        "a still scene changed after a move that changed its record count"
    );

    move_to(&mut session, Vec4::new(0.0, 0.0, -4.0, 2.0));
    let slid_again = shown(&mut session);
    assert_ne!(
        slid_again.0, lifted.0,
        "a slide after the lift drew nothing new"
    );
    assert_eq!(slid_again.2, lifted.2);
}

#[test]
fn section_without_whole_edges() {
    let mut session = Session::new(Bare::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let tesseract = session.prepare(PreparedGeometry::Polytope4 {
        polytope: Polytope4::Tesseract,
        scale: 0.5,
    });
    let edge = session.add_material(Material::lines([1.0; 4], 1.0));
    session
        .try_dispatch(|d| {
            d.spawn(
                SpawnBundle::new()
                    .at(r4, Pose::at(Vec4::new(0.0, 0.0, -4.0, 0.0)))
                    .instance(Instance::new(tesseract, edge).sectioned(edge)),
            )?;
            d.add_section(r4, Pose::at(Vec4::ZERO), Section4 { w: 0.0 })
        })
        .unwrap();

    let mut publication = Publication::default();
    session.publish(&mut publication, 1.0).unwrap();
    let records = &publication.views[0].records;
    assert_eq!(records.triangles().len(), 24);
    assert_eq!(
        records.segments().len(),
        24,
        "a section view drew the 32 whole edges beside the cut's 24 perimeter segments"
    );
}
