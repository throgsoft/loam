use loam_math::{EuclideanR4, Iso4Flat, Rotor4};
use loam_runtime::{
    Ctx, DomainBuilder, DomainError, Phase, Pose, Session, SimConfig, SpawnBundle, DOMAIN_STEP,
};

#[derive(Clone, Copy)]
struct Tally {
    ticks: u32,
}

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Probe {
        tallies: Store<Tally>,
        elapsed: Value<f32>,
        seen: Value<Option<f32>>,
    }
}

#[test]
fn typed_borrow_never_reaches_another_domain_or_session() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let near = session.register_domain(DomainBuilder::new("near", EuclideanR4));
    let far = session.register_domain(DomainBuilder::new("far", EuclideanR4));
    let mut other = Session::new(Probe::default(), SimConfig::default());
    let foreign = other.register_domain(DomainBuilder::new("foreign", EuclideanR4));

    let quarter = Rotor4 {
        s: 0.0,
        xy: 1.0,
        ..Rotor4::IDENTITY
    };
    session.dispatch(|d| {
        d.spawn(
            SpawnBundle::new()
                .at(far, Pose::from(Iso4Flat::from_rotation(quarter)))
                .row(Tally { ticks: 0 }),
        )
        .unwrap();
    });

    session.system(Phase::Simulation, "count", |ctx: Ctx<'_, Probe>| {
        for (_, tally) in ctx.app.tallies.iter_mut() {
            tally.ticks += 1;
        }
        *ctx.app.elapsed.get_mut() += ctx.step.dt;
        Ok(())
    });
    session.system(Phase::Simulation, "read far", move |ctx: Ctx<'_, Probe>| {
        assert!(matches!(
            ctx.domains.read(foreign),
            Err(DomainError::ForeignRuntime)
        ));
        assert!(ctx.domains.read(near).unwrap().poses().is_empty());
        let far = ctx.domains.read(far).unwrap();
        for (entity, tally) in ctx.app.tallies.iter() {
            let pose = far.poses().get(entity).unwrap();
            ctx.app.seen.set(Some(pose.frame.xy * tally.ticks as f32));
        }
        Ok(())
    });

    session.tick().unwrap();

    assert_eq!(*session.app.elapsed.get(), 1.0 / 60.0);
    assert_eq!(*session.app.seen.get(), Some(1.0));
    match session.entries(Phase::Simulation) {
        [step, count, _] => {
            assert_eq!(step.name(), DOMAIN_STEP);
            assert_eq!(count.name(), "count");
        }
        entries => panic!("{} entries registered", entries.len()),
    }
}
