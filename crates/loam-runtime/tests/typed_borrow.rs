use loam_math::{EuclideanR4, Iso4Flat, Rotor4};
use loam_runtime::{
    DomainBuilder, DomainError, Domains, Entity, Phase, Pose, Publish, Session, SimConfig,
    SpawnBundle, Step, DOMAIN_STEP,
};

#[derive(Clone, Copy)]
struct Tally {
    ticks: u32,
}

impl Publish for Tally {
    type Record = u32;

    fn record(&self, _entity: Entity) -> u32 {
        self.ticks
    }
}

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Probe {
        tallies: Published<Tally>,
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

    session.system(Phase::Simulation, "count", |app: &mut Probe, step: Step| {
        for (_, tally) in app.tallies.iter_mut() {
            tally.ticks += 1;
        }
        *app.elapsed.get_mut() += step.dt;
    });
    session.system(
        Phase::Simulation,
        "read far",
        move |app: &mut Probe, domains: &mut Domains| {
            assert!(matches!(
                domains.read(foreign),
                Err(DomainError::ForeignRuntime)
            ));
            assert!(domains.read(near).unwrap().poses().is_empty());
            let far = domains.read(far).unwrap();
            for (entity, tally) in app.tallies.iter() {
                let pose = far.poses().get(entity).unwrap();
                app.seen.set(Some(pose.frame.xy * tally.ticks as f32));
            }
        },
    );

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
