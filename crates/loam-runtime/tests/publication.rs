use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use loam_math::{EuclideanR3, EuclideanR4, Space};
use loam_runtime::{
    DomainBuilder, DomainError, DomainHandle, Entity, Identity3, Instance, LogCapacity, Material,
    Phase, PhaseError, Pose, PreparedGeometry, Projection4, Publication, Section4, Session,
    SimConfig, SpawnBundle, ViewSpec,
};
use loam_shape::polytope::Polytope4;
use loam_time::par::{ChunkSource, Executor, Sequential};

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Bare {}
}

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Scored {
        frames: Value<u32>,
    }
    publishes Vec<u32>;
}

type Point4 = <EuclideanR4 as Space>::Point;
type Point3 = <EuclideanR3 as Space>::Point;

const WORKERS: usize = 4;
// Far above any count the session could keep on the calling thread, so the views reach the workers.
const POPULATION: usize = 256;

struct Threads {
    shared: AtomicUsize,
}

impl Executor for Threads {
    fn parallelism(&self) -> usize {
        WORKERS
    }

    fn for_each_chunk(&self, workers: usize, chunks: &(dyn ChunkSource + Sync)) {
        self.shared.fetch_add(1, Ordering::Relaxed);
        thread::scope(|scope| {
            for _ in 1..workers.clamp(1, WORKERS) {
                scope.spawn(|| while chunks.run_next() {});
            }
            while chunks.run_next() {}
        });
    }
}

struct Stage {
    session: Session<Bare>,
    r4: DomainHandle<EuclideanR4>,
    eye: Entity,
    bodies: Vec<Entity>,
}

fn at(index: usize, shift: f32) -> Point4 {
    let x = (index % 16) as f32 * 0.1 - 0.8 + shift;
    let y = (index / 16) as f32 * 0.1 - 0.8;
    Point4::new(x, y, -3.0, 0.0)
}

fn stage() -> Stage {
    let mut session = Session::new(Bare::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let r3 = session
        .register_domain(DomainBuilder::new("r3", EuclideanR3).tracked(LogCapacity::default()));
    let tesseract = session.prepare(PreparedGeometry::Polytope4 {
        polytope: Polytope4::Tesseract,
        scale: 0.04,
    });
    let line = session.prepare(PreparedGeometry::Lines3 {
        segments: vec![[[0.0; 3], [1.0, 0.0, 0.0]]],
    });
    let material = session.add_material(Material::lines([0.9, 0.6, 0.2, 1.0], 1.0));
    let root = session.views().root();
    let (eye, bodies) = session.dispatch(|d| {
        let eye = d
            .spawn(SpawnBundle::new().at(r4, Pose::at(Point4::ZERO)))
            .unwrap();
        let bodies = (0..POPULATION)
            .map(|index| {
                d.spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(at(index, 0.0)))
                        .instance(Instance::new(tesseract, material).sectioned(material)),
                )
                .unwrap()
            })
            .collect();
        let views = d.domains.typed(r4).unwrap();
        views
            .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))
            .unwrap();
        views
            .add_view(ViewSpec::new(root, eye, Projection4 { focal: 2.0 }))
            .unwrap();
        let flat_eye = d
            .spawn(SpawnBundle::new().at(r3, Pose::at(Point3::new(0.0, 0.0, 4.0))))
            .unwrap();
        d.spawn(
            SpawnBundle::new()
                .at(r3, Pose::at(Point3::ZERO))
                .instance(Instance::new(line, material)),
        )
        .unwrap();
        d.domains
            .typed(r3)
            .unwrap()
            .add_view(ViewSpec::new(root, flat_eye, Identity3))
            .unwrap();
        (eye, bodies)
    });
    Stage {
        session,
        r4,
        eye,
        bodies,
    }
}

fn assert_same(threaded: &Publication, sequential: &Publication, when: &str) {
    assert_eq!(
        threaded.views.len(),
        sequential.views.len(),
        "{when}: the view count differs"
    );
    assert_eq!(threaded.stamp.tick, sequential.stamp.tick, "{when}");
    for (index, (threaded, sequential)) in threaded.views.iter().zip(&sequential.views).enumerate()
    {
        assert_eq!(
            (threaded.domain, threaded.target, threaded.placement),
            (sequential.domain, sequential.target, sequential.placement),
            "{when}: view {index} is another view"
        );
        assert_eq!(
            threaded.records.instances.rows(),
            sequential.records.instances.rows(),
            "{when}: view {index} instances differ"
        );
        assert!(
            threaded.records.segments() == sequential.records.segments(),
            "{when}: view {index} segments differ"
        );
        assert!(
            threaded.records.triangles() == sequential.records.triangles(),
            "{when}: view {index} triangles differ"
        );
    }
}

struct Pair {
    executor: Threads,
    threaded: Publication,
    sequential: Publication,
}

impl Pair {
    /// Returns whether the publish asked for a redraw and whether it reached the executor.
    fn publish(&mut self, session: &mut Session<Bare>, when: &str) -> (bool, bool) {
        let before = (self.threaded.content(), self.sequential.content());
        let shared = self.executor.shared.load(Ordering::Relaxed);
        session
            .publish_with(&mut self.threaded, 1.0, &self.executor)
            .unwrap();
        session
            .publish_with(&mut self.sequential, 1.0, &Sequential)
            .unwrap();
        assert_same(&self.threaded, &self.sequential, when);
        let drawn = self.threaded.content() != before.0;
        assert_eq!(
            drawn,
            self.sequential.content() != before.1,
            "{when}: only one of the two asked for a redraw"
        );
        (
            drawn,
            self.executor.shared.load(Ordering::Relaxed) != shared,
        )
    }
}

#[test]
fn a_publication_built_on_worker_threads_matches_the_one_built_on_the_caller() {
    let mut stage = stage();
    let mut pair = Pair {
        executor: Threads {
            shared: AtomicUsize::new(0),
        },
        threaded: Publication::default(),
        sequential: Publication::default(),
    };

    let (drawn, shared) = pair.publish(&mut stage.session, "first publish");
    assert!(drawn);
    assert!(
        shared,
        "the fixture never reached the executor, so nothing was compared"
    );
    let rows: Vec<usize> = pair
        .threaded
        .views
        .iter()
        .map(|view| view.records.instances.rows().len())
        .collect();
    assert_eq!(rows, [POPULATION, POPULATION, 1]);

    let domain = stage.session.domains_mut().typed(stage.r4).unwrap();
    for (index, &body) in stage.bodies.iter().enumerate().step_by(2) {
        domain.set_point(body, at(index, 0.03)).unwrap();
    }
    let (drawn, shared) = pair.publish(&mut stage.session, "after half the bodies moved");
    assert!(drawn && shared);

    let domain = stage.session.domains_mut().typed(stage.r4).unwrap();
    domain
        .set_point(stage.eye, Point4::new(0.05, 0.0, 0.0, 0.0))
        .unwrap();
    let (drawn, shared) = pair.publish(&mut stage.session, "after the eye moved");
    assert!(drawn && shared);

    let (drawn, _) = pair.publish(&mut stage.session, "with nothing changed");
    assert!(!drawn, "an unchanged scene asked for a redraw");
}

#[test]
fn the_published_value_is_what_the_latest_value_system_wrote_after_the_publication_systems() {
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
            return Err(DomainError::ChartBoundary);
        }
        value.clear();
        value.push(frames);
        Ok(())
    });
    let mut publication = Publication::default();

    session.publish(&mut publication).unwrap();
    assert_eq!(
        publication.value,
        [1],
        "the value missed this publish's publication systems or kept a replaced system's write"
    );
    session.publish(&mut publication).unwrap();
    assert_eq!(publication.value, [2]);

    let failure = PhaseError {
        phase: Phase::Publication,
        system: Some("frames"),
        cause: DomainError::ChartBoundary,
    };
    assert_eq!(session.publish(&mut publication), Err(failure));
    assert_eq!(
        session.phase_error(),
        Some(failure),
        "a failed value system left the session unfaulted"
    );
}
