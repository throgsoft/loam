use std::hint::black_box;
use std::num::NonZeroUsize;
use std::thread;
use std::time::Instant;

use loam_math::{EuclideanR4, Space};
use loam_runtime::{
    DomainBuilder, DomainHandle, Entity, Instance, LogCapacity, Material, Pose, PreparedGeometry,
    Projection4, Publication, Section4, Session, SimConfig, SpawnBundle, ViewSpec,
};
use loam_shape::polytope::Polytope4;
use loam_time::par::{ChunkSource, Executor, Sequential};

const POPULATIONS: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 256, 4096];
const VIEW_PAIRS: [usize; 2] = [1, 2];
const BATCHES: usize = 15;
const WARM_REPS: usize = 8;

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Empty {}
}

type Point4 = <EuclideanR4 as Space>::Point;
type Mutation = fn(&mut Fixture);

/// The same pool as `loam_app::par_native::ScopedThreads`, which this crate cannot name.
struct ScopedThreads {
    threads: usize,
}

impl Executor for ScopedThreads {
    fn parallelism(&self) -> usize {
        self.threads
    }

    fn for_each_chunk(&self, workers: usize, chunks: &(dyn ChunkSource + Sync)) {
        let workers = workers.clamp(1, self.threads);
        thread::scope(|scope| {
            for _ in 1..workers {
                scope.spawn(|| while chunks.run_next() {});
            }
            while chunks.run_next() {}
        });
    }
}

struct Fixture {
    session: Session<Empty>,
    domain: DomainHandle<EuclideanR4>,
    eye: Entity,
    entities: Vec<Entity>,
    publication: Publication,
    shifted: bool,
}

impl Fixture {
    fn new(population: usize, pairs: usize) -> Self {
        let mut session = Session::new(Empty::default(), SimConfig::default());
        let domain =
            session.register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity {
                dirty: population + 64,
                removals: 64,
            }));
        let geometry = session.prepare(PreparedGeometry::Polytope4 {
            polytope: Polytope4::Tesseract,
            scale: 0.04,
        });
        let material = session.add_material(Material::lines([0.9, 0.6, 0.2, 1.0], 1.0));
        let root = session.views().root();
        let (eye, entities) = session.dispatch(|dispatch| {
            let eye = dispatch
                .spawn(SpawnBundle::new().at(domain, Pose::at(Point4::ZERO)))
                .expect("R4 eye");
            let entities = (0..population)
                .map(|index| {
                    dispatch
                        .spawn(
                            SpawnBundle::new()
                                .at(domain, Pose::at(point4(index, 0.0)))
                                .instance(Instance::new(geometry, material).sectioned(material)),
                        )
                        .expect("R4 instance")
                })
                .collect();
            let views = dispatch.domains.typed(domain).expect("R4 domain");
            for _ in 0..pairs {
                views
                    .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))
                    .expect("R4 section view");
                views
                    .add_view(ViewSpec::new(root, eye, Projection4 { focal: 2.0 }))
                    .expect("R4 projection view");
            }
            (eye, entities)
        });
        Self {
            session,
            domain,
            eye,
            entities,
            publication: Publication::default(),
            shifted: false,
        }
    }

    fn idle(&mut self) {}

    fn moved(&mut self) {
        self.shifted = !self.shifted;
        let shift = if self.shifted { 0.002 } else { 0.0 };
        let domain = self
            .session
            .domains_mut()
            .typed(self.domain)
            .expect("R4 domain");
        for (index, &entity) in self.entities.iter().enumerate() {
            domain
                .set_point(entity, point4(index, shift))
                .expect("R4 pose edit");
        }
    }

    fn eye(&mut self) {
        self.shifted = !self.shifted;
        let shift = if self.shifted { 0.002 } else { 0.0 };
        self.session
            .domains_mut()
            .typed(self.domain)
            .expect("R4 domain")
            .set_point(self.eye, Point4::new(shift, 0.0, 0.0, 0.0))
            .expect("R4 eye edit");
    }

    fn publish(&mut self, executor: &dyn Executor) {
        self.session
            .publish_with(&mut self.publication, 1.0, executor)
            .expect("R4 publication");
        black_box(self.publication.stamp);
    }
}

fn point4(index: usize, shift: f32) -> Point4 {
    let x = (index % 128) as f32 * 0.016 - 1.0 + shift;
    let y = (index / 128) as f32 * 0.016 - 1.0;
    Point4::new(x, y, -3.0, 0.0)
}

fn reps(population: usize) -> usize {
    (65_536 / population.max(16)).clamp(2, 512)
}

fn median_nanos(
    fixture: &mut Fixture,
    executor: &dyn Executor,
    reps: usize,
    mutate: Mutation,
) -> f64 {
    for _ in 0..WARM_REPS {
        mutate(fixture);
        fixture.publish(executor);
    }
    let mut batches = [0.0_f64; BATCHES];
    for batch in &mut batches {
        let mut elapsed = 0u128;
        for _ in 0..reps {
            mutate(fixture);
            let start = Instant::now();
            fixture.publish(executor);
            elapsed += start.elapsed().as_nanos();
        }
        *batch = elapsed as f64 / reps as f64;
    }
    batches.sort_unstable_by(f64::total_cmp);
    batches[BATCHES / 2]
}

fn main() {
    // `cargo test --all-targets` runs this unoptimized and without `--bench`, where the timings mean nothing.
    if !std::env::args().any(|arg| arg == "--bench") {
        return;
    }
    let threads = thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let scoped = ScopedThreads { threads };
    println!(
        "mode={} threads={threads}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    );
    println!("case views population instance_rows seq_ns par_ns par_over_seq");
    let cases: [(&str, Mutation); 3] = [
        ("moved", Fixture::moved),
        ("eye", Fixture::eye),
        ("idle", Fixture::idle),
    ];
    for (name, mutate) in cases {
        for pairs in VIEW_PAIRS {
            for population in POPULATIONS {
                let reps = reps(population);
                let mut fixture = Fixture::new(population, pairs);
                let seq_ns = median_nanos(&mut fixture, &Sequential, reps, mutate);
                let par_ns = median_nanos(&mut fixture, &scoped, reps, mutate);
                let rows: usize = fixture
                    .publication
                    .views
                    .iter()
                    .map(|view| view.records.instances.rows().len())
                    .sum();
                println!(
                    "{name} {} {population} {rows} {seq_ns:.0} {par_ns:.0} {:.2}",
                    2 * pairs,
                    par_ns / seq_ns,
                );
            }
        }
    }
}
