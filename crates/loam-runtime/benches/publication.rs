use std::alloc::System;
use std::hint::black_box;
use std::mem::size_of_val;
use std::time::Instant;

use loam_math::{BlendedSpace, EuclideanR3, EuclideanR4, HyperbolicH3, LinearBlendX, Mat3, Space};
use loam_runtime::{
    Command, DepthEnvelope, DomainBuilder, DomainHandle, DomainRay, DomainSpace, Entity, Identity3,
    ImageRay, Input, Instance, LogCapacity, Material, Outcome, Pose, PreparedGeometry, Projection4,
    Publication, Section4, Session, SimConfig, SpawnBundle, ViewMapping, ViewSpec,
};
use loam_shape::polytope::Polytope4;
use loam_time::alloc::{current_snapshot, delta, AllocDelta, CountingAllocator};

#[global_allocator]
static ALLOCATOR: CountingAllocator<System> = CountingAllocator::new(System);

const BATCHES: usize = 7;
const CHURN_RING: usize = 1024;
const CHURN_SOAK: usize = 4096;
const WARM_REPS: usize = 4;
const BLEND_REPS: usize = 64;
const BLEND_START: f32 = -0.5;
const BLEND_END: f32 = 0.5;
const BLEND_TURN: f32 = 0.3;
const BLEND_SPAN: f32 = 0.05;
const BLEND_FOCAL: f32 = 2.0;
const BLEND_EYE: Point3 = Point3::new(-0.7, 0.0, 0.0);
const BLEND_IMAGE: Point3 = Point3::new(0.0, -0.0866, -0.25);
const BLEND_LANDMARK: Point4 = Point4::new(1.5, 0.0, -4.0, 0.0);
const BLEND_LANDMARK_SCALE: f32 = 1.0;

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Empty {}
}

type Point3 = <EuclideanR3 as Space>::Point;
type Point4 = <EuclideanR4 as Space>::Point;

#[derive(Clone, Copy)]
struct Counts {
    instances: usize,
    segments: usize,
    triangles: usize,
    refusals: u32,
    output_bytes: usize,
}

struct Allocation {
    publication: AllocDelta,
    total: AllocDelta,
}

struct LinesFixture {
    session: Session<Empty>,
    domain: DomainHandle<EuclideanR3>,
    eye: Entity,
    entities: Vec<Entity>,
    geometry: loam_runtime::PreparedId,
    material: loam_runtime::MaterialId,
    publication: Publication,
    shifted: bool,
}

impl LinesFixture {
    fn new(population: usize) -> Self {
        let mut session = Session::new(Empty::default(), SimConfig::default());
        let domain =
            session.register_domain(DomainBuilder::new("r3", EuclideanR3).tracked(LogCapacity {
                dirty: population + 64,
                removals: CHURN_RING,
            }));
        let geometry = session.prepare(PreparedGeometry::Lines3 {
            segments: cube_lines(),
        });
        let material = session.add_material(Material::lines([0.4, 0.7, 1.0, 1.0], 1.0));
        let root = session.views().root();
        let (eye, entities) = session.dispatch(|dispatch| {
            let eye = dispatch
                .spawn(SpawnBundle::new().at(domain, Pose::at(Point3::ZERO)))
                .expect("R3 eye");
            let entities = (0..population)
                .map(|index| {
                    dispatch
                        .spawn(
                            SpawnBundle::new()
                                .at(domain, Pose::at(point3(index, 0.0)))
                                .instance(Instance::new(geometry, material)),
                        )
                        .expect("R3 instance")
                })
                .collect();
            dispatch
                .domains
                .typed(domain)
                .expect("R3 domain")
                .add_view(ViewSpec::new(root, eye, Identity3))
                .expect("R3 view");
            (eye, entities)
        });
        Self {
            session,
            domain,
            eye,
            entities,
            geometry,
            material,
            publication: Publication::default(),
            shifted: false,
        }
    }

    fn idle(&mut self) {}

    fn edit_one(&mut self) {
        self.shifted = !self.shifted;
        let shift = if self.shifted { 0.002 } else { 0.0 };
        self.session
            .domains_mut()
            .typed(self.domain)
            .expect("R3 domain")
            .set_point(self.entities[0], point3(0, shift))
            .expect("R3 pose edit");
    }

    fn edit_all(&mut self) {
        self.shifted = !self.shifted;
        let shift = if self.shifted { 0.002 } else { 0.0 };
        let domain = self
            .session
            .domains_mut()
            .typed(self.domain)
            .expect("R3 domain");
        for (index, &entity) in self.entities.iter().enumerate() {
            domain
                .set_point(entity, point3(index, shift))
                .expect("R3 pose edit");
        }
    }

    fn edit_eye(&mut self) {
        self.shifted = !self.shifted;
        let shift = if self.shifted { 0.002 } else { 0.0 };
        self.session
            .domains_mut()
            .typed(self.domain)
            .expect("R3 domain")
            .set_point(self.eye, Point3::new(shift, 0.0, 0.0))
            .expect("R3 eye edit");
    }

    fn churn(&mut self) {
        self.shifted = !self.shifted;
        let shift = if self.shifted { 0.002 } else { 0.0 };
        let retired = self.entities.swap_remove(0);
        self.session.submit(Command::Despawn(retired));
        let spawned = self.session.submit(Command::Spawn(
            SpawnBundle::new()
                .at(self.domain, Pose::at(point3(0, shift)))
                .instance(Instance::new(self.geometry, self.material)),
        ));
        self.session
            .boundary(Input::default())
            .expect("R3 churn boundary");
        let entity = self
            .session
            .results()
            .iter()
            .find(|result| result.request == spawned)
            .and_then(|result| match result.outcome {
                Ok(Outcome::Spawned(entity)) => Some(entity),
                _ => None,
            })
            .expect("R3 replacement");
        self.entities.push(entity);
    }

    fn publish(&mut self) {
        self.session
            .publish(&mut self.publication)
            .expect("R3 publication");
        black_box(self.publication.stamp);
    }

    fn counts(&self) -> Counts {
        counts(&self.publication)
    }
}

struct SectionFixture {
    session: Session<Empty>,
    domain: DomainHandle<EuclideanR4>,
    eye: Entity,
    entities: Vec<Entity>,
    publication: Publication,
    shifted: bool,
}

impl SectionFixture {
    fn new(population: usize) -> Self {
        let mut session = Session::new(Empty::default(), SimConfig::default());
        let domain =
            session.register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity {
                dirty: population + 64,
                removals: CHURN_RING,
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
            dispatch
                .domains
                .typed(domain)
                .expect("R4 domain")
                .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))
                .expect("R4 view");
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

    fn edit_one(&mut self) {
        self.shifted = !self.shifted;
        let shift = if self.shifted { 0.002 } else { 0.0 };
        self.session
            .domains_mut()
            .typed(self.domain)
            .expect("R4 domain")
            .set_point(self.entities[0], point4(0, shift))
            .expect("R4 pose edit");
    }

    fn edit_all(&mut self) {
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

    fn edit_eye(&mut self) {
        self.shifted = !self.shifted;
        let shift = if self.shifted { 0.002 } else { 0.0 };
        self.session
            .domains_mut()
            .typed(self.domain)
            .expect("R4 domain")
            .set_point(self.eye, Point4::new(shift, 0.0, 0.0, 0.0))
            .expect("R4 eye edit");
    }

    fn publish(&mut self) {
        self.session
            .publish(&mut self.publication)
            .expect("R4 publication");
        black_box(self.publication.stamp);
    }

    fn counts(&self) -> Counts {
        counts(&self.publication)
    }
}

type Blend = BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>;

struct ChartIdentity;

impl<S: DomainSpace<Point = Point3, Frame = Mat3>> ViewMapping<S> for ChartIdentity {
    fn name(&self) -> &'static str {
        "chart-identity"
    }

    fn image_point(&self, eye: &Pose<S>, point: Point3) -> Option<[f32; 3]> {
        let inverse = eye.frame.inverse();
        inverse
            .is_finite()
            .then(|| (inverse * (point - eye.point)).to_array())
    }

    fn lift(&self, _eye: &Pose<S>, _ray: &ImageRay) -> Option<DomainRay<S>> {
        None
    }

    fn ray_lift(&self) -> bool {
        false
    }

    fn depth_envelope(&self) -> DepthEnvelope {
        DepthEnvelope {
            near: 0.0,
            far: f32::INFINITY,
        }
    }
}

fn blend_space() -> Blend {
    BlendedSpace::new(
        EuclideanR3,
        HyperbolicH3,
        LinearBlendX::new(BLEND_START, BLEND_END).expect("blend zone width"),
    )
}

struct BlendFixture {
    session: Session<Empty>,
    blend: DomainHandle<Blend>,
    r4: DomainHandle<EuclideanR4>,
    mark: Entity,
    mark_pose: Pose<Blend>,
    landmark: Entity,
    landmark_pose: Pose<EuclideanR4>,
    publication: Publication,
}

impl BlendFixture {
    fn new() -> Self {
        let mut session = Session::new(Empty::default(), SimConfig::default());
        let blend = session.register_domain(
            DomainBuilder::new("blend", blend_space())
                .tracked(LogCapacity::default())
                .fields(),
        );
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let space = blend_space();
        let mark_edges = session.prepare(PreparedGeometry::Lines3 {
            segments: vec![[[BLEND_SPAN, 0.0, 0.0], [-BLEND_SPAN, 0.0, 0.0]]],
        });
        let edges4 = session.prepare(PreparedGeometry::edges_of(
            Polytope4::Tesseract.topology(),
            BLEND_LANDMARK_SCALE,
        ));
        let material = session.add_material(Material::lines([1.0, 1.0, 1.0, 0.95], 1.6));
        let root = session.views().root();
        let turn = Mat3::from_rotation_y(BLEND_TURN);
        let (mark, landmark) = session.dispatch(|dispatch| {
            let eye = dispatch
                .spawn(SpawnBundle::new().at(
                    blend,
                    Pose {
                        point: BLEND_EYE,
                        frame: turn,
                    },
                ))
                .expect("blend eye");
            let mark = dispatch
                .spawn(
                    SpawnBundle::new()
                        .at(blend, Pose::new(&space, BLEND_EYE + turn * BLEND_IMAGE))
                        .instance(Instance::new(mark_edges, material)),
                )
                .expect("blend mark");
            dispatch
                .domains
                .typed(blend)
                .expect("blend domain")
                .add_view(ViewSpec::new(root, eye, ChartIdentity))
                .expect("blend view");
            let walker = dispatch
                .spawn(SpawnBundle::new().at(r4, Pose::at(Point4::ZERO)))
                .expect("R4 eye");
            let landmark = dispatch
                .spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(BLEND_LANDMARK))
                        .instance(Instance::new(edges4, material)),
                )
                .expect("R4 landmark");
            dispatch
                .domains
                .typed(r4)
                .expect("R4 domain")
                .add_view(ViewSpec::new(
                    root,
                    walker,
                    Projection4 { focal: BLEND_FOCAL },
                ))
                .expect("R4 view");
            (mark, landmark)
        });
        let mark_pose = *session
            .domains()
            .read(blend)
            .expect("blend domain")
            .poses()
            .get(mark)
            .expect("blend mark pose");
        let landmark_pose = *session
            .domains()
            .read(r4)
            .expect("R4 domain")
            .poses()
            .get(landmark)
            .expect("R4 landmark pose");
        Self {
            session,
            blend,
            r4,
            mark,
            mark_pose,
            landmark,
            landmark_pose,
            publication: Publication::default(),
        }
    }

    fn idle(&mut self) {}

    fn move_blend(&mut self) {
        self.session
            .domains_mut()
            .typed(self.blend)
            .expect("blend domain")
            .set_pose(self.mark, self.mark_pose)
            .expect("blend pose edit");
    }

    fn move_r4(&mut self) {
        self.session
            .domains_mut()
            .typed(self.r4)
            .expect("R4 domain")
            .set_pose(self.landmark, self.landmark_pose)
            .expect("R4 pose edit");
    }

    fn publish(&mut self) {
        self.session
            .publish(&mut self.publication)
            .expect("blend publication");
        black_box(self.publication.stamp);
    }

    fn counts(&self) -> Counts {
        counts(&self.publication)
    }
}

fn cube_lines() -> Vec<[[f32; 3]; 2]> {
    let vertices = [
        [-0.04, -0.04, -0.04],
        [0.04, -0.04, -0.04],
        [-0.04, 0.04, -0.04],
        [0.04, 0.04, -0.04],
        [-0.04, -0.04, 0.04],
        [0.04, -0.04, 0.04],
        [-0.04, 0.04, 0.04],
        [0.04, 0.04, 0.04],
    ];
    [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    .map(|[a, b]| [vertices[a], vertices[b]])
    .to_vec()
}

fn point3(index: usize, shift: f32) -> Point3 {
    let x = (index % 100) as f32 * 0.02 - 1.0 + shift;
    let y = ((index / 100) % 100) as f32 * 0.02 - 1.0;
    Point3::new(x, y, -3.0)
}

fn point4(index: usize, shift: f32) -> Point4 {
    let x = (index % 100) as f32 * 0.02 - 1.0 + shift;
    let y = ((index / 100) % 100) as f32 * 0.02 - 1.0;
    Point4::new(x, y, -3.0, 0.0)
}

fn counts(publication: &Publication) -> Counts {
    let view = publication.views.first().expect("published view");
    let instances = view.records.instances.rows();
    let segments = view.records.segments();
    let triangles = view.records.triangles();
    Counts {
        instances: instances.len(),
        segments: segments.len(),
        triangles: triangles.len(),
        refusals: view.records.refusals().count,
        output_bytes: size_of_val(instances) + size_of_val(segments) + size_of_val(triangles),
    }
}

fn median_time<T>(
    state: &mut T,
    reps: usize,
    mut mutate: impl FnMut(&mut T),
    mut publish: impl FnMut(&mut T),
) -> (f64, f64) {
    let mut publication = [0.0; BATCHES];
    let mut total = [0.0; BATCHES];
    for batch in 0..BATCHES {
        let total_start = Instant::now();
        let mut publication_elapsed = 0u128;
        for _ in 0..reps {
            mutate(state);
            let publication_start = Instant::now();
            publish(state);
            publication_elapsed += publication_start.elapsed().as_nanos();
        }
        publication[batch] = publication_elapsed as f64 / reps as f64;
        total[batch] = total_start.elapsed().as_nanos() as f64 / reps as f64;
    }
    publication.sort_unstable_by(f64::total_cmp);
    total.sort_unstable_by(f64::total_cmp);
    (publication[BATCHES / 2], total[BATCHES / 2])
}

fn allocation<T>(
    state: &mut T,
    mutate: impl FnOnce(&mut T),
    publish: impl FnOnce(&mut T),
) -> Allocation {
    let total_start = current_snapshot().expect("counting allocator");
    mutate(state);
    let publication_start = current_snapshot().expect("counting allocator");
    publish(state);
    let end = current_snapshot().expect("counting allocator");
    Allocation {
        publication: delta(publication_start, end),
        total: delta(total_start, end),
    }
}

fn run_case<T>(
    filters: &[String],
    name: &str,
    population: usize,
    state: impl FnOnce() -> T,
    mut mutate: impl FnMut(&mut T),
    mut publish: impl FnMut(&mut T),
    counts: impl Fn(&T) -> Counts,
) {
    if !selected(filters, name) {
        return;
    }
    let mut state = state();
    let reps = reps(population);
    let before_warm = current_snapshot().expect("counting allocator");
    for _ in 0..WARM_REPS {
        mutate(&mut state);
        publish(&mut state);
    }
    let warm = delta(before_warm, current_snapshot().expect("counting allocator"));
    let (publication_ns, total_ns) = median_time(&mut state, reps, &mut mutate, &mut publish);
    let allocation = allocation(&mut state, &mut mutate, &mut publish);
    let counts = counts(&state);
    let mutation_alloc_count = allocation
        .total
        .alloc_count
        .saturating_sub(allocation.publication.alloc_count);
    let mutation_alloc_bytes = allocation
        .total
        .alloc_bytes
        .saturating_sub(allocation.publication.alloc_bytes);
    println!(
        "{name} {population} {reps} {publication_ns:.1} {total_ns:.1} {} {} {mutation_alloc_count} {mutation_alloc_bytes} {} {} {} {} {} {}",
        allocation.publication.alloc_count,
        allocation.publication.alloc_bytes,
        warm.net_bytes,
        counts.instances,
        counts.segments,
        counts.triangles,
        counts.refusals,
        counts.output_bytes,
    );
}

fn reps(population: usize) -> usize {
    match population {
        1 => BLEND_REPS,
        1_000 => 10,
        _ => 2,
    }
}

fn filters() -> Vec<String> {
    std::env::args()
        .skip(1)
        .filter(|arg| !arg.starts_with('-') && arg.parse::<f64>().is_err())
        .collect()
}

fn selected(filters: &[String], name: &str) -> bool {
    filters.is_empty() || filters.iter().any(|filter| name.contains(filter.as_str()))
}

fn soak_churn(state: &mut LinesFixture, population: usize) {
    state.publish();
    let before = current_snapshot().expect("counting allocator");
    for _ in 0..CHURN_RING {
        state.churn();
    }
    let turnover = current_snapshot().expect("counting allocator");
    for _ in CHURN_RING..CHURN_SOAK {
        state.churn();
    }
    let after = current_snapshot().expect("counting allocator");
    let warm = delta(before, turnover);
    let post = delta(turnover, after);
    let publication_start = current_snapshot().expect("counting allocator");
    state.publish();
    let publication = delta(
        publication_start,
        current_snapshot().expect("counting allocator"),
    );
    assert_eq!(state.counts().instances, population);
    println!("churn_soak stage iterations alloc_count alloc_bytes dealloc_count net_bytes");
    println!(
        "churn_soak warm {CHURN_RING} {} {} {} {}",
        warm.alloc_count, warm.alloc_bytes, warm.dealloc_count, warm.net_bytes,
    );
    println!(
        "churn_soak post {} {} {} {} {}",
        CHURN_SOAK - CHURN_RING,
        post.alloc_count,
        post.alloc_bytes,
        post.dealloc_count,
        post.net_bytes,
    );
    println!(
        "churn_soak catchup_publication 1 {} {} {} {}",
        publication.alloc_count,
        publication.alloc_bytes,
        publication.dealloc_count,
        publication.net_bytes,
    );
}

fn main() {
    let filters = filters();
    println!(
        "environment os={} arch={} host={} mode={} source={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::env::var("COMPUTERNAME")
            .or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_else(|_| "unknown".to_owned()),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        std::env::var("LOAM_SOURCE_MANIFEST").unwrap_or_else(|_| "unspecified".to_owned()),
    );
    println!("case population reps publication_ns mutation_plus_publication_ns publication_alloc_count publication_alloc_bytes mutation_alloc_count mutation_alloc_bytes warm_retained_bytes instances segments triangles refusals output_bytes");
    for population in [1_000, 10_000] {
        run_case(
            &filters,
            "r3_lines_idle",
            population,
            || LinesFixture::new(population),
            LinesFixture::idle,
            LinesFixture::publish,
            LinesFixture::counts,
        );
        run_case(
            &filters,
            "r3_lines_one",
            population,
            || LinesFixture::new(population),
            LinesFixture::edit_one,
            LinesFixture::publish,
            LinesFixture::counts,
        );
        run_case(
            &filters,
            "r3_lines_all",
            population,
            || LinesFixture::new(population),
            LinesFixture::edit_all,
            LinesFixture::publish,
            LinesFixture::counts,
        );
        run_case(
            &filters,
            "r3_lines_eye",
            population,
            || LinesFixture::new(population),
            LinesFixture::edit_eye,
            LinesFixture::publish,
            LinesFixture::counts,
        );
        run_case(
            &filters,
            "r4_section_one",
            population,
            || SectionFixture::new(population),
            SectionFixture::edit_one,
            SectionFixture::publish,
            SectionFixture::counts,
        );
        run_case(
            &filters,
            "r4_section_all",
            population,
            || SectionFixture::new(population),
            SectionFixture::edit_all,
            SectionFixture::publish,
            SectionFixture::counts,
        );
        run_case(
            &filters,
            "r4_section_eye",
            population,
            || SectionFixture::new(population),
            SectionFixture::edit_eye,
            SectionFixture::publish,
            SectionFixture::counts,
        );
    }
    run_case(
        &filters,
        "blend_publish_blend_moved",
        1,
        BlendFixture::new,
        BlendFixture::move_blend,
        BlendFixture::publish,
        BlendFixture::counts,
    );
    run_case(
        &filters,
        "blend_publish_r4_moved",
        1,
        BlendFixture::new,
        BlendFixture::move_r4,
        BlendFixture::publish,
        BlendFixture::counts,
    );
    run_case(
        &filters,
        "blend_publish_idle",
        1,
        BlendFixture::new,
        BlendFixture::idle,
        BlendFixture::publish,
        BlendFixture::counts,
    );
    run_case(
        &filters,
        "r3_churn",
        10_000,
        || {
            let mut churn = LinesFixture::new(10_000);
            soak_churn(&mut churn, 10_000);
            churn
        },
        LinesFixture::churn,
        LinesFixture::publish,
        LinesFixture::counts,
    );
}
