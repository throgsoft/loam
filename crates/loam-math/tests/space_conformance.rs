//! Residuals use the fixture metric; point residuals use Space::distance.

use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use loam_math::{
    Bivector, Bivector2, Bivector4, BlendedSpace, ConformallyFlat, EuclideanR2, EuclideanR3,
    EuclideanR4, HyperbolicH3, Iso2, Iso3, Iso3H, Iso4, Iso4Flat, IsometryGroup, LensSpace,
    LinearBlendX, Space, SphericalS3, SphericalS3Embedded,
};

#[derive(Clone, Copy)]
struct Tol {
    point: f32,

    vector: f32,

    scalar: f32,

    degenerate: f32,
}

trait SpaceFixture {
    type Point: Copy;
    type Vector: Copy;
    type S: Space<Point = Self::Point, Vector = Self::Vector>;

    fn space(&self) -> Self::S;

    fn points(&self) -> Vec<Self::Point>;

    fn tangents(&self, at: Self::Point) -> Vec<Self::Vector>;

    fn inner(&self, at: Self::Point, u: Self::Vector, v: Self::Vector) -> f32;

    fn combine(&self, u: Self::Vector, s: f32, v: Self::Vector, t: f32) -> Self::Vector;

    fn degenerate_pairs(&self) -> Vec<(Self::Point, Self::Point)>;

    fn curvature(&self) -> Option<f32>;

    fn tol(&self) -> Tol;

    fn point_components(&self, p: Self::Point) -> [f32; 4];

    fn vector_components(&self, v: Self::Vector) -> [f32; 4];
}

trait IsometryFixture: SpaceFixture
where
    Self::S: IsometryGroup<Iso = Self::Iso>,
{
    type Iso: Copy;

    fn isos(&self) -> Vec<Self::Iso>;
}

// Marsaglia, Xorshift RNGs, 2003, §3.
struct Xorshift32(u32);

impl Xorshift32 {
    fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn signed_unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn ball_samples(seed: u32, count: usize, max_radius: f32) -> Vec<Vec3> {
    let mut rng = Xorshift32::new(seed);
    (0..count)
        .map(|_| {
            let dir =
                Vec3::new(rng.signed_unit(), rng.signed_unit(), rng.signed_unit()).normalize();
            dir * (0.1 + 0.9 * rng.signed_unit().abs()) * max_radius
        })
        .collect()
}

fn agrees(got: f32, want: f32, tol: f32) -> bool {
    (got - want).abs() <= tol * want.abs().max(1.0)
}

fn chart_magnitude<F: SpaceFixture>(f: &F, p: F::Point) -> f32 {
    f.point_components(p)
        .iter()
        .fold(0.0f32, |m, c| m.max(c.abs()))
}

fn metric_norm<F: SpaceFixture>(f: &F, at: F::Point, v: F::Vector) -> f32 {
    let squared = f.inner(at, v, v);
    assert!(
        squared.is_finite() && squared >= 0.0,
        "invalid squared norm: {squared}"
    );
    squared.sqrt()
}

fn metric_residual<F: SpaceFixture>(f: &F, at: F::Point, u: F::Vector, v: F::Vector) -> f32 {
    metric_norm(f, at, f.combine(u, 1.0, v, -1.0))
}

fn metric_angle<F: SpaceFixture>(f: &F, at: F::Point, u: F::Vector, v: F::Vector) -> f32 {
    let denom = metric_norm(f, at, u) * metric_norm(f, at, v);
    (f.inner(at, u, v) / denom).clamp(-1.0, 1.0).acos()
}

const TRIANGLE_SIDE: f32 = 0.1;

const GAUSS_BONNET_RELATIVE_TOL: f32 = 0.05;

mod invariants {
    use super::*;

    pub fn distance_is_symmetric_and_zero_on_the_diagonal<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        for &a in &points {
            let self_distance = s.distance(a, a);
            assert!(
                self_distance.abs() <= tol.point,
                "d(a, a) = {self_distance} at {:?}",
                f.point_components(a)
            );
            for &b in &points {
                let ab = s.distance(a, b);
                let ba = s.distance(b, a);
                assert!(
                    agrees(ab, ba, tol.scalar),
                    "d(a, b) = {ab} but d(b, a) = {ba} at {:?} {:?}",
                    f.point_components(a),
                    f.point_components(b)
                );
            }
        }
    }

    pub fn distance_is_finite_and_nonnegative<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let points = f.points();
        for &a in &points {
            for &b in &points {
                let d = s.distance(a, b);
                assert!(
                    d.is_finite() && d >= 0.0,
                    "d = {d} at {:?} {:?}",
                    f.point_components(a),
                    f.point_components(b)
                );
            }
        }
    }

    pub fn distance_satisfies_the_triangle_inequality<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();

        let n = points.len();
        let mut d = Vec::with_capacity(n * n);
        for &a in &points {
            for &b in &points {
                d.push(s.distance(a, b));
            }
        }
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let direct = d[i * n + k];
                    let via = d[i * n + j] + d[j * n + k];
                    assert!(
                        direct <= via + tol.scalar * via.max(1.0),
                        "d(a, c) = {direct} exceeds d(a, b) + d(b, c) = {via} at \
                         {:?} {:?} {:?}",
                        f.point_components(points[i]),
                        f.point_components(points[j]),
                        f.point_components(points[k])
                    );
                }
            }
        }
    }

    pub fn exp_inverts_log_on_sampled_pairs<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        for &a in &points {
            for &b in &points {
                let recovered = s.exp(a, s.log(a, b));
                assert_point_agrees(f, &s, recovered, b, tol.point, "exp(a, log(a, b))");
            }
        }
    }

    pub fn log_inverts_exp_on_sampled_tangents<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        for a in f.points() {
            for v in f.tangents(a) {
                let recovered = s.log(a, s.exp(a, v));
                let residual = metric_residual(f, a, recovered, v);
                let scale = metric_norm(f, a, v).max(1.0);
                assert!(
                    residual <= tol.vector * scale,
                    "log(a, exp(a, v)) missed v by {residual} at {:?} {:?}",
                    f.point_components(a),
                    f.vector_components(v)
                );
            }
        }
    }

    pub fn log_magnitude_equals_geodesic_distance<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        for &a in &points {
            for &b in &points {
                let magnitude = metric_norm(f, a, s.log(a, b));
                let distance = s.distance(a, b);
                assert!(
                    agrees(magnitude, distance, tol.scalar),
                    "|log(a, b)| = {magnitude} but d(a, b) = {distance} at {:?} {:?}",
                    f.point_components(a),
                    f.point_components(b)
                );
            }
        }
    }

    pub fn exp_advances_by_the_metric_norm_of_its_tangent<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        for a in f.points() {
            for v in f.tangents(a) {
                let travelled = s.distance(a, s.exp(a, v));
                let norm = metric_norm(f, a, v);
                assert!(
                    agrees(travelled, norm, tol.scalar),
                    "exp travelled {travelled} for a tangent of norm {norm} at {:?} {:?}",
                    f.point_components(a),
                    f.vector_components(v)
                );
            }
        }
    }

    pub fn distance_is_invariant_under_isometry<F: IsometryFixture>(f: &F)
    where
        F::S: IsometryGroup<Iso = F::Iso>,
    {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        for g in f.isos() {
            for &a in &points {
                for &b in &points {
                    let before = s.distance(a, b);
                    let after = s.distance(s.iso_apply(g, a), s.iso_apply(g, b));
                    assert!(
                        agrees(after, before, tol.scalar),
                        "isometry changed d from {before} to {after} at {:?} {:?}",
                        f.point_components(a),
                        f.point_components(b)
                    );
                }
            }
        }
    }

    pub fn iso_identity_is_neutral<F: IsometryFixture>(f: &F)
    where
        F::S: IsometryGroup<Iso = F::Iso>,
    {
        let s = f.space();
        let tol = f.tol();
        let id = s.iso_identity();
        let isos = f.isos();
        for p in f.points() {
            assert_point_agrees(f, &s, s.iso_apply(id, p), p, tol.point, "id.p");
            for &g in &isos {
                let want = s.iso_apply(g, p);
                assert_point_agrees(
                    f,
                    &s,
                    s.iso_apply(s.iso_compose(id, g), p),
                    want,
                    tol.point,
                    "(id . g).p",
                );
                assert_point_agrees(
                    f,
                    &s,
                    s.iso_apply(s.iso_compose(g, id), p),
                    want,
                    tol.point,
                    "(g . id).p",
                );
            }
        }
    }

    pub fn iso_inverse_is_two_sided<F: IsometryFixture>(f: &F)
    where
        F::S: IsometryGroup<Iso = F::Iso>,
    {
        let s = f.space();
        let tol = f.tol();
        for g in f.isos() {
            let inv = s.iso_inverse(g);
            for p in f.points() {
                assert_point_agrees(
                    f,
                    &s,
                    s.iso_apply(s.iso_compose(g, inv), p),
                    p,
                    tol.point,
                    "(g . g^-1).p",
                );
                assert_point_agrees(
                    f,
                    &s,
                    s.iso_apply(s.iso_compose(inv, g), p),
                    p,
                    tol.point,
                    "(g^-1 . g).p",
                );
            }
        }
    }

    pub fn iso_compose_is_associative<F: IsometryFixture>(f: &F)
    where
        F::S: IsometryGroup<Iso = F::Iso>,
    {
        let s = f.space();
        let tol = f.tol();
        let isos = f.isos();
        let points = f.points();
        for &a in &isos {
            for &b in &isos {
                for &c in &isos {
                    let left = s.iso_compose(s.iso_compose(a, b), c);
                    let right = s.iso_compose(a, s.iso_compose(b, c));
                    for &p in &points {
                        assert_point_agrees(
                            f,
                            &s,
                            s.iso_apply(left, p),
                            s.iso_apply(right, p),
                            tol.point,
                            "((a . b) . c).p vs (a . (b . c)).p",
                        );
                    }
                }
            }
        }
    }

    pub fn iso_compose_matches_sequential_apply<F: IsometryFixture>(f: &F)
    where
        F::S: IsometryGroup<Iso = F::Iso>,
    {
        let s = f.space();
        let tol = f.tol();
        let isos = f.isos();
        for &a in &isos {
            for &b in &isos {
                for p in f.points() {
                    assert_point_agrees(
                        f,
                        &s,
                        s.iso_apply(s.iso_compose(a, b), p),
                        s.iso_apply(a, s.iso_apply(b, p)),
                        tol.point,
                        "(a . b).p vs a.(b.p)",
                    );
                }
            }
        }
    }

    pub fn iso_transport_is_the_differential_of_iso_apply<F: IsometryFixture>(f: &F)
    where
        F::S: IsometryGroup<Iso = F::Iso>,
    {
        let s = f.space();
        let tol = f.tol();
        for g in f.isos() {
            for p in f.points() {
                for v in f.tangents(p) {
                    let moved = s.iso_apply(g, s.exp(p, v));
                    let transported = s.exp(s.iso_apply(g, p), s.iso_transport(g, p, v));
                    assert_point_agrees(
                        f,
                        &s,
                        moved,
                        transported,
                        tol.point,
                        "g.exp(p, v) vs exp(g.p, dg(v))",
                    );
                }
            }
        }
    }

    pub fn iso_transport_preserves_the_metric_norm<F: IsometryFixture>(f: &F)
    where
        F::S: IsometryGroup<Iso = F::Iso>,
    {
        let s = f.space();
        let tol = f.tol();
        for g in f.isos() {
            for p in f.points() {
                let moved = s.iso_apply(g, p);
                for v in f.tangents(p) {
                    let before = metric_norm(f, p, v);
                    let after = metric_norm(f, moved, s.iso_transport(g, p, v));
                    assert!(
                        agrees(after, before, tol.scalar),
                        "iso_transport changed the norm from {before} to {after} at \
                         {:?} {:?}",
                        f.point_components(p),
                        f.vector_components(v)
                    );
                }
            }
        }
    }

    pub fn parallel_transport_preserves_the_metric_norm<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        for &a in &points {
            for &b in &points {
                for v in f.tangents(a) {
                    let before = metric_norm(f, a, v);
                    let after = metric_norm(f, b, s.parallel_transport(a, b, v));
                    assert!(
                        agrees(after, before, tol.scalar),
                        "transport changed the norm from {before} to {after} at \
                         {:?} {:?} {:?}",
                        f.point_components(a),
                        f.point_components(b),
                        f.vector_components(v)
                    );
                }
            }
        }
    }

    pub fn parallel_transport_is_linear_in_the_transported_vector<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();

        let (cu, cw) = (1.5, -0.75);
        for &a in &points {
            let tangents = f.tangents(a);
            for &b in &points {
                for (i, &u) in tangents.iter().enumerate() {
                    for &w in &tangents[i..] {
                        let combined = s.parallel_transport(a, b, f.combine(u, cu, w, cw));
                        let separate = f.combine(
                            s.parallel_transport(a, b, u),
                            cu,
                            s.parallel_transport(a, b, w),
                            cw,
                        );
                        let residual = metric_residual(f, b, combined, separate);
                        let scale = metric_norm(f, b, separate).max(1.0);
                        assert!(
                            residual <= tol.vector * scale,
                            "transport is not linear by {residual} at {:?} {:?} {:?} {:?}",
                            f.point_components(a),
                            f.point_components(b),
                            f.vector_components(u),
                            f.vector_components(w)
                        );
                    }
                }
            }
        }
    }

    pub fn parallel_transport_along_one_segment_matches_parallel_transport<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        for &a in &points {
            for v in f.tangents(a) {
                let residual = metric_residual(f, a, s.parallel_transport_along(&[], v), v);
                assert!(residual <= tol.vector, "empty path moved v by {residual}");
                let residual = metric_residual(f, a, s.parallel_transport_along(&[a], v), v);
                assert!(
                    residual <= tol.vector,
                    "one-point path moved v by {residual}"
                );
                for &b in &points {
                    let along = s.parallel_transport_along(&[a, b], v);
                    let direct = s.parallel_transport(a, b, v);
                    let residual = metric_residual(f, b, along, direct);
                    let scale = metric_norm(f, b, direct).max(1.0);
                    assert!(
                        residual <= tol.vector * scale,
                        "polyline transport differs from segment transport by {residual} \
                         at {:?} {:?}",
                        f.point_components(a),
                        f.point_components(b)
                    );
                }
            }
        }
    }

    pub fn parallel_transport_carries_a_geodesic_tangent_along_its_own_geodesic<F: SpaceFixture>(
        f: &F,
    ) {
        let (ratio, a, b) = worst_geodesic_tangent_ratio(f);
        assert!(
            ratio <= 1.0,
            "transport of log(a, b) misses the forward tangent at b by \
             {ratio} of the vector budget at {:?} {:?}",
            f.point_components(a),
            f.point_components(b)
        );
    }

    // Pennec, Parallel Transport with Pole Ladder, 2018, §3; Helgason, Symmetric Spaces, 1978, ch. IV §3.
    pub fn parallel_transport_matches_the_one_its_own_geodesics_imply<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        let mut worst = (0.0f32, points[0], points[0], f.tangents(points[0])[0]);
        for &a in &points {
            for &b in &points {
                for v in f.tangents(a) {
                    let ladder = pole_ladder(f, &s, a, b, v);
                    let residual = metric_residual(f, b, s.parallel_transport(a, b, v), ladder);
                    let ratio = residual / (tol.vector * metric_norm(f, b, ladder).max(1.0));

                    if ratio.is_nan() || ratio > worst.0 {
                        worst = (ratio, a, b, v);
                    }
                }
            }
        }
        let (ratio, a, b, v) = worst;
        assert!(
            ratio <= 1.0,
            "transport misses the map its own geodesics imply by {ratio} of \
             the vector budget at {:?} {:?} {:?}",
            f.point_components(a),
            f.point_components(b),
            f.vector_components(v)
        );
    }

    // do Carmo, Differential Geometry of Curves and Surfaces, 1976, §4.5.
    pub fn geodesic_triangle_angle_excess_matches_gauss_bonnet<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let l = TRIANGLE_SIDE;
        let apex = f.points()[0];
        let (e1, e2) = metric_orthonormal_pair(f, apex);

        let b = s.exp(apex, f.combine(e1, l, e2, 0.0));
        let c = s.exp(apex, f.combine(e1, l * 0.5, e2, l * 3.0_f32.sqrt() * 0.5));

        let angle = |at, u, w| metric_angle(f, at, s.log(at, u), s.log(at, w));
        let sum = angle(apex, b, c) + angle(b, apex, c) + angle(c, apex, b);
        assert!(sum.is_finite(), "triangle angle sum is {sum}");

        let area = 3.0_f32.sqrt() / 4.0 * l * l;
        let excess = sum - std::f32::consts::PI;
        let k = f.curvature().expect("constant-curvature fixture");
        assert!(
            (excess - k * area).abs() <= GAUSS_BONNET_RELATIVE_TOL * area,
            "angle excess {excess} misses K * area = {} at K = {k}",
            k * area
        );
    }

    pub fn degenerate_inputs_stay_finite<F: SpaceFixture>(f: &F) {
        let s = f.space();
        let tol = f.tol();
        for (a, b) in f.degenerate_pairs() {
            let d = s.distance(a, b);
            assert!(
                d.is_finite() && d >= 0.0,
                "d = {d} at {:?} {:?}",
                f.point_components(a),
                f.point_components(b)
            );
            let v = s.log(a, b);
            assert_vector_is_finite(f, v, "log");
            assert_point_is_finite(f, s.exp(a, v), "exp(a, log(a, b))");
            for w in f.tangents(a) {
                let transported = s.parallel_transport(a, b, w);
                assert_vector_is_finite(f, transported, "parallel_transport");

                if !metric_is_defined(f, a, w) || !metric_is_defined(f, b, transported) {
                    continue;
                }
                let before = metric_norm(f, a, w);
                let after = metric_norm(f, b, transported);
                assert!(
                    agrees(after, before, tol.degenerate),
                    "transport changed the norm from {before} to {after} at {:?} {:?} {:?}",
                    f.point_components(a),
                    f.point_components(b),
                    f.vector_components(w)
                );
            }
        }
    }

    pub fn isometries_of_degenerate_inputs_stay_finite<F: IsometryFixture>(f: &F)
    where
        F::S: IsometryGroup<Iso = F::Iso>,
    {
        let s = f.space();
        let isos = f.isos();
        for (a, _) in f.degenerate_pairs() {
            for &g in &isos {
                assert_point_is_finite(f, s.iso_apply(g, a), "iso_apply");
                for w in f.tangents(a) {
                    assert_vector_is_finite(f, s.iso_transport(g, a, w), "iso_transport");
                }
            }
        }
    }

    fn worst_geodesic_tangent_ratio<F: SpaceFixture>(f: &F) -> (f32, F::Point, F::Point) {
        let s = f.space();
        let tol = f.tol();
        let points = f.points();
        let mut worst = (0.0, points[0], points[0]);
        for &a in &points {
            for &b in &points {
                let reverse = s.log(b, a);
                let forward = scaled(f, reverse, -1.0);
                let transported = s.parallel_transport(a, b, s.log(a, b));
                let residual = metric_residual(f, b, transported, forward);
                let ratio = residual / (tol.vector * metric_norm(f, b, forward).max(1.0));

                if ratio.is_nan() || ratio > worst.0 {
                    worst = (ratio, a, b);
                }
            }
        }
        worst
    }

    fn pole_ladder<F: SpaceFixture>(
        f: &F,
        s: &F::S,
        a: F::Point,
        b: F::Point,
        v: F::Vector,
    ) -> F::Vector {
        let midpoint = s.exp(a, scaled(f, s.log(a, b), 0.5));
        let mirrored = s.exp(midpoint, scaled(f, s.log(midpoint, s.exp(a, v)), -1.0));
        scaled(f, s.log(b, mirrored), -1.0)
    }

    fn scaled<F: SpaceFixture>(f: &F, v: F::Vector, s: f32) -> F::Vector {
        f.combine(v, s, v, 0.0)
    }

    // do Carmo, Differential Geometry of Curves and Surfaces, 1976, §1.4.
    fn metric_orthonormal_pair<F: SpaceFixture>(f: &F, at: F::Point) -> (F::Vector, F::Vector) {
        let tangents = f.tangents(at);
        assert!(
            tangents.len() >= 2,
            "the angle-excess item needs two independent tangents"
        );
        let n1 = metric_norm(f, at, tangents[0]);
        assert!(n1 > 0.0, "the first sampled tangent has zero metric norm");
        let e1 = f.combine(tangents[0], 1.0 / n1, tangents[0], 0.0);

        let raw = tangents[1];
        let projection = f.inner(at, raw, e1);
        let orthogonal = f.combine(raw, 1.0, e1, -projection);
        let n2 = metric_norm(f, at, orthogonal);
        assert!(
            n2 > 0.0,
            "the first two sampled tangents are metrically parallel"
        );
        (e1, f.combine(orthogonal, 1.0 / n2, orthogonal, 0.0))
    }

    fn assert_point_agrees<F: SpaceFixture>(
        f: &F,
        s: &F::S,
        got: F::Point,
        want: F::Point,
        tol: f32,
        what: &str,
    ) {
        let residual = s.distance(got, want);
        assert!(
            residual >= 0.0 && residual <= tol * chart_magnitude(f, want).max(1.0),
            "{what} is off by {residual}: {:?} vs {:?}",
            f.point_components(got),
            f.point_components(want)
        );
    }

    fn metric_is_defined<F: SpaceFixture>(f: &F, at: F::Point, v: F::Vector) -> bool {
        f.inner(at, v, v).is_finite()
    }

    fn assert_point_is_finite<F: SpaceFixture>(f: &F, p: F::Point, what: &str) {
        let c = f.point_components(p);
        assert!(c.iter().all(|x| x.is_finite()), "{what} returned {c:?}");
    }

    fn assert_vector_is_finite<F: SpaceFixture>(f: &F, v: F::Vector, what: &str) {
        let c = f.vector_components(v);
        assert!(c.iter().all(|x| x.is_finite()), "{what} returned {c:?}");
    }
}

macro_rules! conformance_tests {
    ($fixture:expr; $($invariant:ident),+ $(,)?) => {
        $(
            #[test]
            fn $invariant() {
                invariants::$invariant(&$fixture);
            }
        )+
    };
}

macro_rules! conformance_suite {
    ($suite:ident, $fixture:expr) => {
        conformance_suite!($suite, $fixture;
            log_magnitude_equals_geodesic_distance,
            parallel_transport_carries_a_geodesic_tangent_along_its_own_geodesic,
            parallel_transport_matches_the_one_its_own_geodesics_imply,
            geodesic_triangle_angle_excess_matches_gauss_bonnet,
        );
    };
    ($suite:ident, $fixture:expr; $($extra:ident),* $(,)?) => {
        mod $suite {
            use super::*;

            conformance_tests!($fixture;
                distance_is_symmetric_and_zero_on_the_diagonal,
                distance_is_finite_and_nonnegative,
                distance_satisfies_the_triangle_inequality,
                exp_inverts_log_on_sampled_pairs,
                log_inverts_exp_on_sampled_tangents,
                exp_advances_by_the_metric_norm_of_its_tangent,
                parallel_transport_preserves_the_metric_norm,
                parallel_transport_is_linear_in_the_transported_vector,
                parallel_transport_along_one_segment_matches_parallel_transport,
                degenerate_inputs_stay_finite,
                $($extra,)*
            );
        }
    };
}

macro_rules! isometry_conformance_suite {
    ($suite:ident, $fixture:expr) => {
        mod $suite {
            use super::*;

            conformance_tests!($fixture;
                distance_is_invariant_under_isometry,
                iso_identity_is_neutral,
                iso_inverse_is_two_sided,
                iso_compose_is_associative,
                iso_compose_matches_sequential_apply,
                iso_transport_is_the_differential_of_iso_apply,
                iso_transport_preserves_the_metric_norm,
                isometries_of_degenerate_inputs_stay_finite,
            );
        }
    };
}

struct EuclideanR2Fixture;

impl SpaceFixture for EuclideanR2Fixture {
    type Point = Vec2;
    type Vector = Vec2;
    type S = EuclideanR2;

    fn space(&self) -> EuclideanR2 {
        EuclideanR2
    }

    fn points(&self) -> Vec<Vec2> {
        let mut rng = Xorshift32::new(0x00E2_0F1A);
        (0..6)
            .map(|_| Vec2::new(rng.signed_unit(), rng.signed_unit()) * 1.5)
            .collect()
    }

    fn tangents(&self, _at: Vec2) -> Vec<Vec2> {
        let mut rng = Xorshift32::new(0x00E2_7A46);
        (0..3)
            .map(|_| Vec2::new(rng.signed_unit(), rng.signed_unit()) * 0.4)
            .collect()
    }

    fn inner(&self, _at: Vec2, u: Vec2, v: Vec2) -> f32 {
        u.dot(v)
    }

    fn combine(&self, u: Vec2, s: f32, v: Vec2, t: f32) -> Vec2 {
        u * s + v * t
    }

    fn degenerate_pairs(&self) -> Vec<(Vec2, Vec2)> {
        let far = Vec2::new(1.0e7, -3.0e6);
        vec![
            (Vec2::ZERO, Vec2::ZERO),
            (far, far),
            (Vec2::ZERO, far),
            (Vec2::new(1.0e-30, 0.0), Vec2::ZERO),
        ]
    }

    fn curvature(&self) -> Option<f32> {
        Some(0.0)
    }

    fn tol(&self) -> Tol {
        Tol {
            point: 1e-6,
            vector: 1e-6,
            scalar: 1e-6,
            degenerate: 1e-6,
        }
    }

    fn point_components(&self, p: Vec2) -> [f32; 4] {
        [p.x, p.y, 0.0, 0.0]
    }

    fn vector_components(&self, v: Vec2) -> [f32; 4] {
        [v.x, v.y, 0.0, 0.0]
    }
}

impl IsometryFixture for EuclideanR2Fixture {
    type Iso = Iso2;

    fn isos(&self) -> Vec<Iso2> {
        vec![
            Iso2 {
                rotation: Bivector2(0.5).exp(),
                translation: Vec2::new(1.0, 0.0),
            },
            Iso2 {
                rotation: Bivector2(-0.9).exp(),
                translation: Vec2::new(0.0, 2.0),
            },
            Iso2::from_translation(Vec2::new(-0.7, 0.3)),
        ]
    }
}

struct EuclideanR3Fixture;

impl SpaceFixture for EuclideanR3Fixture {
    type Point = Vec3;
    type Vector = Vec3;
    type S = EuclideanR3;

    fn space(&self) -> EuclideanR3 {
        EuclideanR3
    }

    fn points(&self) -> Vec<Vec3> {
        let mut rng = Xorshift32::new(0x00E3_0F1A);
        (0..6)
            .map(|_| Vec3::new(rng.signed_unit(), rng.signed_unit(), rng.signed_unit()) * 1.5)
            .collect()
    }

    fn tangents(&self, _at: Vec3) -> Vec<Vec3> {
        let mut rng = Xorshift32::new(0x00E3_7A46);
        (0..3)
            .map(|_| Vec3::new(rng.signed_unit(), rng.signed_unit(), rng.signed_unit()) * 0.4)
            .collect()
    }

    fn inner(&self, _at: Vec3, u: Vec3, v: Vec3) -> f32 {
        u.dot(v)
    }

    fn combine(&self, u: Vec3, s: f32, v: Vec3, t: f32) -> Vec3 {
        u * s + v * t
    }

    fn degenerate_pairs(&self) -> Vec<(Vec3, Vec3)> {
        let far = Vec3::new(1.0e7, -3.0e6, 5.0e6);
        vec![
            (Vec3::ZERO, Vec3::ZERO),
            (far, far),
            (Vec3::ZERO, far),
            (Vec3::new(1.0e-30, 0.0, 0.0), Vec3::ZERO),
        ]
    }

    fn curvature(&self) -> Option<f32> {
        Some(0.0)
    }

    fn tol(&self) -> Tol {
        Tol {
            point: 1e-6,
            vector: 1e-6,
            scalar: 1e-6,
            degenerate: 1e-6,
        }
    }

    fn point_components(&self, p: Vec3) -> [f32; 4] {
        [p.x, p.y, p.z, 0.0]
    }

    fn vector_components(&self, v: Vec3) -> [f32; 4] {
        [v.x, v.y, v.z, 0.0]
    }
}

impl IsometryFixture for EuclideanR3Fixture {
    type Iso = Iso3;

    fn isos(&self) -> Vec<Iso3> {
        vec![
            Iso3 {
                rotation: Quat::from_rotation_z(0.4),
                translation: Vec3::new(1.0, 0.0, 0.0),
            },
            Iso3 {
                rotation: Quat::from_rotation_x(0.9),
                translation: Vec3::new(0.0, 2.0, -1.0),
            },
            Iso3 {
                rotation: Quat::from_rotation_y(-0.6),
                translation: Vec3::new(-0.5, 0.25, 3.0),
            },
        ]
    }
}

struct EuclideanR4Fixture;

impl SpaceFixture for EuclideanR4Fixture {
    type Point = Vec4;
    type Vector = Vec4;
    type S = EuclideanR4;

    fn space(&self) -> EuclideanR4 {
        EuclideanR4
    }

    fn points(&self) -> Vec<Vec4> {
        let mut rng = Xorshift32::new(0x00E4_0F1A);
        (0..6)
            .map(|_| {
                Vec4::new(
                    rng.signed_unit(),
                    rng.signed_unit(),
                    rng.signed_unit(),
                    rng.signed_unit(),
                ) * 1.5
            })
            .collect()
    }

    fn tangents(&self, _at: Vec4) -> Vec<Vec4> {
        let mut rng = Xorshift32::new(0x00E4_7A46);
        (0..3)
            .map(|_| {
                Vec4::new(
                    rng.signed_unit(),
                    rng.signed_unit(),
                    rng.signed_unit(),
                    rng.signed_unit(),
                ) * 0.4
            })
            .collect()
    }

    fn inner(&self, _at: Vec4, u: Vec4, v: Vec4) -> f32 {
        u.dot(v)
    }

    fn combine(&self, u: Vec4, s: f32, v: Vec4, t: f32) -> Vec4 {
        u * s + v * t
    }

    fn degenerate_pairs(&self) -> Vec<(Vec4, Vec4)> {
        let far = Vec4::new(1.0e7, -3.0e6, 5.0e6, 2.0e6);
        vec![
            (Vec4::ZERO, Vec4::ZERO),
            (far, far),
            (Vec4::ZERO, far),
            (Vec4::new(1.0e-30, 0.0, 0.0, 0.0), Vec4::ZERO),
        ]
    }

    fn curvature(&self) -> Option<f32> {
        Some(0.0)
    }

    fn tol(&self) -> Tol {
        Tol {
            point: 1e-6,
            vector: 1e-6,
            scalar: 1e-6,
            degenerate: 1e-6,
        }
    }

    fn point_components(&self, p: Vec4) -> [f32; 4] {
        p.to_array()
    }

    fn vector_components(&self, v: Vec4) -> [f32; 4] {
        v.to_array()
    }
}

impl IsometryFixture for EuclideanR4Fixture {
    type Iso = Iso4Flat;

    fn isos(&self) -> Vec<Iso4Flat> {
        vec![
            Iso4Flat {
                rotation: Bivector4::new(0.4, 0.0, 0.0, 0.0, 0.0, 0.2).exp(),
                translation: Vec4::new(1.0, 0.0, 0.0, 0.0),
            },
            Iso4Flat {
                rotation: Bivector4::new(0.0, 0.0, 0.0, 0.9, 0.0, 0.0).exp(),
                translation: Vec4::new(0.0, 2.0, -1.0, 0.5),
            },
            Iso4Flat {
                rotation: Bivector4::new(0.3, 0.1, -0.2, 0.4, 0.0, 0.15).exp(),
                translation: Vec4::new(-0.5, 0.25, 3.0, -2.0),
            },
        ]
    }
}

struct HyperbolicH3Fixture;

impl SpaceFixture for HyperbolicH3Fixture {
    type Point = Vec3;
    type Vector = Vec3;
    type S = HyperbolicH3;

    fn space(&self) -> HyperbolicH3 {
        HyperbolicH3
    }

    fn points(&self) -> Vec<Vec3> {
        ball_samples(0x0083_0F1A, 5, 0.4)
    }

    fn tangents(&self, _at: Vec3) -> Vec<Vec3> {
        let mut rng = Xorshift32::new(0x0083_7A46);
        (0..3)
            .map(|_| Vec3::new(rng.signed_unit(), rng.signed_unit(), rng.signed_unit()) * 0.06)
            .collect()
    }

    fn inner(&self, at: Vec3, u: Vec3, v: Vec3) -> f32 {
        HyperbolicH3.conformal_factor(at) * u.dot(v)
    }

    fn combine(&self, u: Vec3, s: f32, v: Vec3, t: f32) -> Vec3 {
        u * s + v * t
    }

    fn degenerate_pairs(&self) -> Vec<(Vec3, Vec3)> {
        let interior = Vec3::new(0.3, 0.0, 0.0);

        let near_boundary = Vec3::new(0.85, 0.0, 0.0);

        let off_axis_outside = Vec3::new(1.2, 0.9, 1.4);
        vec![
            (interior, interior),
            (interior, Vec3::new(1.0, 0.0, 0.0)),
            (interior, Vec3::new(2.0, 0.0, 0.0)),
            (Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, -1.0, 0.0)),
            (near_boundary, -near_boundary),
            (off_axis_outside, off_axis_outside),
        ]
    }

    fn curvature(&self) -> Option<f32> {
        Some(-1.0)
    }

    fn tol(&self) -> Tol {
        Tol {
            point: 1e-4,
            vector: 1e-4,
            scalar: 1e-4,

            degenerate: 1e-6,
        }
    }

    fn point_components(&self, p: Vec3) -> [f32; 4] {
        [p.x, p.y, p.z, 0.0]
    }

    fn vector_components(&self, v: Vec3) -> [f32; 4] {
        [v.x, v.y, v.z, 0.0]
    }
}

impl IsometryFixture for HyperbolicH3Fixture {
    type Iso = Iso3H;

    fn isos(&self) -> Vec<Iso3H> {
        vec![
            Iso3H::from_translation(Vec3::new(0.15, 0.0, 0.0)),
            Iso3H::from_rotation(Quat::from_rotation_z(0.4)),
            Iso3H::from_translation(Vec3::new(-0.05, 0.2, 0.1)),
        ]
    }
}

struct SphericalS3Fixture;

impl SphericalS3Fixture {
    fn lift(p: Vec3, v: Vec3) -> Vec4 {
        let w = (1.0 - p.length_squared()).sqrt();
        Vec4::new(v.x, v.y, v.z, -v.dot(p) / w)
    }
}

impl SpaceFixture for SphericalS3Fixture {
    type Point = Vec3;
    type Vector = Vec3;
    type S = SphericalS3;

    fn space(&self) -> SphericalS3 {
        SphericalS3
    }

    fn points(&self) -> Vec<Vec3> {
        ball_samples(0x0053_0F1A, 5, 0.4)
    }

    fn tangents(&self, _at: Vec3) -> Vec<Vec3> {
        let mut rng = Xorshift32::new(0x0053_7A46);
        (0..3)
            .map(|_| Vec3::new(rng.signed_unit(), rng.signed_unit(), rng.signed_unit()) * 0.15)
            .collect()
    }

    fn inner(&self, at: Vec3, u: Vec3, v: Vec3) -> f32 {
        Self::lift(at, u).dot(Self::lift(at, v))
    }

    fn combine(&self, u: Vec3, s: f32, v: Vec3, t: f32) -> Vec3 {
        u * s + v * t
    }

    fn degenerate_pairs(&self) -> Vec<(Vec3, Vec3)> {
        let interior = Vec3::new(0.3, 0.0, 0.0);

        let near_equator = Vec3::new((1.0 - 2.0e-6_f32).sqrt(), 1e-3, 0.0);
        vec![
            (interior, interior),
            (interior, Vec3::new(1.0, 0.0, 0.0)),
            (interior, Vec3::new(2.0, 0.0, 0.0)),
            (
                near_equator,
                Vec3::new(-near_equator.x, near_equator.y, 0.0),
            ),
        ]
    }

    fn curvature(&self) -> Option<f32> {
        Some(1.0)
    }

    fn tol(&self) -> Tol {
        Tol {
            point: 1e-5,
            vector: 1e-5,
            scalar: 1e-5,
            degenerate: 1e-3,
        }
    }

    fn point_components(&self, p: Vec3) -> [f32; 4] {
        [p.x, p.y, p.z, 0.0]
    }

    fn vector_components(&self, v: Vec3) -> [f32; 4] {
        [v.x, v.y, v.z, 0.0]
    }
}

impl IsometryFixture for SphericalS3Fixture {
    type Iso = Iso4;

    fn isos(&self) -> Vec<Iso4> {
        vec![
            Iso4::from_translation(Vec3::new(0.15, 0.0, 0.0)),
            Iso4::from_rotation(Quat::from_rotation_z(0.4)),
            Iso4::from_translation(Vec3::new(-0.05, 0.2, 0.1)),
        ]
    }
}

struct SphericalS3EmbeddedFixture;

impl SpaceFixture for SphericalS3EmbeddedFixture {
    type Point = Vec4;
    type Vector = Vec4;
    type S = SphericalS3Embedded;

    fn space(&self) -> SphericalS3Embedded {
        SphericalS3Embedded
    }

    fn points(&self) -> Vec<Vec4> {
        let mut rng = Xorshift32::new(0x0054_0F1A);
        (0..5)
            .map(|_| {
                (Vec4::W
                    + Vec4::new(
                        rng.signed_unit(),
                        rng.signed_unit(),
                        rng.signed_unit(),
                        rng.signed_unit(),
                    ) * 0.4)
                    .normalize()
            })
            .collect()
    }

    fn tangents(&self, at: Vec4) -> Vec<Vec4> {
        let mut rng = Xorshift32::new(0x0054_7A46);
        (0..3)
            .map(|_| {
                let raw = Vec4::new(
                    rng.signed_unit(),
                    rng.signed_unit(),
                    rng.signed_unit(),
                    rng.signed_unit(),
                );

                (raw - raw.dot(at) * at) * 0.2
            })
            .collect()
    }

    fn inner(&self, _at: Vec4, u: Vec4, v: Vec4) -> f32 {
        u.dot(v)
    }

    fn combine(&self, u: Vec4, s: f32, v: Vec4, t: f32) -> Vec4 {
        u * s + v * t
    }

    fn degenerate_pairs(&self) -> Vec<(Vec4, Vec4)> {
        let p = Vec4::new(0.1, -0.2, 0.3, 0.9).normalize();
        let omega = std::f32::consts::PI - 1e-3;
        vec![
            (p, p),
            (Vec4::X, -Vec4::X),
            (
                Vec4::X,
                Vec4::new(omega.cos(), omega.sin(), 0.0, 0.0).normalize(),
            ),
            (p, (p + Vec4::new(1e-9, 0.0, -1e-9, 0.0)).normalize()),
        ]
    }

    fn curvature(&self) -> Option<f32> {
        Some(1.0)
    }

    fn tol(&self) -> Tol {
        Tol {
            point: 1e-5,
            vector: 1e-5,
            scalar: 1e-5,
            degenerate: 1e-3,
        }
    }

    fn point_components(&self, p: Vec4) -> [f32; 4] {
        p.to_array()
    }

    fn vector_components(&self, v: Vec4) -> [f32; 4] {
        v.to_array()
    }
}

impl IsometryFixture for SphericalS3EmbeddedFixture {
    type Iso = Iso4;

    fn isos(&self) -> Vec<Iso4> {
        vec![
            Iso4::from_translation(Vec3::new(0.3, 0.1, -0.2)),
            Iso4::from_rotation(Quat::from_rotation_z(0.4)),
            Iso4::from_translation(Vec3::new(-0.05, 0.2, 0.1)),
        ]
    }
}

// Press et al., Numerical Recipes, 3rd ed., 2007, §18.1.
struct BlendedSpaceFixture;

impl SpaceFixture for BlendedSpaceFixture {
    type Point = Vec3;
    type Vector = Vec3;
    type S = BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>;

    fn space(&self) -> Self::S {
        BlendedSpace::new(
            EuclideanR3,
            HyperbolicH3,
            LinearBlendX::new(-0.15, 0.15).unwrap(),
        )
    }

    fn points(&self) -> Vec<Vec3> {
        let mut rng = Xorshift32::new(0x00B1_0F1A);
        (0..6)
            .map(|i| {
                let x = -0.32 + 0.128 * i as f32;
                Vec3::new(x, rng.signed_unit() * 0.08, rng.signed_unit() * 0.08)
            })
            .collect()
    }

    fn tangents(&self, _at: Vec3) -> Vec<Vec3> {
        let mut rng = Xorshift32::new(0x00B1_7A46);
        (0..2)
            .map(|_| Vec3::new(rng.signed_unit(), rng.signed_unit(), rng.signed_unit()) * 0.05)
            .collect()
    }

    fn inner(&self, at: Vec3, u: Vec3, v: Vec3) -> f32 {
        self.space().conformal_factor(at) * u.dot(v)
    }

    fn combine(&self, u: Vec3, s: f32, v: Vec3, t: f32) -> Vec3 {
        u * s + v * t
    }

    fn degenerate_pairs(&self) -> Vec<(Vec3, Vec3)> {
        let interior = Vec3::new(0.1, 0.0, 0.0);
        vec![
            (interior, interior),
            (Vec3::new(-0.32, 0.0, 0.0), Vec3::new(0.32, 0.0, 0.0)),
            (Vec3::new(0.3, 0.08, 0.08), Vec3::new(0.3, 0.08, 0.08)),
        ]
    }

    fn curvature(&self) -> Option<f32> {
        None
    }

    fn tol(&self) -> Tol {
        Tol {
            point: 5e-4,
            vector: 5e-3,
            scalar: 5e-3,
            degenerate: 5e-3,
        }
    }

    fn point_components(&self, p: Vec3) -> [f32; 4] {
        [p.x, p.y, p.z, 0.0]
    }

    fn vector_components(&self, v: Vec3) -> [f32; 4] {
        [v.x, v.y, v.z, 0.0]
    }
}

struct LensSpaceFixture(LensSpace);

impl LensSpaceFixture {
    fn plane_rotations(xy: f32, zw: f32) -> Iso4 {
        let (sin_xy, cos_xy) = xy.sin_cos();
        let (sin_zw, cos_zw) = zw.sin_cos();
        Iso4 {
            matrix: Mat4::from_cols(
                Vec4::new(cos_xy, sin_xy, 0.0, 0.0),
                Vec4::new(-sin_xy, cos_xy, 0.0, 0.0),
                Vec4::new(0.0, 0.0, cos_zw, sin_zw),
                Vec4::new(0.0, 0.0, -sin_zw, cos_zw),
            ),
        }
    }

    fn conjugation() -> Iso4 {
        Iso4 {
            matrix: Mat4::from_cols(
                Vec4::X,
                Vec4::new(0.0, -1.0, 0.0, 0.0),
                Vec4::Z,
                Vec4::new(0.0, 0.0, 0.0, -1.0),
            ),
        }
    }
}

impl SpaceFixture for LensSpaceFixture {
    type Point = Vec4;
    type Vector = Vec4;
    type S = LensSpace;

    fn space(&self) -> LensSpace {
        self.0
    }

    fn points(&self) -> Vec<Vec4> {
        let mut rng = Xorshift32::new(0x004C_0F1A);
        let spread = 0.25 * self.0.injectivity_radius();
        (0..5)
            .map(|_| {
                (Vec4::X
                    + Vec4::new(
                        rng.signed_unit(),
                        rng.signed_unit(),
                        rng.signed_unit(),
                        rng.signed_unit(),
                    ) * spread)
                    .normalize()
            })
            .collect()
    }

    fn tangents(&self, at: Vec4) -> Vec<Vec4> {
        let mut rng = Xorshift32::new(0x004C_7A46);
        let reach = 0.4 * self.0.injectivity_radius();
        (0..3)
            .map(|_| {
                let raw = Vec4::new(
                    rng.signed_unit(),
                    rng.signed_unit(),
                    rng.signed_unit(),
                    rng.signed_unit(),
                );

                (raw - raw.dot(at) * at) * reach
            })
            .collect()
    }

    fn inner(&self, _at: Vec4, u: Vec4, v: Vec4) -> f32 {
        u.dot(v)
    }

    fn combine(&self, u: Vec4, s: f32, v: Vec4, t: f32) -> Vec4 {
        u * s + v * t
    }

    fn degenerate_pairs(&self) -> Vec<(Vec4, Vec4)> {
        let p = self.0.p() as f32;
        let a = Vec4::new(0.1, -0.2, 0.3, 0.9).normalize();

        let (sin, cos) = (std::f32::consts::PI / p).sin_cos();
        let cut = Vec4::new(cos, sin, 0.0, 0.0);
        vec![
            (a, a),
            (Vec4::X, -Vec4::X),
            (Vec4::X, cut),
            (Vec4::X, Vec4::Z),
        ]
    }

    fn curvature(&self) -> Option<f32> {
        Some(1.0)
    }

    fn tol(&self) -> Tol {
        Tol {
            point: 1e-5,
            vector: 1e-5,
            scalar: 1e-5,
            degenerate: 1e-3,
        }
    }

    fn point_components(&self, p: Vec4) -> [f32; 4] {
        p.to_array()
    }

    fn vector_components(&self, v: Vec4) -> [f32; 4] {
        v.to_array()
    }
}

impl IsometryFixture for LensSpaceFixture {
    type Iso = Iso4;

    fn isos(&self) -> Vec<Iso4> {
        let space = self.space();
        vec![
            LensSpaceFixture::plane_rotations(0.4, -0.7),
            LensSpaceFixture::conjugation(),
            space.iso_compose(
                LensSpaceFixture::plane_rotations(-0.3, 0.9),
                LensSpaceFixture::conjugation(),
            ),
        ]
    }
}

conformance_suite!(euclidean_r2, EuclideanR2Fixture);
conformance_suite!(euclidean_r3, EuclideanR3Fixture);
conformance_suite!(euclidean_r4, EuclideanR4Fixture);
conformance_suite!(hyperbolic_h3, HyperbolicH3Fixture);
conformance_suite!(spherical_s3, SphericalS3Fixture);
conformance_suite!(spherical_s3_embedded, SphericalS3EmbeddedFixture);
conformance_suite!(blended_space, BlendedSpaceFixture;);

conformance_suite!(lens_space_l5_2, LensSpaceFixture(LensSpace::new(5, 2)));
conformance_suite!(lens_space_rp3, LensSpaceFixture(LensSpace::new(2, 1)));

isometry_conformance_suite!(euclidean_r2_isometries, EuclideanR2Fixture);
isometry_conformance_suite!(euclidean_r3_isometries, EuclideanR3Fixture);
isometry_conformance_suite!(euclidean_r4_isometries, EuclideanR4Fixture);
isometry_conformance_suite!(hyperbolic_h3_isometries, HyperbolicH3Fixture);
isometry_conformance_suite!(spherical_s3_isometries, SphericalS3Fixture);
isometry_conformance_suite!(spherical_s3_embedded_isometries, SphericalS3EmbeddedFixture);
isometry_conformance_suite!(
    lens_space_l5_2_isometries,
    LensSpaceFixture(LensSpace::new(5, 2))
);
isometry_conformance_suite!(
    lens_space_rp3_isometries,
    LensSpaceFixture(LensSpace::new(2, 1))
);
