//! Isometries act on the cover; wrap_to_domain returns a representative and its deck element.
//! LensSpace uses the z1 wedge, which does not select a unique representative on z1 = 0.

use std::f32::consts::{PI, TAU};

use glam::{BVec3, Mat4, Quat, Vec3, Vec4};

use crate::euclidean::Iso3;
use crate::space::{IsometryGroup, Space};
use crate::spherical::Iso4;
use crate::spherical_embedded::SphericalS3Embedded;

pub trait CoveringSpace: Space {
    type CoverIso: Copy + Send + Sync + 'static;

    fn cover_identity(&self) -> Self::CoverIso;

    fn cover_compose(&self, a: Self::CoverIso, b: Self::CoverIso) -> Self::CoverIso;

    fn cover_inverse(&self, a: Self::CoverIso) -> Self::CoverIso;

    fn cover_apply(&self, iso: Self::CoverIso, p: Self::Point) -> Self::Point;

    fn cover_transport(
        &self,
        iso: Self::CoverIso,
        at: Self::Point,
        v: Self::Vector,
    ) -> Self::Vector;
}

pub trait QuotientSpace: CoveringSpace {
    /// Iteration order is part of the contract.
    fn face_pairings(&self) -> impl Iterator<Item = Self::CoverIso>;

    fn in_fundamental_domain(&self, p: Self::Point) -> bool;

    fn wrap_to_domain(&self, p: Self::Point) -> (Self::Point, Self::CoverIso);
}

// Conway and Sloane, Sphere Packings, Lattices and Groups, 1988, ch. 4 §1.
/// Rectangular torus with canonical representatives in `[-cᵢ/2, cᵢ/2)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlatTorus3 {
    cell: Vec3,
}

impl FlatTorus3 {
    /// Panics unless every side is finite and positive.
    pub fn new(cell: Vec3) -> Self {
        assert!(
            cell.is_finite() && cell.cmpgt(Vec3::ZERO).all(),
            "FlatTorus3 cell must be finite and positive, got {cell:?}"
        );
        Self { cell }
    }

    pub fn cube(side: f32) -> Self {
        Self::new(Vec3::splat(side))
    }

    pub fn cell(&self) -> Vec3 {
        self.cell
    }

    pub fn injectivity_radius(&self) -> f32 {
        0.5 * self.cell.min_element()
    }

    fn lattice_index(&self, p: Vec3) -> Vec3 {
        let index = (p / self.cell + Vec3::splat(0.5)).floor();
        let residual = p - self.cell * index;
        let half = 0.5 * self.cell;
        index - unit_where(residual.cmplt(-half)) + unit_where(residual.cmpge(half))
    }

    fn wrap(&self, p: Vec3) -> Vec3 {
        p - self.cell * self.lattice_index(p)
    }
}

fn unit_where(mask: BVec3) -> Vec3 {
    Vec3::select(mask, Vec3::ONE, Vec3::ZERO)
}

impl Space for FlatTorus3 {
    type Point = Vec3;
    type Vector = Vec3;
    type Frame = Quat;

    fn frame_at(&self, _at: Vec3) -> Quat {
        Quat::IDENTITY
    }

    fn distance(&self, a: Vec3, b: Vec3) -> f32 {
        self.wrap(a - b).length()
    }

    fn exp(&self, at: Vec3, v: Vec3) -> Vec3 {
        self.wrap(at + v)
    }

    fn log(&self, from: Vec3, to: Vec3) -> Vec3 {
        self.wrap(to - from)
    }

    fn parallel_transport(&self, _from: Vec3, _to: Vec3, v: Vec3) -> Vec3 {
        v
    }

    fn is_chart_flat(&self) -> bool {
        false
    }

    fn chart_envelope(&self) -> f32 {
        f32::INFINITY
    }

    fn valid_point(&self, p: Vec3) -> bool {
        p.is_finite()
    }
}

impl CoveringSpace for FlatTorus3 {
    type CoverIso = Iso3;

    fn cover_identity(&self) -> Iso3 {
        Iso3::IDENTITY
    }

    fn cover_compose(&self, a: Iso3, b: Iso3) -> Iso3 {
        Iso3 {
            rotation: a.rotation * b.rotation,
            translation: a.rotation * b.translation + a.translation,
        }
    }

    fn cover_inverse(&self, a: Iso3) -> Iso3 {
        let inverse_rotation = a.rotation.inverse();
        Iso3 {
            rotation: inverse_rotation,
            translation: inverse_rotation * (-a.translation),
        }
    }

    fn cover_apply(&self, iso: Iso3, p: Vec3) -> Vec3 {
        iso.rotation * p + iso.translation
    }

    fn cover_transport(&self, iso: Iso3, _at: Vec3, v: Vec3) -> Vec3 {
        iso.rotation * v
    }
}

impl QuotientSpace for FlatTorus3 {
    fn face_pairings(&self) -> impl Iterator<Item = Iso3> {
        [
            Iso3::from_translation(Vec3::new(self.cell.x, 0.0, 0.0)),
            Iso3::from_translation(Vec3::new(0.0, self.cell.y, 0.0)),
            Iso3::from_translation(Vec3::new(0.0, 0.0, self.cell.z)),
        ]
        .into_iter()
    }

    fn in_fundamental_domain(&self, p: Vec3) -> bool {
        let half = 0.5 * self.cell;
        p.cmpge(-half).all() && p.cmplt(half).all()
    }

    fn wrap_to_domain(&self, p: Vec3) -> (Vec3, Iso3) {
        let translation = -(self.cell * self.lattice_index(p));

        (p + translation, Iso3::from_translation(translation))
    }
}

const COVER: SphericalS3Embedded = SphericalS3Embedded;

// Rolfsen, Knots and Links, 1976, §9.B.
// Hatcher, Algebraic Topology, 2002, Example 2.43.
/// Lens space with `z₁ = x + iy`, `z₂ = z + iw` and deck action `(z₁, z₂) ↦ (e^(2πi/p) z₁, e^(2πiq/p) z₂)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LensSpace {
    p: u32,
    q: u32,
}

impl LensSpace {
    /// Panics unless `p ≥ 2` and `gcd(p, q) = 1`; `q` is reduced mod `p`.
    pub fn new(p: u32, q: u32) -> Self {
        assert!(
            p >= 2 && p <= i32::MAX as u32,
            "LensSpace needs p >= 2 and p <= i32::MAX, got p = {p}"
        );
        let q = q % p;
        assert!(
            gcd(p, q) == 1,
            "LensSpace needs gcd(p, q) = 1 for a free action, got p = {p}, q = {q}"
        );
        Self { p, q }
    }

    pub fn p(&self) -> u32 {
        self.p
    }

    /// Twist of the gluing, reduced mod `p`.
    pub fn q(&self) -> u32 {
        self.q
    }

    pub fn injectivity_radius(&self) -> f32 {
        PI / self.p as f32
    }

    /// `g^power`, with the power reduced mod `p` in integers so `deck(0)` is exact.
    pub fn deck(&self, power: i32) -> Iso4 {
        let (beta1, beta2) = self.deck_angles(power);
        let (sin1, cos1) = beta1.sin_cos();
        let (sin2, cos2) = beta2.sin_cos();
        Iso4 {
            matrix: Mat4::from_cols(
                Vec4::new(cos1, sin1, 0.0, 0.0),
                Vec4::new(-sin1, cos1, 0.0, 0.0),
                Vec4::new(0.0, 0.0, cos2, sin2),
                Vec4::new(0.0, 0.0, -sin2, cos2),
            ),
        }
    }

    fn deck_angles(&self, power: i32) -> (f32, f32) {
        let p = self.p as i64;
        let k = (power as i64).rem_euclid(p);
        let step = TAU / self.p as f32;
        (step * k as f32, step * ((k * self.q as i64) % p) as f32)
    }

    fn wedge_parameter(&self, x: Vec4) -> f32 {
        x.y.atan2(x.x) * (self.p as f32 / TAU) + 0.5
    }

    fn wedge_offset(&self, x: Vec4) -> i32 {
        -(self.wedge_parameter(x).floor() as i32)
    }

    fn nearest_lift(&self, from: Vec4, to: Vec4) -> (i32, Vec4, f32) {
        let mut best_lift = self.cover_apply(self.deck(0), to);
        let mut best_power = 0;
        let mut best_distance = COVER.distance(from, best_lift);
        for power in 1..self.p as i32 {
            let lift = self.cover_apply(self.deck(power), to);
            let distance = COVER.distance(from, lift);
            if distance < best_distance {
                best_lift = lift;
                best_power = power;
                best_distance = distance;
            }
        }
        (best_power, best_lift, best_distance)
    }
}

// Knuth, TAOCP vol. 2, 3rd ed., §4.5.2.
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

impl Space for LensSpace {
    type Point = Vec4;
    type Vector = Vec4;
    type Frame = Iso4;

    fn frame_at(&self, at: Vec4) -> Iso4 {
        Iso4::left_translation(at)
    }

    fn distance(&self, a: Vec4, b: Vec4) -> f32 {
        self.nearest_lift(a, b).2
    }

    fn exp(&self, at: Vec4, v: Vec4) -> Vec4 {
        self.wrap_to_domain(COVER.exp(at, v)).0
    }

    fn log(&self, from: Vec4, to: Vec4) -> Vec4 {
        let (_, lift, _) = self.nearest_lift(from, to);
        COVER.log(from, lift)
    }

    fn parallel_transport(&self, from: Vec4, to: Vec4, v: Vec4) -> Vec4 {
        let (power, lift, _) = self.nearest_lift(from, to);
        let carried = COVER.parallel_transport(from, lift, v);

        self.cover_transport(self.deck(-power), lift, carried)
    }

    fn chart_envelope(&self) -> f32 {
        COVER.chart_envelope()
    }

    fn valid_point(&self, p: Vec4) -> bool {
        COVER.valid_point(p)
    }
}

impl CoveringSpace for LensSpace {
    type CoverIso = Iso4;

    fn cover_identity(&self) -> Iso4 {
        COVER.iso_identity()
    }

    fn cover_compose(&self, a: Iso4, b: Iso4) -> Iso4 {
        COVER.iso_compose(a, b)
    }

    fn cover_inverse(&self, a: Iso4) -> Iso4 {
        COVER.iso_inverse(a)
    }

    fn cover_apply(&self, iso: Iso4, p: Vec4) -> Vec4 {
        COVER.iso_apply(iso, p)
    }

    fn cover_transport(&self, iso: Iso4, at: Vec4, v: Vec4) -> Vec4 {
        COVER.iso_transport(iso, at, v)
    }
}

impl QuotientSpace for LensSpace {
    fn face_pairings(&self) -> impl Iterator<Item = Iso4> {
        std::iter::once(self.deck(1))
    }

    fn in_fundamental_domain(&self, p: Vec4) -> bool {
        (0.0..1.0).contains(&self.wedge_parameter(p))
    }

    fn wrap_to_domain(&self, p: Vec4) -> (Vec4, Iso4) {
        let mut power = self.wedge_offset(p);
        let mut deck = self.deck(power);
        let mut wrapped = self.cover_apply(deck, p);

        let correction = self.wedge_offset(wrapped);
        if correction != 0 {
            power += correction;
            deck = self.deck(power);
            wrapped = self.cover_apply(deck, p);
        }
        (wrapped, deck)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const CELL: Vec3 = Vec3::new(2.0, 3.0, 1.5);

    fn torus() -> FlatTorus3 {
        FlatTorus3::new(CELL)
    }

    pub(super) struct Xorshift(pub(super) u32);

    impl Xorshift {
        pub(super) fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }

        pub(super) fn signed(&mut self, span: f32) -> f32 {
            let unit = self.next_u32() as f32 / u32::MAX as f32;
            (unit * 2.0 - 1.0) * span
        }

        fn point(&mut self, span: f32) -> Vec3 {
            Vec3::new(self.signed(span), self.signed(span), self.signed(span))
        }
    }

    fn boundary_lifts(cell: Vec3) -> Vec<Vec3> {
        let half = 0.5 * cell;
        let mut lifts = Vec::new();
        for axis in 0..3 {
            for sign in [-1.0_f32, 1.0] {
                let edge = sign * half[axis];
                for coordinate in [
                    edge,
                    f32::from_bits(edge.to_bits() - 1),
                    f32::from_bits(edge.to_bits() + 1),
                ] {
                    for copy in -3..=3 {
                        let mut p = Vec3::new(0.25, -0.5, 0.125);
                        p[axis] = coordinate + copy as f32 * cell[axis];
                        lifts.push(p);
                    }
                }
            }
        }
        lifts
    }

    fn deck_generators(t: &FlatTorus3) -> Vec<Iso3> {
        t.face_pairings()
            .flat_map(|g| [g, t.cover_inverse(g)])
            .collect()
    }

    fn word(t: &FlatTorus3, generators: &[Iso3], rng: &mut Xorshift) -> Iso3 {
        let length = (rng.next_u32() % 5) as usize;
        (0..length).fold(t.cover_identity(), |accumulated, _| {
            let pick = generators[(rng.next_u32() as usize) % generators.len()];
            t.cover_compose(accumulated, pick)
        })
    }

    #[test]
    fn wrap_picks_one_representative_per_orbit_and_returns_the_deck_element_that_reaches_it() {
        let t = torus();

        let mut rng = Xorshift(0x5eed_1234);
        let mut lifts = boundary_lifts(CELL);
        for _ in 0..200_000 {
            lifts.push(rng.point(40.0));
        }
        for p in lifts {
            let (q, g) = t.wrap_to_domain(p);
            assert!(
                t.in_fundamental_domain(q),
                "wrap({p:?}) = {q:?} escaped the domain of cell {CELL:?}"
            );
            assert_eq!(t.cover_apply(g, p), q, "deck element disagreed at {p:?}");
        }

        let interior = Vec3::new(0.3, -1.1, 0.4);
        assert_eq!(t.wrap_to_domain(interior), (interior, Iso3::IDENTITY));

        let mut rng = Xorshift(0x1337_beef);
        for _ in 0..50_000 {
            let p = rng.point(6.0);
            let (q, _) = t.wrap_to_domain(p);
            assert_eq!(t.in_fundamental_domain(p), p == q, "at {p:?}");
        }

        let generators = deck_generators(&t);
        let mut rng = Xorshift(0x00c0_ffee);
        for _ in 0..20_000 {
            let p = rng.point(5.0);
            let (q, _) = t.wrap_to_domain(p);
            let g = word(&t, &generators, &mut rng);
            let (q_translated, _) = t.wrap_to_domain(t.cover_apply(g, p));
            assert_relative_eq!(q_translated.x, q.x, epsilon = 1e-4);
            assert_relative_eq!(q_translated.y, q.y, epsilon = 1e-4);
            assert_relative_eq!(q_translated.z, q.z, epsilon = 1e-4);
        }
    }

    #[test]
    fn distance_is_the_minimum_over_the_lattice_and_a_metric_on_the_quotient() {
        let t = torus();
        let generators = deck_generators(&t);

        let mut rng = Xorshift(0xdead_10cc);
        for _ in 0..20_000 {
            let a = rng.point(4.0);
            let b = rng.point(4.0);
            let g = word(&t, &generators, &mut rng);

            let center = ((a - b) / CELL).round();
            let mut brute = f32::INFINITY;
            for i in -3..=3 {
                for j in -3..=3 {
                    for k in -3..=3 {
                        let offset = Vec3::new(i as f32, j as f32, k as f32);
                        brute = brute.min((a - b - CELL * (center + offset)).length());
                    }
                }
            }
            assert_relative_eq!(t.distance(a, b), brute, epsilon = 1e-5);

            assert_relative_eq!(t.distance(a, b), t.distance(b, a));
            assert_relative_eq!(
                t.distance(t.cover_apply(g, a), b),
                t.distance(a, b),
                epsilon = 1e-4
            );
            assert_relative_eq!(t.distance(a, t.cover_apply(g, a)), 0.0, epsilon = 1e-4);
        }

        let mut rng = Xorshift(0x7f4a_7c15);
        for _ in 0..50_000 {
            let a = rng.point(5.0);
            let b = rng.point(5.0);
            let c = rng.point(5.0);
            assert!(t.distance(a, c) <= t.distance(a, b) + t.distance(b, c) + 1e-4);
        }
    }

    #[test]
    fn a_geodesic_crossing_the_gluing_lands_where_the_face_pairing_says() {
        let t = torus();
        let half = 0.5 * CELL;
        let pairings: Vec<Iso3> = t.face_pairings().collect();

        for (axis, pairing) in pairings.iter().enumerate() {
            let mut start = Vec3::new(0.1, 0.9, -0.3);
            start[axis] = half[axis] - 0.05;
            let mut step = Vec3::ZERO;
            step[axis] = 0.2;

            let uncorrected = start + step;
            let (arrived, deck) = t.wrap_to_domain(uncorrected);

            let expected = t.cover_inverse(*pairing);
            assert_eq!(deck.translation, expected.translation, "axis {axis}");
            assert_eq!(t.cover_apply(expected, uncorrected), arrived);
            assert_eq!(t.exp(start, step), arrived);
        }
    }

    #[test]
    fn exp_and_log_invert_within_the_injectivity_radius() {
        let t = torus();
        let radius = t.injectivity_radius();
        assert_relative_eq!(radius, 0.75);

        let mut rng = Xorshift(0x6c07_8965);
        for _ in 0..20_000 {
            let from = t.wrap(rng.point(2.0));
            let direction = rng.point(1.0).normalize_or_zero();
            let v = direction * (0.95 * radius);
            let to = t.exp(from, v);
            let recovered = t.log(from, to);
            assert_relative_eq!(recovered.x, v.x, epsilon = 1e-5);
            assert_relative_eq!(recovered.y, v.y, epsilon = 1e-5);
            assert_relative_eq!(recovered.z, v.z, epsilon = 1e-5);
        }

        let short = t.log(Vec3::ZERO, t.exp(Vec3::ZERO, Vec3::new(0.0, 2.0, 0.0)));
        assert_relative_eq!(short.x, 0.0);
        assert_relative_eq!(short.y, -1.0);
        assert_relative_eq!(short.z, 0.0);
    }

    #[test]
    #[should_panic(expected = "must be finite and positive")]
    fn a_degenerate_cell_is_rejected_at_construction() {
        FlatTorus3::new(Vec3::new(1.0, 0.0, 1.0));
    }
}

#[cfg(test)]
mod lens_tests {
    use super::tests::Xorshift;
    use super::*;
    use approx::assert_relative_eq;

    const LENSES: &[(u32, u32)] = &[(2, 1), (3, 1), (5, 2), (7, 3), (8, 3)];

    fn unit4(rng: &mut Xorshift) -> Vec4 {
        let raw = Vec4::new(
            rng.signed(1.0),
            rng.signed(1.0),
            rng.signed(1.0),
            rng.signed(1.0),
        );
        if raw.length_squared() < 1e-6 {
            Vec4::X
        } else {
            raw.normalize()
        }
    }

    fn generator_by_hand(p: u32, q: u32) -> Mat4 {
        let step = TAU / p as f32;
        let (sin1, cos1) = step.sin_cos();
        let (sin2, cos2) = (step * q as f32).sin_cos();
        Mat4::from_cols(
            Vec4::new(cos1, sin1, 0.0, 0.0),
            Vec4::new(-sin1, cos1, 0.0, 0.0),
            Vec4::new(0.0, 0.0, cos2, sin2),
            Vec4::new(0.0, 0.0, -sin2, cos2),
        )
    }

    const WALL_MARGIN: f32 = 1e-4;

    fn wall_lifts(lens: &LensSpace) -> Vec<Vec4> {
        let mut lifts = Vec::new();
        for wedge in 0..lens.p() as i32 {
            let wall = (wedge as f32 + 0.5) * TAU / lens.p() as f32;
            for angle in [wall - WALL_MARGIN, wall + WALL_MARGIN] {
                let (sin, cos) = angle.sin_cos();
                lifts.push(Vec4::new(cos * 0.8, sin * 0.8, 0.5, 0.331_662_5).normalize());
            }
        }
        lifts
    }

    #[test]
    fn the_deck_group_is_cyclic_of_order_p_generated_by_the_face_pairing() {
        let mut rng = Xorshift(0x1e45_9ac1);
        for &(p, q) in LENSES {
            let lens = LensSpace::new(p, q);
            let generator = lens.face_pairings().next().expect("one face pairing");
            let samples: Vec<Vec4> = (0..32).map(|_| unit4(&mut rng)).collect();

            let mut accumulated = lens.cover_identity();
            for step in 1..=p {
                accumulated = lens.cover_compose(accumulated, generator);
                let moved = samples
                    .iter()
                    .map(|&x| SphericalS3Embedded.distance(lens.cover_apply(accumulated, x), x))
                    .fold(0.0f32, f32::max);
                if step == p {
                    assert!(
                        moved < 1e-5,
                        "L({p}, {q}): g^{step} moved a lift by {moved}, so the loop \
                         does not close"
                    );
                } else {
                    assert!(
                        moved > 0.1,
                        "L({p}, {q}): g^{step} already acts trivially, so the deck \
                         group has order {step}, not {p}"
                    );
                }
            }

            let by_hand = generator_by_hand(p, q);
            for power in -3 * p as i32..=3 * p as i32 {
                let mut composed = Mat4::IDENTITY;
                for _ in 0..power.rem_euclid(p as i32) {
                    composed = by_hand * composed;
                }
                for _ in 0..4 {
                    let x = unit4(&mut rng);
                    let want = (composed * x).normalize();
                    let got = lens.cover_apply(lens.deck(power), x);
                    assert!(
                        SphericalS3Embedded.distance(got, want) < 1e-4,
                        "L({p}, {q}): deck({power}) disagrees with the composed \
                         generator at {x:?}"
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "gcd(p, q) = 1")]
    fn an_action_with_fixed_points_is_rejected_at_construction() {
        LensSpace::new(6, 2);
    }

    #[test]
    #[should_panic(expected = "p >= 2")]
    fn the_trivial_quotient_is_rejected_at_construction() {
        LensSpace::new(1, 0);
    }

    #[test]
    fn wrap_picks_one_representative_per_orbit_and_returns_the_deck_element_that_reaches_it() {
        let mut domain_rng = Xorshift(0x5eed_1234);
        let mut fixpoint_rng = Xorshift(0x1337_beef);
        let mut orbit_rng = Xorshift(0x00c0_ffee);
        for &(p, q) in LENSES {
            let lens = LensSpace::new(p, q);

            let mut lifts = wall_lifts(&lens);
            lifts.extend((0..50_000).map(|_| unit4(&mut domain_rng)));
            for x in lifts {
                let (wrapped, deck) = lens.wrap_to_domain(x);
                assert!(
                    lens.in_fundamental_domain(wrapped),
                    "L({p}, {q}): wrap({x:?}) = {wrapped:?} escaped the domain"
                );
                assert_eq!(
                    lens.cover_apply(deck, x),
                    wrapped,
                    "L({p}, {q}): deck element disagreed at {x:?}"
                );
            }

            let mut lifts = wall_lifts(&lens);
            lifts.extend((0..20_000).map(|_| unit4(&mut fixpoint_rng)));
            for x in lifts {
                let (wrapped, deck) = lens.wrap_to_domain(x);
                assert_eq!(
                    lens.in_fundamental_domain(x),
                    deck.matrix == Mat4::IDENTITY,
                    "L({p}, {q}) at {x:?}"
                );
                if lens.in_fundamental_domain(x) {
                    assert!(SphericalS3Embedded.distance(wrapped, x) < 1e-6);
                }
            }

            for _ in 0..20_000 {
                let x = unit4(&mut orbit_rng);
                let power = (orbit_rng.next_u32() % (4 * p)) as i32 - 2 * p as i32;
                let (wrapped, _) = lens.wrap_to_domain(x);
                let (translated, _) = lens.wrap_to_domain(lens.cover_apply(lens.deck(power), x));
                assert!(
                    SphericalS3Embedded.distance(wrapped, translated) < 1e-3,
                    "L({p}, {q}): wrap of g^{power} x is a different lift at {x:?}"
                );
            }
        }
    }

    #[test]
    fn a_lift_on_a_wedge_wall_comes_back_on_a_wall() {
        for &(p, q) in LENSES {
            let lens = LensSpace::new(p, q);
            let half = PI / p as f32;
            for wedge in 0..p as i32 {
                let wall = (wedge as f32 + 0.5) * TAU / p as f32;
                for angle in [
                    wall,
                    f32::from_bits(wall.to_bits() - 1),
                    f32::from_bits(wall.to_bits() + 1),
                ] {
                    let (sin, cos) = angle.sin_cos();
                    let x = Vec4::new(cos * 0.8, sin * 0.8, 0.5, 0.331_662_5).normalize();
                    let (wrapped, deck) = lens.wrap_to_domain(x);
                    assert_eq!(lens.cover_apply(deck, x), wrapped, "L({p}, {q}) at {angle}");
                    assert!(
                        (wrapped.y.atan2(wrapped.x).abs() - half).abs() < 1e-5,
                        "L({p}, {q}): a wall lift came back at {} rad, not on a wall \
                         at {half}",
                        wrapped.y.atan2(wrapped.x)
                    );
                }
            }
        }
    }

    #[test]
    fn crossing_a_face_re_enters_through_the_opposite_face() {
        let mut rng = Xorshift(0x4d59_5f21);
        for &(p, q) in LENSES {
            let lens = LensSpace::new(p, q);
            let pairing = lens.face_pairings().next().expect("one face pairing");
            let half = PI / p as f32;
            let overshoot = 0.01;

            let (sin, cos) = (half + overshoot).sin_cos();
            let outside = Vec4::new(cos * 0.8, sin * 0.8, 0.5, 0.331_662_5).normalize();
            let (wrapped, deck) = lens.wrap_to_domain(outside);

            assert!(lens.in_fundamental_domain(wrapped));
            assert_relative_eq!(
                wrapped.y.atan2(wrapped.x),
                -half + overshoot,
                epsilon = 1e-5
            );

            let inverse = lens.cover_inverse(pairing);
            for _ in 0..64 {
                let x = unit4(&mut rng);
                assert!(
                    SphericalS3Embedded
                        .distance(lens.cover_apply(inverse, x), lens.cover_apply(deck, x))
                        < 1e-5,
                    "L({p}, {q}): the crossing's deck element is not the face pairing"
                );
            }
        }
    }

    #[test]
    fn distance_is_the_minimum_over_the_deck_group_and_a_metric_on_the_quotient() {
        let mut rng = Xorshift(0xdead_10cc);
        for &(p, q) in LENSES {
            let lens = LensSpace::new(p, q);
            let generator = generator_by_hand(p, q);
            for _ in 0..5_000 {
                let a = unit4(&mut rng);
                let b = unit4(&mut rng);
                let c = unit4(&mut rng);
                let power = (rng.next_u32() % (2 * p)) as i32;
                let moved = lens.cover_apply(lens.deck(power), a);

                let mut brute = f32::INFINITY;
                let mut lift = b;
                for _ in 0..p {
                    brute = brute.min(SphericalS3Embedded.distance(a, lift.normalize()));
                    lift = generator * lift;
                }
                assert_relative_eq!(lens.distance(a, b), brute, epsilon = 1e-4);

                assert_relative_eq!(lens.distance(a, b), lens.distance(b, a), epsilon = 1e-5);
                assert_relative_eq!(lens.distance(moved, b), lens.distance(a, b), epsilon = 1e-4);
                assert!(lens.distance(a, moved) < 1e-3);
                assert!(lens.distance(a, c) <= lens.distance(a, b) + lens.distance(b, c) + 1e-4);
            }
        }
    }

    #[test]
    fn the_injectivity_radius_is_half_the_shortest_deck_displacement() {
        let mut rng = Xorshift(0x41c6_4e6d);
        for &(p, q) in LENSES {
            let lens = LensSpace::new(p, q);
            let mut shortest = f32::INFINITY;
            for _ in 0..20_000 {
                let x = unit4(&mut rng);
                for power in 1..p as i32 {
                    let moved = lens.cover_apply(lens.deck(power), x);
                    shortest = shortest.min(SphericalS3Embedded.distance(x, moved));
                }
            }
            assert_relative_eq!(shortest, TAU / p as f32, epsilon = 2e-3);
            assert_relative_eq!(lens.injectivity_radius(), 0.5 * shortest, epsilon = 1e-3);
        }
    }

    #[test]
    fn exp_and_log_invert_within_the_injectivity_radius() {
        let mut rng = Xorshift(0x6c07_8965);
        for &(p, q) in LENSES {
            let lens = LensSpace::new(p, q);
            let radius = lens.injectivity_radius();
            for _ in 0..5_000 {
                let at = unit4(&mut rng);
                let raw = unit4(&mut rng);
                let tangent = raw - raw.dot(at) * at;
                if tangent.length_squared() < 1e-6 {
                    continue;
                }
                let v = tangent.normalize() * (0.9 * radius);
                let recovered = lens.log(at, lens.exp(at, v));
                assert!(
                    (recovered - v).length() < 1e-4,
                    "L({p}, {q}): log(exp(v)) missed v by {}",
                    (recovered - v).length()
                );
            }

            let at = Vec4::X;
            let v = Vec4::Y * (TAU / p as f32);
            assert!(lens.distance(at, lens.exp(at, v)) < 1e-3);
            assert!(lens.log(at, lens.exp(at, v)).length() < 1e-3);
        }
    }

    #[test]
    fn parallel_transport_lands_tangent_to_the_lift_the_caller_named() {
        let mut rng = Xorshift(0x339a_cb15);
        for &(p, q) in LENSES {
            let lens = LensSpace::new(p, q);
            for _ in 0..5_000 {
                let from = unit4(&mut rng);
                let to = unit4(&mut rng);
                let raw = unit4(&mut rng);
                let v = raw - raw.dot(from) * from;
                let carried = lens.parallel_transport(from, to, v);
                assert!(
                    carried.dot(to).abs() < 1e-4,
                    "L({p}, {q}): transported vector is not tangent at `to`"
                );
                assert_relative_eq!(carried.length(), v.length(), epsilon = 1e-4);
            }
        }
    }
}
