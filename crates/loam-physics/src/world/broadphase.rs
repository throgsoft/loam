use crate::body::{BodyArena, BodyId};
use crate::collider::Collider;
use crate::geometry::GeometryStore;
use crate::integrator::{BroadphaseBound, PhysicsSpace};

use super::{canonical_pair, PairKey, StepCounters};

// do Carmo, Riemannian Geometry, 1992, ch. 7, prop. 3.6.
const BROADPHASE_TRIANGLE_SLACK: f32 = 4.0 * f32::EPSILON;

#[derive(Clone, Copy)]
struct RadialInterval {
    lo: f32,
    hi: f32,
    radius: f32,
    dense: u32,
    id: BodyId,
    dynamic: bool,
    group: u32,
    mask: u32,
}

#[derive(Default)]
pub(super) struct Broadphase {
    intervals: Vec<RadialInterval>,
    active: Vec<u32>,
}

impl Broadphase {
    pub(super) fn clear(&mut self) {
        self.intervals.clear();
        self.active.clear();
    }

    // Cohen, Lin, Manocha, Ponamgi, I-COLLIDE, 1995, sec. 3; sweep radial distances.
    pub(super) fn fill<S: PhysicsSpace>(
        &mut self,
        bodies: &BodyArena<S>,
        geometry: &GeometryStore,
        space: &S,
        pairs: &mut Vec<PairKey>,
        counters: &mut StepCounters,
    ) {
        let Self { intervals, active } = self;
        pairs.clear();
        intervals.clear();
        active.clear();
        let n = bodies.len();
        if n < 2 {
            return;
        }
        let certified = space.broadphase_bound() == BroadphaseBound::Certified;

        let anchor = (1..n).fold(0, |lowest, dense| {
            if bodies.id_at(dense) < bodies.id_at(lowest) {
                dense
            } else {
                lowest
            }
        });
        let origin = bodies[anchor].position;

        for dense in 0..n {
            let body = &bodies[dense];
            let radius = geometry
                .get(body.collider())
                .map_or(0.0, Collider::bounding_radius);
            let (lo, hi) = if certified {
                counters.distance_evals += 1;
                let d = space.distance(origin, body.position);
                let slack = d * BROADPHASE_TRIANGLE_SLACK;
                (d - radius - slack, d + radius + slack)
            } else {
                (f32::NEG_INFINITY, f32::INFINITY)
            };
            intervals.push(RadialInterval {
                lo,
                hi,
                radius,
                dense: dense as u32,
                id: bodies.id_at(dense),
                dynamic: body.inv_mass() != 0.0,
                group: body.collision_group,
                mask: body.collision_mask,
            });
        }

        intervals.sort_unstable_by(|a, b| a.lo.total_cmp(&b.lo).then(a.id.cmp(&b.id)));

        for i in 0..n {
            let entry = intervals[i];

            let mut visits = counters.index_visits;
            active.retain(|&open| {
                visits += 1;
                intervals[open as usize].hi >= entry.lo
            });
            counters.index_visits = visits;
            for &open in active.iter() {
                counters.index_visits += 1;
                let other = intervals[open as usize];
                if !entry.dynamic && !other.dynamic {
                    continue;
                }

                if entry.group & other.mask == 0 || other.group & entry.mask == 0 {
                    continue;
                }
                if certified {
                    counters.distance_evals += 1;
                    let gap = space.distance(
                        bodies[other.dense as usize].position,
                        bodies[entry.dense as usize].position,
                    );
                    if gap > other.radius + entry.radius {
                        continue;
                    }
                }
                pairs.push(canonical_pair(other.id, entry.id));
            }
            active.push(i as u32);
        }

        pairs.sort_unstable();
        counters.candidates = pairs.len() as u32;
    }
}

#[cfg(all(test, feature = "r3"))]
mod tests {
    use glam::Vec3;
    use loam_math::EuclideanR3;

    use crate::euclidean_r3::{register_default_narrowphase, sphere_body_r3};
    use crate::world::World;

    #[test]
    fn a_group_outside_the_others_mask_never_reaches_the_narrowphase() {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        let a = world.push_body(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 1.0, 1.0).unwrap());
        let b = world
            .push_body(sphere_body_r3(Vec3::new(0.5, 0.0, 0.0), Vec3::ZERO, 1.0, 1.0).unwrap());
        let mut pairs = Vec::new();
        world.broadphase_into(&mut pairs);
        assert_eq!(pairs.len(), 1, "the pair does not overlap to begin with");

        world.bodies[a].collision_group = 0b01;
        world.bodies[a].collision_mask = 0b01;
        world.bodies[b].collision_group = 0b10;
        world.bodies[b].collision_mask = 0b10;
        world.broadphase_into(&mut pairs);
        assert!(
            pairs.is_empty(),
            "a filtered pair still reached the narrowphase"
        );

        world.bodies[b].collision_mask = 0b11;
        world.broadphase_into(&mut pairs);
        assert!(
            pairs.is_empty(),
            "a one-sided mask edit produced a pair that collides in one direction"
        );

        world.bodies[a].collision_mask = 0b11;
        world.broadphase_into(&mut pairs);
        assert_eq!(pairs.len(), 1, "agreement did not restore the pair");
    }
}
