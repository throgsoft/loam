use std::collections::HashMap;

use loam_math::Bivector;
use loam_time::par::{self, Executor};

use crate::body::{BodyArena, RigidBody};
use crate::manifold::{
    ContactPoint, Manifold, BAUMGARTE_BETA, MAX_LINEAR_CORRECTION, PENETRATION_SLOP,
};
use crate::response::FRICTION_COEFF;
use crate::space::{integrate_body, PhysicsSpace, VectorOps};

use super::{split_two_mut, ConstraintUnit, PairKey, STALE_CONSTRAINT_KEY};

/// Measured with scoped threads on settled three-high stacks, where 128 box islands already gain.
pub const ISLANDS_PER_SOLVE_WORKER: usize = 64;

const SCATTERED_NOWHERE: u32 = u32::MAX;

struct ScratchUnit {
    key: PairKey,
    a: u32,
    b: u32,
    first: u32,
    count: u32,
}

struct IslandSolve<S: PhysicsSpace> {
    bodies: Vec<RigidBody<S>>,
    dense: Vec<u32>,
    units: Vec<ScratchUnit>,
    points: Vec<ContactPoint<S>>,
    push: Vec<f32>,
    saved: Vec<RigidBody<S>>,
}

impl<S: PhysicsSpace> Default for IslandSolve<S> {
    fn default() -> Self {
        Self {
            bodies: Vec::new(),
            dense: Vec::new(),
            units: Vec::new(),
            points: Vec::new(),
            push: Vec::new(),
            saved: Vec::new(),
        }
    }
}

pub(super) struct IslandSolver<S: PhysicsSpace> {
    scratch: Vec<IslandSolve<S>>,
    scratch_islands: usize,
    scratch_local: Vec<u32>,
}

impl<S: PhysicsSpace> Default for IslandSolver<S> {
    fn default() -> Self {
        Self {
            scratch: Vec::new(),
            scratch_islands: 0,
            scratch_local: Vec::new(),
        }
    }
}

impl<S: PhysicsSpace> IslandSolver<S> {
    pub(super) fn clear(&mut self) {
        self.scratch_islands = 0;
    }

    pub(super) fn gather(
        &mut self,
        bodies: &BodyArena<S>,
        manifolds: &HashMap<PairKey, Manifold<S>>,
        constraints: &[ConstraintUnit],
    ) {
        self.scratch_local.clear();
        self.scratch_local.resize(bodies.len(), SCATTERED_NOWHERE);
        self.scratch_islands = 0;
        let mut current = None;
        for unit in constraints {
            if current != Some(unit.island) {
                current = Some(unit.island);
                if self.scratch.len() == self.scratch_islands {
                    self.scratch.push(IslandSolve::default());
                }
                let island = &mut self.scratch[self.scratch_islands];
                island.bodies.clear();
                island.dense.clear();
                island.units.clear();
                island.points.clear();
                self.scratch_islands += 1;
            }
            let island = &mut self.scratch[self.scratch_islands - 1];
            let (i, j) = unit.dense;
            let a = gather_body(island, &mut self.scratch_local, bodies, i);
            let b = gather_body(island, &mut self.scratch_local, bodies, j);
            let manifold = manifolds.get(&unit.key).expect(STALE_CONSTRAINT_KEY);
            let first = island.points.len() as u32;
            island.points.extend_from_slice(&manifold.points);
            island.units.push(ScratchUnit {
                key: unit.key,
                a,
                b,
                first,
                count: manifold.points.len() as u32,
            });
        }
    }

    pub(super) fn solve(&mut self, space: &S, iterations: usize, dt: f32, executor: &dyn Executor) {
        let islands = &mut self.scratch[..self.scratch_islands];
        let workers = islands.len() / ISLANDS_PER_SOLVE_WORKER;
        if workers < 2 {
            for island in islands.iter_mut() {
                solve_island(space, iterations, dt, island);
            }
        } else {
            let chunk = islands.len().div_ceil(workers);
            par::for_each_chunk(executor, islands, chunk, |chunk| {
                for island in chunk {
                    solve_island(space, iterations, dt, island);
                }
            });
        }
    }

    pub(super) fn scatter(
        &self,
        bodies: &mut BodyArena<S>,
        manifolds: &mut HashMap<PairKey, Manifold<S>>,
    ) {
        for island in &self.scratch[..self.scratch_islands] {
            for (local, &dense) in island.dense.iter().enumerate() {
                if dense != SCATTERED_NOWHERE {
                    bodies[dense as usize] = island.bodies[local];
                }
            }
            for unit in &island.units {
                let manifold = manifolds.get_mut(&unit.key).expect(STALE_CONSTRAINT_KEY);
                let first = unit.first as usize;
                manifold
                    .points
                    .copy_from_slice(&island.points[first..first + unit.count as usize]);
            }
        }
    }
}

fn gather_body<S: PhysicsSpace>(
    island: &mut IslandSolve<S>,
    local: &mut [u32],
    bodies: &BodyArena<S>,
    dense: usize,
) -> u32 {
    if bodies[dense].inv_mass() == 0.0 {
        island.bodies.push(bodies[dense]);
        island.dense.push(SCATTERED_NOWHERE);
        return island.bodies.len() as u32 - 1;
    }
    if local[dense] == SCATTERED_NOWHERE {
        local[dense] = island.bodies.len() as u32;
        island.bodies.push(bodies[dense]);
        island.dense.push(dense as u32);
    }
    local[dense]
}

// Catto 2005, "Iterative Dynamics with Temporal Coherence", accumulated impulses with warm start.
fn solve_island<S>(space: &S, iterations: usize, dt: f32, island: &mut IslandSolve<S>)
where
    S: PhysicsSpace,
{
    let IslandSolve {
        bodies,
        units,
        points,
        push,
        saved,
        ..
    } = island;
    for _ in 0..iterations {
        for unit in units.iter() {
            let (a, b) = split_two_mut(bodies, unit.a as usize, unit.b as usize);
            let first = unit.first as usize;
            for cp in &mut points[first..first + unit.count as usize] {
                solve_normal_then_tangent(space, a, b, cp);
            }
        }
    }
    if dt <= 0.0 {
        return;
    }
    // Split impulse: a dropped pseudo velocity translates bodies apart, adding no energy or creep.
    saved.clear();
    saved.extend_from_slice(bodies);
    for body in bodies.iter_mut() {
        body.velocity = VectorOps::zero();
        body.angular_velocity = S::AngVel::zero();
    }
    push.clear();
    push.resize(points.len(), 0.0);
    for _ in 0..iterations {
        for unit in units.iter() {
            let (a, b) = split_two_mut(bodies, unit.a as usize, unit.b as usize);
            let range = unit.first as usize..(unit.first + unit.count) as usize;
            for (cp, accumulated) in points[range.clone()].iter().zip(&mut push[range]) {
                let depth = (cp.penetration - PENETRATION_SLOP).max(0.0);
                let target = (depth * BAUMGARTE_BETA).min(MAX_LINEAR_CORRECTION) / dt;
                let k_n = a.inv_mass() + b.inv_mass();
                if k_n <= 0.0 {
                    continue;
                }
                let v_rel = b.velocity - a.velocity;
                let next =
                    (*accumulated + (target - VectorOps::dot(v_rel, cp.normal)) / k_n).max(0.0);
                let impulse = next - *accumulated;
                *accumulated = next;
                if impulse != 0.0 {
                    a.velocity = a.velocity - cp.normal * (impulse * a.inv_mass());
                    b.velocity = b.velocity + cp.normal * (impulse * b.inv_mass());
                }
            }
        }
    }
    for (body, before) in bodies.iter_mut().zip(saved.iter()) {
        let from = body.position;
        integrate_body(space, body, dt);
        body.velocity = space.parallel_transport(from, body.position, before.velocity);
        body.angular_velocity = before.angular_velocity;
    }
}

fn solve_normal_then_tangent<S>(
    space: &S,
    a: &mut RigidBody<S>,
    b: &mut RigidBody<S>,
    cp: &mut ContactPoint<S>,
) where
    S: PhysicsSpace,
{
    debug_assert!(
        VectorOps::is_finite(cp.normal) && cp.penetration.is_finite(),
        "non-finite contact in solve_normal_then_tangent",
    );

    let v_rel_n_vec =
        space.velocity_at_point(b, cp.world_point) - space.velocity_at_point(a, cp.world_point);
    let v_n = VectorOps::dot(v_rel_n_vec, cp.normal);
    let k_n = space.effective_mass_inv(a, b, cp.world_point, cp.normal);

    if k_n > 0.0 {
        let dj = -(v_n + cp.velocity_bias) / k_n;
        let new_acc = (cp.normal_impulse + dj).max(0.0);
        let actual = new_acc - cp.normal_impulse;
        cp.normal_impulse = new_acc;
        if actual.abs() > 0.0 {
            space.apply_contact_impulse(a, b, cp.world_point, cp.normal, actual);
        }
    }

    let previous = cp.tangent_dir * cp.tangent_impulse;
    let v_rel =
        space.velocity_at_point(b, cp.world_point) - space.velocity_at_point(a, cp.world_point);
    let slip = v_rel - cp.normal * VectorOps::dot(v_rel, cp.normal);
    let speed = VectorOps::length(slip);
    let mut impulse = previous;
    if speed > 1e-8 {
        let direction = slip * (1.0 / speed);
        let inverse_mass = space.effective_mass_inv(a, b, cp.world_point, direction);
        if inverse_mass > 0.0 {
            impulse = impulse + slip * (1.0 / inverse_mass);
        }
    }
    let magnitude = VectorOps::length(impulse);
    let limit = cp.normal_impulse * FRICTION_COEFF;
    if magnitude > limit {
        impulse = impulse * (limit / magnitude);
    }
    cp.tangent_impulse = VectorOps::length(impulse);
    cp.tangent_dir = if cp.tangent_impulse > 0.0 {
        impulse * (1.0 / cp.tangent_impulse)
    } else {
        VectorOps::zero()
    };

    let delta = impulse - previous;
    let delta_magnitude = VectorOps::length(delta);
    if delta_magnitude > 0.0 {
        space.apply_contact_impulse(
            a,
            b,
            cp.world_point,
            delta * (1.0 / delta_magnitude),
            -delta_magnitude,
        );
    }
}
