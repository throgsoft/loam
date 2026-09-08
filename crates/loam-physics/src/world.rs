use std::collections::BTreeMap;

use loam_math::{EuclideanR2, EuclideanR3, EuclideanR4};
use loam_time::StateHash;

use crate::body::{BodyArena, BodyId, RigidBody};
use crate::collider::Collider;
use crate::collision::VectorOps;
use crate::integrator::{integrate_body, PhysicsSpace};
use crate::manifold::{
    ContactPoint, Manifold, BAUMGARTE_BETA, DEFAULT_PGS_ITERS, MAX_LINEAR_CORRECTION,
    PENETRATION_SLOP, RESTITUTION_THRESHOLD,
};
use crate::narrowphase::Narrowphase;
use crate::response::FRICTION_COEFF;

/// Handles in ascending order.
pub type PairKey = (BodyId, BodyId);

fn canonical_pair(a: BodyId, b: BodyId) -> PairKey {
    debug_assert_ne!(a, b, "a body cannot pair with itself");
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// A connected component of the contact graph, over dynamic bodies only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Island {
    /// Lowest handle among [`Self::bodies`].
    pub id: BodyId,
    /// Ascending.
    pub bodies: Vec<BodyId>,
    /// Ascending pairs; static contacts belong to their dynamic side.
    pub constraints: Vec<PairKey>,
}

const STALE_CONSTRAINT_KEY: &str = "constraint buffer outlived its manifold";
const STALE_MANIFOLD_BODY: &str = "manifold key names a body that is gone";

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

fn bounding_radius(collider: &Collider) -> f32 {
    match collider {
        Collider::Sphere { radius, .. } | Collider::HyperSphere4D { radius, .. } => *radius,
        Collider::Box3 { half_extents } => half_extents.length(),
        Collider::Polygon2D { vertices } => max_norm(vertices.iter().map(|v| v.length_squared())),
        Collider::ConvexPolytope3D { vertices } => {
            max_norm(vertices.iter().map(|v| v.length_squared()))
        }
        Collider::ConvexPolytope4D { vertices } => {
            max_norm(vertices.iter().map(|v| v.length_squared()))
        }
        Collider::HalfSpace { .. } | Collider::HalfSpace4D { .. } => f32::INFINITY,
    }
}

fn max_norm(norms_squared: impl Iterator<Item = f32>) -> f32 {
    norms_squared.fold(0.0_f32, f32::max).sqrt()
}

#[derive(Clone, Copy)]
struct ConstraintUnit {
    island: BodyId,
    key: PairKey,
    /// Positions of `key.0` and `key.1`, in the key's order.
    dense: (usize, usize),
}

pub struct World<S: PhysicsSpace> {
    pub space: S,
    pub bodies: BodyArena<S>,
    pub gravity: Option<S::Vector>,
    pub narrowphase: Narrowphase<S>,
    /// PGS convergence depends on constraint order.
    pub manifolds: BTreeMap<PairKey, Manifold<S>>,
    pub pgs_iters: usize,
    pub time: f32,
    pair_order: Vec<PairKey>,
    constraints: Vec<ConstraintUnit>,
    broadphase_intervals: Vec<RadialInterval>,
    broadphase_active: Vec<u32>,
    touched_pairs: Vec<PairKey>,
    island_parent: Vec<u32>,
    island_labels: Vec<BodyId>,
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<World<EuclideanR2>>();
    assert_send_sync::<World<EuclideanR3>>();
    assert_send_sync::<World<EuclideanR4>>();
};

impl<S: PhysicsSpace> World<S> {
    pub fn new(space: S) -> Self {
        Self {
            space,
            bodies: BodyArena::new(),
            gravity: None,
            narrowphase: Narrowphase::new(),
            manifolds: BTreeMap::new(),
            pgs_iters: DEFAULT_PGS_ITERS,
            time: 0.0,
            pair_order: Vec::new(),
            constraints: Vec::new(),
            broadphase_intervals: Vec::new(),
            broadphase_active: Vec::new(),
            touched_pairs: Vec::new(),
            island_parent: Vec::new(),
            island_labels: Vec::new(),
        }
    }

    pub fn push_body(&mut self, body: RigidBody<S>) -> BodyId {
        self.bodies.spawn(body)
    }

    /// Also removes every manifold the body takes part in.
    pub fn despawn_body(&mut self, id: BodyId) -> bool {
        if self.bodies.despawn(id).is_none() {
            return false;
        }
        self.manifolds.retain(|&(a, b), _| a != id && b != id);
        true
    }

    /// `dt` is in seconds.
    pub fn step(&mut self, dt: f32)
    where
        S::Vector: VectorOps,
        S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    {
        self.apply_forces(dt);
        self.integrate(dt);
        self.update_manifolds();
        self.collect_constraints();
        self.prepare_solve(dt);
        self.warm_start();
        self.solve();

        self.time += dt;
    }

    fn apply_forces(&mut self, dt: f32)
    where
        S::Vector: VectorOps,
    {
        let Some(acceleration) = self.gravity else {
            return;
        };
        for body in self.bodies.iter_mut() {
            if body.inv_mass() != 0.0 {
                body.velocity = body.velocity + acceleration * dt;
            }
        }
    }

    fn integrate(&mut self, dt: f32)
    where
        S::Vector: VectorOps,
    {
        for i in 0..self.bodies.len() {
            integrate_body(&self.space, &mut self.bodies[i], dt);
        }
    }

    fn update_manifolds(&mut self)
    where
        S::Vector: VectorOps,
        S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    {
        let mut pairs = std::mem::take(&mut self.pair_order);
        Self::fill_broadphase(
            &self.bodies,
            &self.space,
            &mut self.broadphase_intervals,
            &mut self.broadphase_active,
            &mut pairs,
        );
        let mut touched = std::mem::take(&mut self.touched_pairs);
        touched.clear();

        for &key in &pairs {
            let (i, j) = self.dense_pair(key);
            let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
            // Refresh anchors before the narrowphase merges contacts.
            if let Some(manifold) = self.manifolds.get_mut(&key) {
                manifold.refresh(&self.space, a, b);
            }
            let Some(contact) = self.narrowphase.test(a, b, &self.space) else {
                continue;
            };
            touched.push(key);
            let restitution = contact.restitution;
            let manifold = self
                .manifolds
                .entry(key)
                .or_insert_with(|| Manifold::new(key.0, key.1, restitution));
            manifold.add_or_update(&self.space, a, b, contact);
        }

        touched.sort_unstable();
        self.manifolds
            .retain(|k, _| touched.binary_search(k).is_ok());
        self.touched_pairs = touched;
        self.pair_order = pairs;
    }

    // Manifold membership must stay fixed until the solve ends.
    fn collect_constraints(&mut self) {
        let mut parent = std::mem::take(&mut self.island_parent);
        let mut labels = std::mem::take(&mut self.island_labels);
        Self::fill_islands(
            &self.bodies,
            self.manifolds.keys().copied(),
            &mut parent,
            &mut labels,
        );
        let mut units = std::mem::take(&mut self.constraints);
        units.clear();
        units.extend(self.manifolds.keys().map(|&key| {
            let dense = self.dense_pair(key);
            ConstraintUnit {
                island: constraint_island(&self.bodies, &labels, dense),
                key,
                dense,
            }
        }));
        units.sort_unstable_by_key(|unit| (unit.island, unit.key));
        self.island_parent = parent;
        self.island_labels = labels;
        self.constraints = units;
    }

    // Restitution uses velocities from before the warm-start impulse.
    fn prepare_solve(&mut self, dt: f32)
    where
        S::Vector: VectorOps,
    {
        for unit in &self.constraints {
            debug_assert_eq!(unit.dense, self.dense_pair(unit.key));
            let (i, j) = unit.dense;
            let manifold = self
                .manifolds
                .get_mut(&unit.key)
                .unwrap_or_else(|| panic!("{STALE_CONSTRAINT_KEY}"));
            let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
            for cp in &mut manifold.points {
                let v_rel = self.space.velocity_at_point(b, cp.world_point)
                    - self.space.velocity_at_point(a, cp.world_point);
                let v_n = VectorOps::dot(v_rel, cp.normal);

                let restitution_bias = if v_n < -RESTITUTION_THRESHOLD {
                    manifold.restitution * v_n
                } else {
                    0.0
                };

                let baumgarte_bias = if dt > 0.0 {
                    let target = (cp.penetration - PENETRATION_SLOP).max(0.0) * BAUMGARTE_BETA / dt;
                    -target.min(MAX_LINEAR_CORRECTION / dt)
                } else {
                    0.0
                };

                cp.velocity_bias = restitution_bias + baumgarte_bias;

                cp.tangent_impulse = 0.0;
                cp.tangent_dir = VectorOps::zero();
            }
        }
    }

    fn warm_start(&mut self)
    where
        S::Vector: VectorOps,
    {
        for unit in &self.constraints {
            debug_assert_eq!(unit.dense, self.dense_pair(unit.key));
            let (i, j) = unit.dense;
            let manifold = self
                .manifolds
                .get(&unit.key)
                .unwrap_or_else(|| panic!("{STALE_CONSTRAINT_KEY}"));
            let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
            for cp in &manifold.points {
                if cp.normal_impulse > 0.0 {
                    self.space.apply_contact_impulse(
                        a,
                        b,
                        cp.world_point,
                        cp.normal,
                        cp.normal_impulse,
                    );
                }
            }
        }
    }

    fn solve(&mut self)
    where
        S::Vector: VectorOps,
    {
        for _ in 0..self.pgs_iters {
            for unit in &self.constraints {
                debug_assert_eq!(unit.dense, self.dense_pair(unit.key));
                let (i, j) = unit.dense;
                let manifold = self
                    .manifolds
                    .get_mut(&unit.key)
                    .unwrap_or_else(|| panic!("{STALE_CONSTRAINT_KEY}"));
                let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
                for cp in &mut manifold.points {
                    solve_normal_then_tangent(&self.space, a, b, cp);
                }
            }
        }
    }

    /// Sorted overlapping bounding-ball pairs, excluding pairs of static bodies.
    pub fn broadphase(&self) -> Vec<PairKey> {
        let mut pairs = Vec::new();
        Self::fill_broadphase(
            &self.bodies,
            &self.space,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut pairs,
        );
        pairs
    }

    /// Reuses the world's sweep storage and replaces `pairs` with the current candidates.
    pub fn broadphase_into(&mut self, pairs: &mut Vec<PairKey>) {
        Self::fill_broadphase(
            &self.bodies,
            &self.space,
            &mut self.broadphase_intervals,
            &mut self.broadphase_active,
            pairs,
        );
    }

    // Cohen, Lin, Manocha, Ponamgi, I-COLLIDE, 1995, sec. 3; sweep radial distances.
    fn fill_broadphase(
        bodies: &BodyArena<S>,
        space: &S,
        intervals: &mut Vec<RadialInterval>,
        active: &mut Vec<u32>,
        pairs: &mut Vec<PairKey>,
    ) {
        pairs.clear();
        intervals.clear();
        active.clear();
        let n = bodies.len();
        if n < 2 {
            return;
        }

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
            let radius = bounding_radius(body.collider());
            let d = space.distance(origin, body.position);
            let slack = d * BROADPHASE_TRIANGLE_SLACK;
            intervals.push(RadialInterval {
                lo: d - radius - slack,
                hi: d + radius + slack,
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

            active.retain(|&open| intervals[open as usize].hi >= entry.lo);
            for &open in active.iter() {
                let other = intervals[open as usize];
                if !entry.dynamic && !other.dynamic {
                    continue;
                }

                if entry.group & other.mask == 0 || other.group & entry.mask == 0 {
                    continue;
                }
                let gap = space.distance(
                    bodies[other.dense as usize].position,
                    bodies[entry.dense as usize].position,
                );
                if gap <= other.radius + entry.radius {
                    pairs.push(canonical_pair(other.id, entry.id));
                }
            }
            active.push(i as u32);
        }

        pairs.sort_unstable();
    }

    /// Hashes contact keys, point counts, and normal impulses in key order.
    pub fn hash_contacts(&self, hash: &mut StateHash) {
        for (key, manifold) in &self.manifolds {
            for id in [key.0, key.1] {
                hash.write_u32(id.slot());
                hash.write_u32(id.generation());
            }
            hash.write_u32(manifold.points.len() as u32);
            for point in &manifold.points {
                hash.write_f32(point.normal_impulse);
            }
        }
    }

    /// Hashes bodies by handle, then contacts; `sample_body` must emit fixed-width records.
    pub fn state_hash(&self, sample_body: impl Fn(&RigidBody<S>, &mut Vec<u32>)) -> u64 {
        let mut order: Vec<BodyId> = (0..self.bodies.len())
            .map(|dense| self.bodies.id_at(dense))
            .collect();
        order.sort_unstable();

        let mut words = Vec::new();
        for id in order {
            sample_body(&self.bodies[id], &mut words);
        }
        let mut hash = StateHash::new();
        hash.write_u32s(&words);
        self.hash_contacts(&mut hash);
        hash.finish()
    }

    /// Returns islands by id; stale manifold handles panic after a bare [`BodyArena::despawn`].
    pub fn islands(&self) -> Vec<Island> {
        let mut parent = Vec::new();
        let mut labels = Vec::new();
        Self::fill_islands(
            &self.bodies,
            self.manifolds.keys().copied(),
            &mut parent,
            &mut labels,
        );

        let mut by_id: BTreeMap<BodyId, Island> = BTreeMap::new();
        for &key in self.manifolds.keys() {
            if self.bodies[key.0].inv_mass() == 0.0 && self.bodies[key.1].inv_mass() == 0.0 {
                continue;
            }
            let id = constraint_island(&self.bodies, &labels, self.dense_pair(key));
            let island = by_id.entry(id).or_insert_with(|| Island {
                id,
                bodies: Vec::new(),
                constraints: Vec::new(),
            });
            island.constraints.push(key);
            for member in [key.0, key.1] {
                if self.bodies[member].inv_mass() != 0.0 {
                    island.bodies.push(member);
                }
            }
        }

        let mut islands: Vec<Island> = by_id.into_values().collect();
        for island in &mut islands {
            island.bodies.sort_unstable();
            island.bodies.dedup();
        }
        islands
    }

    fn fill_islands(
        bodies: &BodyArena<S>,
        touched: impl Iterator<Item = PairKey>,
        parent: &mut Vec<u32>,
        labels: &mut Vec<BodyId>,
    ) {
        let n = bodies.len();
        parent.clear();
        parent.extend(0..n as u32);
        labels.clear();
        labels.extend((0..n).map(|dense| bodies.id_at(dense)));

        for key in touched {
            let (i, j) = (
                bodies
                    .dense_index(key.0)
                    .unwrap_or_else(|| panic!("{STALE_MANIFOLD_BODY}")),
                bodies
                    .dense_index(key.1)
                    .unwrap_or_else(|| panic!("{STALE_MANIFOLD_BODY}")),
            );
            if bodies[i].inv_mass() == 0.0 || bodies[j].inv_mass() == 0.0 {
                continue;
            }
            let (a, b) = (find_root(parent, i), find_root(parent, j));
            if a != b {
                parent[a.max(b)] = a.min(b) as u32;
            }
        }

        for dense in 0..n {
            let root = find_root(parent, dense);
            labels[root] = labels[root].min(bodies.id_at(dense));
        }
        for dense in 0..n {
            let label = labels[find_root(parent, dense)];
            labels[dense] = label;
        }
    }

    fn dense_pair(&self, key: PairKey) -> (usize, usize) {
        (
            self.bodies
                .dense_index(key.0)
                .unwrap_or_else(|| panic!("{STALE_MANIFOLD_BODY}")),
            self.bodies
                .dense_index(key.1)
                .unwrap_or_else(|| panic!("{STALE_MANIFOLD_BODY}")),
        )
    }
}

// Tarjan and van Leeuwen, JACM 31(2), 1984, sec. 2.
fn find_root(parent: &mut [u32], mut dense: usize) -> usize {
    while parent[dense] as usize != dense {
        parent[dense] = parent[parent[dense] as usize];
        dense = parent[dense] as usize;
    }
    dense
}

fn constraint_island<S: PhysicsSpace>(
    bodies: &BodyArena<S>,
    labels: &[BodyId],
    dense: (usize, usize),
) -> BodyId {
    let (i, j) = dense;
    debug_assert!(
        bodies[i].inv_mass() != 0.0 || bodies[j].inv_mass() != 0.0,
        "a contact between two static bodies has no island to solve in",
    );
    if bodies[i].inv_mass() != 0.0 {
        labels[i]
    } else {
        labels[j]
    }
}

// Handle order can differ from dense storage order.
fn split_two_mut<T>(slice: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    debug_assert_ne!(i, j, "split_two_mut requires distinct indices");
    if i < j {
        let (left, right) = slice.split_at_mut(j);
        (&mut left[i], &mut right[0])
    } else {
        let (left, right) = slice.split_at_mut(i);
        (&mut right[0], &mut left[j])
    }
}

fn solve_normal_then_tangent<S>(
    space: &S,
    a: &mut RigidBody<S>,
    b: &mut RigidBody<S>,
    cp: &mut ContactPoint<S>,
) where
    S: PhysicsSpace,
    S::Vector: VectorOps,
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::determinism_fixture::{
        multi_island_world, record_flick_chamber_tape, replay_flick_chamber_tape, sample_body_r3,
        MULTI_ISLAND_DT, MULTI_ISLAND_STEPS, REPLAY_SEED,
    };
    use crate::euclidean_r3::{
        box_body, halfspace_body_r3, register_default_narrowphase, sphere_body_r3,
    };
    use glam::Vec3;
    use loam_math::{Bivector3, EuclideanR3, Space};
    use loam_time::Tape;

    const PERMUTATION_SEEDS: [u64; 4] = [1, 0x9e37_79b9_7f4a_7c15, 0xdead_beef_cafe_f00d, 424_242];

    fn constraint_order<S: PhysicsSpace>(world: &World<S>) -> Vec<PairKey> {
        world.constraints.iter().map(|unit| unit.key).collect()
    }

    mod alloc_probe {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        // try_with skips destroyed TLS; the const Cell initializer and callbacks cannot panic.
        thread_local! {
            static BYTES: Cell<usize> = const { Cell::new(0) };
        }

        pub struct Counting;

        // SAFETY: Methods preserve System contracts; const TLS and wrapping Cell updates cannot unwind.
        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                let _ = BYTES.try_with(|bytes| bytes.set(bytes.get().wrapping_add(layout.size())));
                // SAFETY: The caller supplies a valid nonzero allocation layout.
                unsafe { System.alloc(layout) }
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                // SAFETY: The caller supplies a live System allocation and its original layout.
                unsafe { System.dealloc(ptr, layout) }
            }

            unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                let _ = BYTES.try_with(|bytes| bytes.set(bytes.get().wrapping_add(new_size)));
                // SAFETY: The caller supplies a live System allocation, its layout, and a valid new size.
                unsafe { System.realloc(ptr, layout, new_size) }
            }
        }

        pub fn bytes_allocated_by(body: impl FnOnce()) -> usize {
            let before = BYTES.with(Cell::get);
            body();
            BYTES.with(Cell::get).wrapping_sub(before)
        }
    }

    #[global_allocator]
    static COUNTING_ALLOCATOR: alloc_probe::Counting = alloc_probe::Counting;

    #[test]
    fn a_group_outside_the_others_mask_never_reaches_the_narrowphase() {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        let a = world.push_body(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 1.0, 1.0).unwrap());
        let b = world
            .push_body(sphere_body_r3(Vec3::new(0.5, 0.0, 0.0), Vec3::ZERO, 1.0, 1.0).unwrap());
        assert_eq!(
            world.broadphase().len(),
            1,
            "the pair does not overlap to begin with"
        );

        world.bodies[a].collision_group = 0b01;
        world.bodies[a].collision_mask = 0b01;
        world.bodies[b].collision_group = 0b10;
        world.bodies[b].collision_mask = 0b10;
        assert!(
            world.broadphase().is_empty(),
            "a filtered pair still reached the narrowphase"
        );

        world.bodies[b].collision_mask = 0b11;
        assert!(
            world.broadphase().is_empty(),
            "a one-sided mask edit produced a pair that collides in one direction"
        );

        world.bodies[a].collision_mask = 0b11;
        assert_eq!(
            world.broadphase().len(),
            1,
            "agreement did not restore the pair"
        );
    }

    #[test]
    fn a_tape_replays_the_same_after_a_round_trip_through_its_byte_format() {
        let recorded = record_flick_chamber_tape(REPLAY_SEED);
        let decoded = Tape::decode(&recorded.encode()).expect("own encoding decodes");
        assert_eq!(decoded, recorded);
        assert_eq!(replay_flick_chamber_tape(&decoded), decoded.checkpoints());
    }

    #[test]
    fn a_flipped_input_word_moves_the_replayed_state_hash() {
        let recorded = record_flick_chamber_tape(REPLAY_SEED);
        let throw = (0..recorded.ticks())
            .find(|&tick| recorded.input(tick).expect("inside the tape")[0] != u32::MAX)
            .expect("the scripted stream throws at least once");

        let mut tampered = Tape::new(
            recorded.tick_hz(),
            recorded.seed(),
            recorded.words_per_tick(),
        );
        for tick in 0..recorded.ticks() {
            let mut frame: Vec<u32> = recorded.input(tick).expect("inside the tape").to_vec();
            if tick == throw {
                frame[2] ^= 1;
            }
            tampered.push_tick(&frame);
        }

        assert_ne!(
            replay_flick_chamber_tape(&tampered),
            recorded.checkpoints(),
            "one ulp of input made no difference, so the tape is not driving \
             the simulation"
        );
    }

    #[test]
    fn state_hash_is_invariant_under_arena_compaction() {
        let radii = [0.3_f32, 0.4, 0.5];
        let body = |i: usize| {
            sphere_body_r3(
                Vec3::new(i as f32, 2.0 * i as f32, 0.0),
                Vec3::new(0.0, -1.0, 0.5 * i as f32),
                radii[i],
                1.0 + i as f32,
            )
            .unwrap()
        };

        let mut direct = World::new(EuclideanR3);
        for i in 0..3 {
            direct.push_body(body(i));
        }

        let mut compacted = World::new(EuclideanR3);
        compacted.push_body(body(0));
        let doomed = compacted.push_body(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 9.0, 4.0).unwrap());
        compacted.push_body(body(1));
        compacted.push_body(body(2));
        assert!(compacted.despawn_body(doomed));

        let dense_order = |world: &World<EuclideanR3>| {
            world
                .bodies
                .iter()
                .map(|body| body.mass().to_bits())
                .collect::<Vec<_>>()
        };
        assert_ne!(
            dense_order(&direct),
            dense_order(&compacted),
            "the two worlds must differ in storage order or the pin is vacuous"
        );
        assert_eq!(
            direct.state_hash(sample_body_r3),
            compacted.state_hash(sample_body_r3),
        );
    }

    #[test]
    fn state_hash_covers_carried_contact_impulses() {
        let mut world = multi_island_world();
        for _ in 0..MULTI_ISLAND_STEPS {
            world.step(MULTI_ISLAND_DT);
        }
        let settled = world.state_hash(sample_body_r3);

        let point = world
            .manifolds
            .values_mut()
            .flat_map(|manifold| manifold.points.iter_mut())
            .next()
            .expect("the fixture rests in contact");
        point.normal_impulse = f32::from_bits(point.normal_impulse.to_bits() ^ 1);

        assert_ne!(
            settled,
            world.state_hash(sample_body_r3),
            "one ulp of warm-start impulse left the state hash unmoved"
        );
    }

    const SPHERE_RADIUS: f32 = 0.5;
    const GRAVITY_Y: f32 = -9.8;

    fn settled_sphere_stack(dt: f32, settle_steps: usize) -> World<EuclideanR3> {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.gravity = Some(Vec3::new(0.0, GRAVITY_Y, 0.0));

        for level in 0..3 {
            let y = SPHERE_RADIUS + level as f32 * 2.0 * SPHERE_RADIUS;
            let id = world.push_body(
                sphere_body_r3(Vec3::new(0.0, y, 0.0), Vec3::ZERO, SPHERE_RADIUS, 1.0).unwrap(),
            );
            world.bodies[id].restitution = 0.0;
        }
        let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());
        world.bodies[floor].restitution = 0.0;

        for _ in 0..settle_steps {
            world.step(dt);
        }
        world
    }

    mod solver_contracts {
        use glam::{Vec2, Vec4};
        use loam_math::{EuclideanR2, EuclideanR4};

        use super::*;
        use crate::euclidean_r2::{sphere_body, static_wall};
        use crate::euclidean_r4::{halfspace4_body_r4, sphere_body_r4};

        fn kinetic_energy<S: PhysicsSpace<Inertia = f32>>(body: &RigidBody<S>, omega: f32) -> f32
        where
            S::Vector: VectorOps,
        {
            0.5 * body.mass() * VectorOps::length_squared(body.velocity)
                + 0.5 * body.inertia * omega * omega
        }

        const FLOOR_HALF: Vec2 = Vec2::new(50.0, 1.0);

        fn floor_r2() -> RigidBody<EuclideanR2> {
            static_wall(Vec2::new(0.0, -FLOOR_HALF.y), FLOOR_HALF).unwrap()
        }

        const REBOUND_APPROACH: f32 = 2.0;
        const REBOUND_DT: f32 = 1.0 / 1000.0;
        const REBOUND_GAP: f32 = 0.01;

        fn assert_elastic_rebound_conserves_kinetic_energy<S>(
            mut world: World<S>,
            faller: BodyId,
            up: S::Vector,
            angular_speed: impl Fn(S::AngVel) -> f32,
        ) where
            S: PhysicsSpace<Inertia = f32>,
            S::Vector: VectorOps,
            S::Point: Copy + std::ops::Sub<Output = S::Vector>,
        {
            let energy_before = kinetic_energy(
                &world.bodies[faller],
                angular_speed(world.bodies[faller].angular_velocity),
            );
            let steps = (4.0 * REBOUND_GAP / (REBOUND_APPROACH * REBOUND_DT)).ceil() as usize;
            for _ in 0..steps {
                world.step(REBOUND_DT);
            }

            let body = &world.bodies[faller];
            let rebound = VectorOps::dot(body.velocity, up);
            assert!(rebound > 0.0, "body did not rebound: v·up = {rebound}");
            let spin = angular_speed(body.angular_velocity);
            assert!(spin < 1e-6, "central impact spun the body: |ω| = {spin}");

            let energy_after = kinetic_energy(body, spin);
            let ratio = energy_after / energy_before;
            assert!(
                ratio <= 1.0 + 1e-4,
                "e = 1 impact added energy: {energy_before} -> {energy_after}"
            );
            assert!(
                ratio >= 1.0 - 1e-4,
                "e = 1 impact lost energy: {energy_before} -> {energy_after}"
            );
        }

        #[test]
        fn perfectly_elastic_rebound_conserves_kinetic_energy_r2() {
            let mut world = World::new(EuclideanR2);
            crate::euclidean_r2::register_default_narrowphase(&mut world.narrowphase);
            let disk = world.push_body(
                sphere_body(
                    Vec2::new(0.0, SPHERE_RADIUS + REBOUND_GAP),
                    Vec2::new(0.0, -REBOUND_APPROACH),
                    SPHERE_RADIUS,
                    1.0,
                )
                .unwrap(),
            );
            let floor = world.push_body(floor_r2());
            world.bodies[disk].restitution = 1.0;
            world.bodies[floor].restitution = 1.0;

            assert_elastic_rebound_conserves_kinetic_energy(world, disk, Vec2::Y, |omega| {
                omega.0.abs()
            });
        }

        #[test]
        fn perfectly_elastic_rebound_conserves_kinetic_energy_r3() {
            let mut world = World::new(EuclideanR3);
            register_default_narrowphase(&mut world.narrowphase);
            let sphere = world.push_body(
                sphere_body_r3(
                    Vec3::new(0.0, SPHERE_RADIUS + REBOUND_GAP, 0.0),
                    Vec3::new(0.0, -REBOUND_APPROACH, 0.0),
                    SPHERE_RADIUS,
                    1.0,
                )
                .unwrap(),
            );
            let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());
            world.bodies[sphere].restitution = 1.0;
            world.bodies[floor].restitution = 1.0;

            assert_elastic_rebound_conserves_kinetic_energy(world, sphere, Vec3::Y, |omega| {
                omega.magnitude()
            });
        }

        #[test]
        fn perfectly_elastic_rebound_conserves_kinetic_energy_r4() {
            let mut world = World::new(EuclideanR4);
            crate::euclidean_r4::register_default_narrowphase(&mut world.narrowphase);
            let sphere = world.push_body(
                sphere_body_r4(
                    Vec4::new(0.0, SPHERE_RADIUS + REBOUND_GAP, 0.0, 0.0),
                    Vec4::new(0.0, -REBOUND_APPROACH, 0.0, 0.0),
                    SPHERE_RADIUS,
                    1.0,
                )
                .unwrap(),
            );
            let floor = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0).unwrap());
            world.bodies[sphere].restitution = 1.0;
            world.bodies[floor].restitution = 1.0;

            assert_elastic_rebound_conserves_kinetic_energy(world, sphere, Vec4::Y, |omega| {
                omega.magnitude()
            });
        }

        const SLIDE_SPEED: f32 = 5.0;
        const SLIDE_DT: f32 = 1.0 / 240.0;
        const SLIDE_STEPS: usize = 240;

        fn assert_tangent_impulse_stays_inside_the_coulomb_cone<S>(
            mut world: World<S>,
            slider: BodyId,
            slide: S::Vector,
            up: S::Vector,
        ) where
            S: PhysicsSpace,
            S::Vector: VectorOps,
            S::Point: Copy + std::ops::Sub<Output = S::Vector>,
        {
            let mut cone_ever_binds = false;
            for _ in 0..SLIDE_STEPS {
                world.step(SLIDE_DT);
                for manifold in world.manifolds.values() {
                    for cp in &manifold.points {
                        let cap = cp.normal_impulse * FRICTION_COEFF;
                        assert!(
                            cp.tangent_impulse <= cap + 1e-6,
                            "friction escaped the cone: jt = {}, μ·jn = {cap}",
                            cp.tangent_impulse
                        );
                        assert!(
                            cp.tangent_impulse >= 0.0,
                            "tangent accumulator went negative: {}",
                            cp.tangent_impulse
                        );
                        if cap > 1e-6 && cp.tangent_impulse >= cap - 1e-6 {
                            cone_ever_binds = true;
                        }
                    }
                }
            }

            assert!(
                cone_ever_binds,
                "friction never saturated, so the clamp was never exercised"
            );
            let body = &world.bodies[slider];
            let along = VectorOps::dot(body.velocity, slide);
            assert!(
                along < SLIDE_SPEED,
                "friction did not brake the slide: v·slide = {along}"
            );

            let contact = world.space.exp(body.position, up * -SPHERE_RADIUS);
            let angular_at_contact = world.space.velocity_at_point(body, contact) - body.velocity;
            let backspin = VectorOps::dot(angular_at_contact, slide);
            assert!(
                backspin < -1e-3,
                "friction spun the body away from rolling: contact-point \
                 angular velocity along the slide is {backspin}"
            );
        }

        #[test]
        fn tangent_impulse_stays_inside_the_coulomb_cone_r2() {
            let mut world = World::new(EuclideanR2);
            crate::euclidean_r2::register_default_narrowphase(&mut world.narrowphase);
            world.gravity = Some(Vec2::new(0.0, GRAVITY_Y));
            let disk = world.push_body(
                sphere_body(
                    Vec2::new(0.0, SPHERE_RADIUS),
                    Vec2::new(SLIDE_SPEED, 0.0),
                    SPHERE_RADIUS,
                    1.0,
                )
                .unwrap(),
            );
            let floor = world.push_body(floor_r2());
            world.bodies[disk].restitution = 0.0;
            world.bodies[floor].restitution = 0.0;

            assert_tangent_impulse_stays_inside_the_coulomb_cone(world, disk, Vec2::X, Vec2::Y);
        }

        #[test]
        fn tangent_impulse_stays_inside_the_coulomb_cone_r3() {
            let mut world = World::new(EuclideanR3);
            register_default_narrowphase(&mut world.narrowphase);
            world.gravity = Some(Vec3::new(0.0, GRAVITY_Y, 0.0));
            let sphere = world.push_body(
                sphere_body_r3(
                    Vec3::new(0.0, SPHERE_RADIUS, 0.0),
                    Vec3::new(SLIDE_SPEED, 0.0, 0.0),
                    SPHERE_RADIUS,
                    1.0,
                )
                .unwrap(),
            );
            let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());
            world.bodies[sphere].restitution = 0.0;
            world.bodies[floor].restitution = 0.0;

            assert_tangent_impulse_stays_inside_the_coulomb_cone(world, sphere, Vec3::X, Vec3::Y);
        }

        #[test]
        fn tangent_impulse_stays_inside_the_coulomb_cone_r4() {
            let mut world = World::new(EuclideanR4);
            crate::euclidean_r4::register_default_narrowphase(&mut world.narrowphase);
            world.gravity = Some(Vec4::new(0.0, GRAVITY_Y, 0.0, 0.0));
            let sphere = world.push_body(
                sphere_body_r4(
                    Vec4::new(0.0, SPHERE_RADIUS, 0.0, 0.0),
                    Vec4::new(SLIDE_SPEED, 0.0, 0.0, 0.0),
                    SPHERE_RADIUS,
                    1.0,
                )
                .unwrap(),
            );
            let floor = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0).unwrap());
            world.bodies[sphere].restitution = 0.0;
            world.bodies[floor].restitution = 0.0;

            assert_tangent_impulse_stays_inside_the_coulomb_cone(world, sphere, Vec4::X, Vec4::Y);
        }

        const STACK_DT: f32 = 1.0 / 240.0;
        const STACK_SETTLE_STEPS: usize = 400;
        const STACK_LEVELS: usize = 3;

        fn clear_warm_start<S: PhysicsSpace>(world: &mut World<S>) {
            for manifold in world.manifolds.values_mut() {
                for cp in &mut manifold.points {
                    cp.normal_impulse = 0.0;
                }
            }
        }

        fn velocities<S: PhysicsSpace>(world: &World<S>) -> Vec<S::Vector>
        where
            S::Vector: VectorOps,
        {
            world.bodies.iter().map(|b| b.velocity).collect()
        }

        fn normal_impulses<S: PhysicsSpace>(world: &World<S>) -> Vec<f32> {
            world
                .manifolds
                .values()
                .flat_map(|m| m.points.iter().map(|cp| cp.normal_impulse))
                .collect()
        }

        fn max_velocity_gap<V: VectorOps>(a: &[V], b: &[V]) -> f32 {
            assert_eq!(a.len(), b.len(), "body layouts diverged");
            a.iter()
                .zip(b)
                .map(|(x, y)| VectorOps::length(*x - *y))
                .fold(0.0_f32, f32::max)
        }

        fn max_scalar_gap(a: &[f32], b: &[f32]) -> f32 {
            assert_eq!(a.len(), b.len(), "manifold layouts diverged");
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f32, f32::max)
        }

        fn assert_warm_started_step_matches_cold_started_converged_step<S>(
            fixture: impl Fn() -> World<S>,
        ) where
            S: PhysicsSpace,
            S::Vector: VectorOps,
            S::Point: Copy + std::ops::Sub<Output = S::Vector>,
        {
            let mut warm = fixture();
            let mut cold_converged = fixture();
            let mut cold_default = fixture();

            assert!(
                warm.manifolds
                    .values()
                    .flat_map(|m| &m.points)
                    .any(|cp| cp.normal_impulse > 0.0),
                "fixture carries no warm-start payload"
            );

            clear_warm_start(&mut cold_converged);
            clear_warm_start(&mut cold_default);
            cold_converged.pgs_iters = 400;

            warm.step(STACK_DT);
            cold_converged.step(STACK_DT);
            cold_default.step(STACK_DT);

            let converged = velocities(&cold_converged);
            let warm_gap = max_velocity_gap(&velocities(&warm), &converged);
            let cold_gap = max_velocity_gap(&velocities(&cold_default), &converged);

            assert!(
                warm_gap < 1e-5,
                "warm-started step diverged from the converged solve by {warm_gap} m/s"
            );
            assert!(
                warm_gap < cold_gap,
                "warm start bought no convergence: warm {warm_gap} vs cold {cold_gap}"
            );

            let impulse_gap =
                max_scalar_gap(&normal_impulses(&warm), &normal_impulses(&cold_converged));
            let reference = normal_impulses(&cold_converged)
                .into_iter()
                .fold(0.0_f32, f32::max);
            assert!(
                impulse_gap < 1e-3 * reference,
                "warm-started accumulator diverged by {impulse_gap} against a peak impulse of {reference}"
            );
        }

        fn settled_disk_stack_r2() -> World<EuclideanR2> {
            let mut world = World::new(EuclideanR2);
            crate::euclidean_r2::register_default_narrowphase(&mut world.narrowphase);
            world.gravity = Some(Vec2::new(0.0, GRAVITY_Y));

            for level in 0..STACK_LEVELS {
                let y = SPHERE_RADIUS + level as f32 * 2.0 * SPHERE_RADIUS;
                let id = world.push_body(
                    sphere_body(Vec2::new(0.0, y), Vec2::ZERO, SPHERE_RADIUS, 1.0).unwrap(),
                );
                world.bodies[id].restitution = 0.0;
            }
            let floor = world.push_body(floor_r2());
            world.bodies[floor].restitution = 0.0;

            for _ in 0..STACK_SETTLE_STEPS {
                world.step(STACK_DT);
            }
            world
        }

        fn settled_sphere_stack_r4() -> World<EuclideanR4> {
            let mut world = World::new(EuclideanR4);
            crate::euclidean_r4::register_default_narrowphase(&mut world.narrowphase);
            world.gravity = Some(Vec4::new(0.0, GRAVITY_Y, 0.0, 0.0));

            for level in 0..STACK_LEVELS {
                let y = SPHERE_RADIUS + level as f32 * 2.0 * SPHERE_RADIUS;
                let id = world.push_body(
                    sphere_body_r4(Vec4::new(0.0, y, 0.0, 0.0), Vec4::ZERO, SPHERE_RADIUS, 1.0)
                        .unwrap(),
                );
                world.bodies[id].restitution = 0.0;
            }
            let floor = world.push_body(halfspace4_body_r4(Vec4::Y, 0.0).unwrap());
            world.bodies[floor].restitution = 0.0;

            for _ in 0..STACK_SETTLE_STEPS {
                world.step(STACK_DT);
            }
            world
        }

        #[test]
        fn warm_started_step_matches_cold_started_converged_step_r2() {
            assert_warm_started_step_matches_cold_started_converged_step(settled_disk_stack_r2);
        }

        #[test]
        fn warm_started_step_matches_cold_started_converged_step_r3() {
            assert_warm_started_step_matches_cold_started_converged_step(|| {
                settled_sphere_stack(STACK_DT, STACK_SETTLE_STEPS)
            });
        }

        #[test]
        fn warm_started_step_matches_cold_started_converged_step_r4() {
            assert_warm_started_step_matches_cold_started_converged_step(settled_sphere_stack_r4);
        }
    }

    #[test]
    fn shrinking_normal_impulse_releases_applied_friction_even_without_slip() {
        use crate::euclidean_r2::sphere_body;
        use glam::Vec2;
        use loam_math::EuclideanR2;

        for slip in [0.0, 1.0] {
            let mut a = sphere_body(Vec2::ZERO, Vec2::ZERO, 1.0, 0.0).unwrap();
            let mut b = sphere_body(Vec2::ZERO, Vec2::new(slip, 2.0), 1.0, 1.0).unwrap();
            let previous_friction = 2.0 * FRICTION_COEFF;
            let mut contact = ContactPoint {
                world_point: Vec2::ZERO,
                anchor_a: Vec2::ZERO,
                anchor_b: Vec2::ZERO,
                normal: Vec2::Y,
                penetration: 0.0,
                normal_impulse: 2.0,
                tangent_dir: Vec2::X,
                tangent_impulse: previous_friction,
                velocity_bias: 0.0,
            };
            solve_normal_then_tangent(&EuclideanR2, &mut a, &mut b, &mut contact);
            assert_eq!(contact.normal_impulse, 0.0);
            assert_eq!(contact.tangent_impulse, 0.0);
            assert!((b.velocity - Vec2::X * (slip + previous_friction)).length() < 1e-6);
        }
    }

    const ISLAND_X: [f32; 4] = [-4.0, 0.0, 4.0, 8.0];

    fn settled_islands(dt: f32, settle_steps: usize) -> (World<EuclideanR3>, BodyId, Vec<BodyId>) {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.gravity = Some(Vec3::new(0.0, GRAVITY_Y, 0.0));

        let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());
        world.bodies[floor].restitution = 0.0;
        let mut spheres = Vec::with_capacity(ISLAND_X.len());
        for x in ISLAND_X {
            let id = world.push_body(island_sphere(x));
            world.bodies[id].restitution = 0.0;
            spheres.push(id);
        }

        for _ in 0..settle_steps {
            world.step(dt);
        }
        (world, floor, spheres)
    }

    fn island_sphere(x: f32) -> RigidBody<EuclideanR3> {
        sphere_body_r3(
            Vec3::new(x, SPHERE_RADIUS, 0.0),
            Vec3::ZERO,
            SPHERE_RADIUS,
            1.0,
        )
        .unwrap()
    }

    fn body_state(world: &World<EuclideanR3>, id: BodyId) -> (Vec3, Vec3, Bivector3) {
        let body = &world.bodies[id];
        (body.position, body.velocity, body.angular_velocity)
    }

    fn normal_impulses(world: &World<EuclideanR3>, key: PairKey) -> Vec<f32> {
        world.manifolds[&key]
            .points
            .iter()
            .map(|cp| cp.normal_impulse)
            .collect()
    }

    #[test]
    fn sleeping_after_a_step_removes_inactive_islands_immediately() {
        let (mut world, floor, spheres) = settled_islands(1.0 / 240.0, 1);
        for id in &spheres {
            assert!(world.manifolds.contains_key(&(floor, *id)));
            world.bodies[*id].sleep();
        }
        assert!(world.islands().is_empty());
        world.bodies[spheres[0]].wake();
        let islands = world.islands();
        assert_eq!(islands.len(), 1);
        assert_eq!(islands[0].bodies, vec![spheres[0]]);
    }

    #[test]
    fn despawn_preserves_surviving_manifold_keys_and_warm_start_impulses() {
        let dt = 1.0 / 240.0;
        let settle_steps = 400;
        let (mut world, floor, spheres) = settled_islands(dt, settle_steps);
        let (mut control, _, control_spheres) = settled_islands(dt, settle_steps);

        let doomed = spheres[1];
        let keeper = spheres[3];
        let keeper_key = (floor, keeper);
        let control_key = (floor, control_spheres[3]);

        let impulses_before = normal_impulses(&world, keeper_key);
        assert!(
            impulses_before.iter().any(|&jn| jn > 0.0),
            "fixture carries no warm-start payload, so nothing below is being preserved"
        );
        let keeper_position_before = world.bodies.dense_index(keeper).unwrap();

        assert!(world.despawn_body(doomed));

        assert_ne!(
            world.bodies.dense_index(keeper).unwrap(),
            keeper_position_before,
            "despawn moved no surviving body, so this test never reached the compaction case"
        );
        assert!(
            !world
                .manifolds
                .keys()
                .any(|&(a, b)| a == doomed || b == doomed),
            "the removed body's manifolds outlived it"
        );
        assert_eq!(
            normal_impulses(&world, keeper_key),
            impulses_before,
            "compaction disturbed a surviving manifold's warm-start impulses"
        );

        for _ in 0..60 {
            world.step(dt);
            control.step(dt);
        }

        assert_eq!(
            body_state(&world, keeper),
            body_state(&control, control_spheres[3]),
            "an unrelated despawn perturbed a surviving island"
        );
        let impulses_after = normal_impulses(&world, keeper_key);
        assert_eq!(
            impulses_after,
            normal_impulses(&control, control_key),
            "an unrelated despawn moved a surviving manifold's impulses"
        );
        assert!(
            impulses_after.iter().any(|&jn| jn > 0.0),
            "the surviving contact stopped carrying an impulse"
        );
    }

    #[test]
    fn spawn_mid_simulation_leaves_existing_islands_bit_identical() {
        let dt = 1.0 / 240.0;
        let settle_steps = 400;
        let (mut world, floor, spheres) = settled_islands(dt, settle_steps);
        let (mut control, _, control_spheres) = settled_islands(dt, settle_steps);

        let keeper = spheres[0];
        let keeper_key = (floor, keeper);
        let newcomer = world.push_body(island_sphere(12.0));
        world.bodies[newcomer].restitution = 0.0;

        for _ in 0..60 {
            world.step(dt);
            control.step(dt);
        }

        assert!(
            world.manifolds.contains_key(&(floor, newcomer)),
            "the spawned body never made contact, so it exercised no solver state"
        );
        assert_eq!(
            body_state(&world, keeper),
            body_state(&control, control_spheres[0]),
            "a spawn perturbed an existing island"
        );
        assert_eq!(
            normal_impulses(&world, keeper_key),
            normal_impulses(&control, (floor, control_spheres[0])),
            "a spawn moved an existing manifold's warm-start impulses"
        );
    }

    #[test]
    fn a_recycled_slot_inherits_no_manifold_from_the_previous_body() {
        let dt = 1.0 / 240.0;
        let (mut world, floor, spheres) = settled_islands(dt, 400);
        let doomed = spheres[1];
        assert!(world.manifolds.contains_key(&(floor, doomed)));

        assert!(world.despawn_body(doomed));
        assert!(
            !world.despawn_body(doomed),
            "a stale handle despawned a second body"
        );
        assert!(world.bodies.get(doomed).is_none());

        let reborn = world.push_body(island_sphere(ISLAND_X[1]));
        assert_eq!(
            reborn.slot(),
            doomed.slot(),
            "the slot was not recycled, so this test is not exercising aliasing"
        );
        assert_ne!(reborn, doomed);
        assert!(
            world.bodies.get(doomed).is_none(),
            "the stale handle resolved to the body that took its slot"
        );

        world.step(dt);
        assert!(
            world.manifolds.contains_key(&(floor, reborn)),
            "the respawned body made no contact"
        );
        assert!(
            !world.manifolds.contains_key(&(floor, doomed)),
            "a manifold keyed on the despawned body came back with the slot"
        );
    }

    #[test]
    fn split_two_mut_returns_borrows_in_argument_order() {
        let mut slice = [0u32, 1, 2, 3];
        for (i, j) in [(1usize, 3usize), (3, 1)] {
            let (a, b) = split_two_mut(&mut slice, i, j);
            assert_eq!((*a, *b), (i as u32, j as u32), "split_two_mut({i}, {j})");
        }
    }

    fn stacked_pair_world(doomed_first: bool) -> (World<EuclideanR3>, BodyId, BodyId, BodyId) {
        const LOWER_HALF_EXTENT: f32 = 0.5;
        const UPPER_HALF_EXTENT: f32 = 0.35;
        const DOOMED_X: f32 = -6.0;

        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.gravity = Some(Vec3::new(0.0, GRAVITY_Y, 0.0));

        let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());
        world.bodies[floor].restitution = 0.0;

        let spawned_first = doomed_first.then(|| world.push_body(island_sphere(DOOMED_X)));
        let lower = world.push_body(
            box_body(
                Vec3::new(0.0, LOWER_HALF_EXTENT, 0.0),
                Vec3::ZERO,
                Vec3::splat(LOWER_HALF_EXTENT),
                1.0,
            )
            .unwrap(),
        );
        let upper = world.push_body(
            box_body(
                Vec3::new(0.0, 2.0 * LOWER_HALF_EXTENT + UPPER_HALF_EXTENT, 0.0),
                Vec3::ZERO,
                Vec3::splat(UPPER_HALF_EXTENT),
                3.0,
            )
            .unwrap(),
        );
        let doomed = spawned_first.unwrap_or_else(|| world.push_body(island_sphere(DOOMED_X)));

        for id in [lower, upper, doomed] {
            world.bodies[id].restitution = 0.0;
        }
        (world, lower, upper, doomed)
    }

    fn despawned_pair_world(
        doomed_first: bool,
        dt: f32,
    ) -> (World<EuclideanR3>, BodyId, BodyId, PairKey) {
        const SETTLE_STEPS: usize = 40;

        let (mut world, lower, upper, doomed) = stacked_pair_world(doomed_first);
        for _ in 0..SETTLE_STEPS {
            world.step(dt);
        }

        let key = canonical_pair(lower, upper);
        assert!(
            world.manifolds.contains_key(&key),
            "the two dynamic bodies never settled into contact"
        );
        assert!(world.despawn_body(doomed));

        let (i, j) = world.dense_pair(key);
        assert_eq!(
            i > j,
            doomed_first,
            "the pair is stored at {i}, {j}, which is not the order this fixture \
             was built to produce"
        );
        (world, lower, upper, key)
    }

    #[test]
    fn contact_normal_points_from_the_pair_key_low_body_to_the_high_one() {
        let dt = 1.0 / 240.0;
        for doomed_first in [false, true] {
            let (mut world, _, _, key) = despawned_pair_world(doomed_first, dt);
            world.step(dt);

            let manifold = world.manifolds.get(&key).expect("the pair separated");
            let key_axis = world.bodies[key.1].position - world.bodies[key.0].position;
            assert!(!manifold.points.is_empty(), "manifold carries no contact");
            for cp in &manifold.points {
                assert!(
                    cp.normal.dot(key_axis) > 0.0,
                    "normal {} points back toward the key's low body",
                    cp.normal
                );
            }
        }
    }

    #[test]
    fn storage_order_does_not_reach_a_contacting_pairs_trajectory() {
        let dt = 1.0 / 240.0;
        let steps = 120;
        let mut trajectories = Vec::new();
        for doomed_first in [false, true] {
            let (mut world, lower, upper, key) = despawned_pair_world(doomed_first, dt);
            let mut contact_steps = 0;
            let mut trajectory = Vec::with_capacity(steps);
            for _ in 0..steps {
                world.step(dt);
                if world.manifolds.contains_key(&key) {
                    contact_steps += 1;
                }
                trajectory.push((body_state(&world, lower), body_state(&world, upper)));
            }
            assert_eq!(
                contact_steps, steps,
                "the pair held contact for only {contact_steps} of {steps} steps"
            );
            trajectories.push(trajectory);
        }

        let step = trajectories[0]
            .iter()
            .zip(&trajectories[1])
            .position(|(a, b)| a != b);
        assert!(
            step.is_none(),
            "storage order reached the solve: the pair diverged from the control \
             at step {step:?}"
        );
    }

    // Marsaglia 2003, "Xorshift RNGs", Journal of Statistical Software 8(14), 13/7/17.
    struct Xorshift(u64);

    impl Xorshift {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            let unit = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
            lo + (hi - lo) * unit
        }
    }

    fn random_scene(seed: u64, count: usize, spread: f32) -> World<EuclideanR3> {
        let mut rng = Xorshift::new(seed);
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.gravity = Some(Vec3::new(0.0, GRAVITY_Y, 0.0));
        world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());

        let mut spawned = Vec::with_capacity(count);
        for _ in 0..count {
            let position = Vec3::new(
                rng.range(-spread, spread),
                rng.range(0.5, spread + 0.5),
                rng.range(-spread, spread),
            );
            let id = if rng.next_u64() & 1 == 0 {
                world.push_body(
                    sphere_body_r3(position, Vec3::ZERO, rng.range(0.2, 0.8), 1.0).unwrap(),
                )
            } else {
                world.push_body(
                    box_body(position, Vec3::ZERO, Vec3::splat(rng.range(0.2, 0.6)), 1.0).unwrap(),
                )
            };
            if rng.next_u64().is_multiple_of(8) {
                assert!(world.bodies[id].set_mass(0.0));
            }
            world.bodies[id].restitution = 0.0;
            spawned.push(id);
        }
        for doomed in spawned.iter().step_by(5) {
            assert!(world.despawn_body(*doomed));
        }
        world
    }

    fn all_pairs_reference(world: &World<EuclideanR3>) -> Vec<PairKey> {
        let mut pairs = Vec::new();
        let n = world.bodies.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let (a, b) = (&world.bodies[i], &world.bodies[j]);
                if a.inv_mass() == 0.0 && b.inv_mass() == 0.0 {
                    continue;
                }
                let reach = bounding_radius(a.collider()) + bounding_radius(b.collider());
                if world.space.distance(a.position, b.position) <= reach {
                    pairs.push(canonical_pair(world.bodies.id_at(i), world.bodies.id_at(j)));
                }
            }
        }
        pairs.sort_unstable();
        pairs
    }

    fn dynamic_body_count(world: &World<EuclideanR3>) -> usize {
        world.bodies.iter().filter(|b| b.inv_mass() != 0.0).count()
    }

    const RANDOM_SCENE_SHAPES: [(usize, f32); 3] = [(12, 2.0), (40, 6.0), (80, 3.0)];

    #[test]
    fn sweep_broadphase_emits_exactly_the_all_pairs_candidate_set() {
        let mut ever_beyond_the_floor = false;
        for seed in PERMUTATION_SEEDS {
            for (count, spread) in RANDOM_SCENE_SHAPES {
                let mut world = random_scene(seed, count, spread);
                for step in 0..8 {
                    let expected = all_pairs_reference(&world);
                    assert_eq!(
                        world.broadphase(),
                        expected,
                        "seed {seed}, {count} bodies, spread {spread}, step {step}"
                    );
                    ever_beyond_the_floor |= expected.len() > dynamic_body_count(&world);
                    world.step(1.0 / 240.0);
                }
            }
        }
        assert!(
            ever_beyond_the_floor,
            "no scene ever produced a candidate pair between two finite colliders"
        );
    }

    #[test]
    fn broadphase_culls_only_pairs_the_narrowphase_would_reject() {
        for seed in PERMUTATION_SEEDS {
            let mut world = random_scene(seed, 40, 3.0);
            for step in 0..8 {
                let emitted = world.broadphase();
                let n = world.bodies.len();
                let mut culled = 0usize;
                for i in 0..n {
                    for j in (i + 1)..n {
                        if world.bodies[i].inv_mass() == 0.0 && world.bodies[j].inv_mass() == 0.0 {
                            continue;
                        }
                        let key = canonical_pair(world.bodies.id_at(i), world.bodies.id_at(j));
                        if emitted.binary_search(&key).is_ok() {
                            continue;
                        }
                        culled += 1;
                        let contact = world.narrowphase.test(
                            &world.bodies[i],
                            &world.bodies[j],
                            &world.space,
                        );
                        assert!(
                            contact.is_none(),
                            "seed {seed} step {step}: the sweep culled {key:?}, which the \
                             narrowphase reports in contact"
                        );
                    }
                }
                assert!(
                    culled > 0,
                    "seed {seed} step {step}: nothing was culled, so this pass proved nothing"
                );
                world.step(1.0 / 240.0);
            }
        }
    }

    #[test]
    fn broadphase_emits_strictly_ascending_keys_under_disagreeing_storage_order() {
        for seed in PERMUTATION_SEEDS {
            let world = random_scene(seed, 40, 3.0);
            let disagrees = (1..world.bodies.len())
                .any(|dense| world.bodies.id_at(dense) < world.bodies.id_at(dense - 1));
            assert!(
                disagrees,
                "seed {seed}: storage order still agrees with handle order, so this \
                 scene cannot tell the two apart"
            );

            let pairs = world.broadphase();
            assert!(pairs.len() > 1, "seed {seed}: too few pairs to be ordered");
            assert!(
                pairs.windows(2).all(|w| w[0] < w[1]),
                "seed {seed}: emission order is not strictly ascending in BodyId"
            );
        }
    }

    #[test]
    fn broadphase_prunes_the_quadratic_pair_set_at_scale() {
        let world = random_scene(PERMUTATION_SEEDS[1], 200, 20.0);
        let n = world.bodies.len();
        assert!(
            n >= 100,
            "the scale case needs at least 100 bodies, got {n}"
        );
        let all_pairs = n * (n - 1) / 2;
        let emitted = world.broadphase().len();
        assert!(
            emitted * 10 < all_pairs,
            "the sweep emitted {emitted} of {all_pairs} pairs, which is no better than \
             a constant-factor cull"
        );
    }

    #[test]
    fn broadphase_fill_allocates_nothing_after_the_first_pass() {
        let world = random_scene(PERMUTATION_SEEDS[0], 120, 8.0);
        let mut intervals = Vec::new();
        let mut active = Vec::new();
        let mut pairs = Vec::new();

        for _ in 0..2 {
            World::fill_broadphase(
                &world.bodies,
                &world.space,
                &mut intervals,
                &mut active,
                &mut pairs,
            );
        }
        assert!(!pairs.is_empty(), "the fixture produced no pairs to emit");

        let bytes = alloc_probe::bytes_allocated_by(|| {
            for _ in 0..16 {
                World::fill_broadphase(
                    &world.bodies,
                    &world.space,
                    &mut intervals,
                    &mut active,
                    &mut pairs,
                );
            }
        });
        assert_eq!(
            bytes, 0,
            "16 sweeps over a steady body set asked the allocator for {bytes} bytes"
        );
    }

    #[test]
    fn exactly_tangent_spheres_are_a_candidate_pair() {
        const RADIUS: f32 = 0.5;
        let mut world = World::new(EuclideanR3);
        let anchor = world.push_body(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, RADIUS, 1.0).unwrap());
        let tangent = world.push_body(sphere_body_r3(Vec3::X, Vec3::ZERO, RADIUS, 1.0).unwrap());
        let separated = world.push_body(
            sphere_body_r3(
                Vec3::new(-(1.0 + f32::EPSILON), 0.0, 0.0),
                Vec3::ZERO,
                RADIUS,
                1.0,
            )
            .unwrap(),
        );

        let position = |id: BodyId| {
            let dense = world.bodies.dense_index(id).expect("nothing despawned");
            world.bodies[dense].position
        };
        assert_eq!(
            world.space.distance(position(anchor), position(tangent)),
            RADIUS + RADIUS,
            "the fixture is not exactly tangent, so it cannot state the boundary"
        );
        assert!(
            world.space.distance(position(anchor), position(separated)) > RADIUS + RADIUS,
            "the separated sphere is not past the boundary"
        );

        assert_eq!(world.broadphase(), vec![canonical_pair(anchor, tangent)]);
    }

    #[test]
    fn coincident_point_colliders_are_a_candidate_pair() {
        let mut world = World::new(EuclideanR3);
        let a = world.push_body(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 0.0, 1.0).unwrap());
        let b = world.push_body(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 0.0, 1.0).unwrap());
        assert_eq!(world.broadphase(), vec![canonical_pair(a, b)]);
    }

    #[test]
    fn bounding_radius_contains_every_posed_vertex_of_its_collider() {
        let half_extents = Vec3::new(0.5, 1.25, 0.25);
        let vertices = crate::euclidean_r3::box_vertices(half_extents);
        let radius = bounding_radius(&Collider::ConvexPolytope3D {
            vertices: vertices.clone(),
        });
        assert_eq!(radius, half_extents.length());

        let rotation = glam::Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7);
        for v in &vertices {
            let posed = rotation * *v;
            assert!(
                posed.length() <= radius + 1e-6,
                "posed vertex {posed} escaped the bounding radius {radius}"
            );
        }

        assert_eq!(bounding_radius(&Collider::sphere_at_origin(0.75)), 0.75);
        assert_eq!(
            bounding_radius(&Collider::HalfSpace {
                normal: Vec3::Y,
                offset: 0.0
            }),
            f32::INFINITY,
            "a half-space is unbounded and must never be culled"
        );
    }

    const ISLAND_SETTLE_STEPS: usize = 400;
    const ISLAND_COLUMN_PITCH: f32 = 4.0;
    const ISLAND_COLUMNS: usize = 6;
    const ISLAND_COLUMN_HEIGHT: usize = 3;
    const ISLAND_PINNED_COLUMN: usize = 2;

    fn settled_columns(seed: u64) -> World<EuclideanR3> {
        const GAP: f32 = 0.05;
        const PINNED_TOUCH: f32 = 0.01;

        let mut rng = Xorshift::new(seed);
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.gravity = Some(Vec3::new(0.0, GRAVITY_Y, 0.0));
        world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());

        let mut columns: Vec<Vec<BodyId>> = Vec::with_capacity(ISLAND_COLUMNS);
        for column in 0..ISLAND_COLUMNS {
            let x = column as f32 * ISLAND_COLUMN_PITCH + rng.range(-0.5, 0.5);
            let radius = rng.range(0.3, 0.6);
            let mut ids = Vec::with_capacity(ISLAND_COLUMN_HEIGHT);
            for level in 0..ISLAND_COLUMN_HEIGHT {
                let pinned = column == ISLAND_PINNED_COLUMN && level == 1;
                let y = if pinned {
                    3.0 * radius - PINNED_TOUCH
                } else {
                    radius + GAP + level as f32 * (2.0 * radius + GAP)
                };
                let id = world.push_body(
                    sphere_body_r3(Vec3::new(x, y, 0.0), Vec3::ZERO, radius, 1.0).unwrap(),
                );
                world.bodies[id].restitution = 0.0;
                if pinned {
                    assert!(world.bodies[id].set_mass(0.0));
                }
                ids.push(id);
            }
            columns.push(ids);
        }

        for column in columns.iter().take(2) {
            assert!(world.despawn_body(column[0]));
        }
        for _ in 0..ISLAND_SETTLE_STEPS {
            world.step(1.0 / 240.0);
        }
        world
    }

    #[test]
    fn island_ids_are_the_lowest_body_id_not_the_lowest_storage_position() {
        let mut orders_disagreed = 0usize;
        for seed in PERMUTATION_SEEDS {
            let mut world = settled_columns(seed);
            for step in 0..8 {
                world.step(1.0 / 240.0);
                let islands = world.islands();
                for island in &islands {
                    let lowest_handle = island.bodies.iter().copied().min();
                    assert_eq!(
                        Some(island.id),
                        lowest_handle,
                        "seed {seed} step {step}: island {:?} is not named by its \
                         lowest handle",
                        island.id
                    );
                    let lowest_stored =
                        island.bodies.iter().copied().min_by_key(|&id| {
                            world.bodies.dense_index(id).expect(STALE_MANIFOLD_BODY)
                        });
                    if lowest_stored != lowest_handle {
                        orders_disagreed += 1;
                    }
                }
                assert!(
                    islands.windows(2).all(|w| w[0].id < w[1].id),
                    "seed {seed} step {step}: islands are not strictly ascending in id"
                );
            }
        }
        assert!(
            orders_disagreed > 0,
            "no island ever held a body whose handle order disagreed with its \
             storage order, so the two labelling rules were never told apart"
        );
    }

    #[test]
    fn a_static_body_joins_no_island_and_merges_none() {
        for seed in PERMUTATION_SEEDS {
            let world = settled_columns(seed);
            let pinned = world
                .bodies
                .iter()
                .position(|body| {
                    body.inv_mass() == 0.0 && matches!(body.collider(), Collider::Sphere { .. })
                })
                .map(|dense| world.bodies.id_at(dense))
                .expect("the fixture lost its static sphere");

            let neighbours: Vec<BodyId> = world
                .manifolds
                .keys()
                .filter_map(|&(a, b)| match (a == pinned, b == pinned) {
                    (true, false) => Some(b),
                    (false, true) => Some(a),
                    _ => None,
                })
                .collect();
            assert_eq!(
                neighbours.len(),
                2,
                "seed {seed}: the static sphere touches {} bodies, so it is \
                 not wedged between two",
                neighbours.len()
            );

            let islands = world.islands();
            let island_of = |id: BodyId| {
                islands
                    .iter()
                    .find(|island| island.bodies.contains(&id))
                    .map(|island| island.id)
            };
            assert!(
                island_of(pinned).is_none(),
                "seed {seed}: a static body joined an island"
            );
            assert_ne!(
                island_of(neighbours[0]),
                island_of(neighbours[1]),
                "seed {seed}: two bodies that meet only through a static one \
                 were merged into one island"
            );
        }
    }

    fn flood_fill_islands(world: &World<EuclideanR3>) -> Vec<Island> {
        let dynamic = |id: BodyId| world.bodies[id].inv_mass() != 0.0;
        let mut adjacency: BTreeMap<BodyId, Vec<BodyId>> = BTreeMap::new();
        for &(a, b) in world.manifolds.keys() {
            for id in [a, b].into_iter().filter(|&id| dynamic(id)) {
                adjacency.entry(id).or_default();
            }
            if dynamic(a) && dynamic(b) {
                adjacency.entry(a).or_default().push(b);
                adjacency.entry(b).or_default().push(a);
            }
        }

        let mut seen: BTreeSet<BodyId> = BTreeSet::new();
        let mut islands = Vec::new();
        for &seed in adjacency.keys() {
            if !seen.insert(seed) {
                continue;
            }
            let mut bodies = vec![seed];
            let mut frontier = vec![seed];
            while let Some(body) = frontier.pop() {
                for &next in &adjacency[&body] {
                    if seen.insert(next) {
                        bodies.push(next);
                        frontier.push(next);
                    }
                }
            }
            bodies.sort_unstable();
            let constraints = world
                .manifolds
                .keys()
                .copied()
                .filter(|&(a, b)| {
                    bodies.binary_search(&a).is_ok() || bodies.binary_search(&b).is_ok()
                })
                .collect();
            islands.push(Island {
                id: bodies[0],
                bodies,
                constraints,
            });
        }
        islands.sort_unstable_by_key(|island| island.id);
        islands
    }

    #[test]
    fn islands_match_a_flood_fill_of_the_contact_graph() {
        let mut ever_multi_body = false;
        for seed in PERMUTATION_SEEDS {
            let mut world = settled_columns(seed);
            for step in 0..8 {
                world.step(1.0 / 240.0);
                let islands = world.islands();
                ever_multi_body |= islands.iter().any(|island| island.bodies.len() > 1);
                assert_eq!(
                    islands,
                    flood_fill_islands(&world),
                    "seed {seed} step {step}: union-find disagreed with the flood fill"
                );
            }
        }
        assert!(
            ever_multi_body,
            "no island ever held two bodies, so the comparison never covered a union"
        );
    }

    #[test]
    fn a_single_island_solves_in_the_global_ascending_key_order() {
        let world = settled_sphere_stack(1.0 / 240.0, 200);
        let islands = world.islands();
        assert_eq!(
            islands.len(),
            1,
            "the stack is not one island, so this fixture cannot state the \
             single-island case"
        );

        let ascending: Vec<PairKey> = world.manifolds.keys().copied().collect();
        assert!(ascending.len() > 1, "too few constraints to be ordered");
        assert_eq!(
            constraint_order(&world),
            ascending,
            "grouping moved a constraint in a world with a single island"
        );
        assert_eq!(islands[0].constraints, ascending);
        assert_eq!(
            islands[0].bodies.len(),
            3,
            "the island should hold the three spheres and not the static floor"
        );
    }

    #[test]
    fn constraint_buffer_runs_island_by_island() {
        let mut world = multi_island_world();
        for _ in 0..MULTI_ISLAND_STEPS {
            world.step(MULTI_ISLAND_DT);
        }

        let islands = world.islands();
        assert_eq!(
            islands.len(),
            3,
            "the groups share only the static floor, so they are three islands"
        );
        let grouped: Vec<PairKey> = islands
            .iter()
            .flat_map(|island| island.constraints.iter().copied())
            .collect();
        assert_eq!(
            constraint_order(&world),
            grouped,
            "the solved buffer is not the islands in order"
        );

        let ascending: Vec<PairKey> = world.manifolds.keys().copied().collect();
        assert_ne!(
            grouped, ascending,
            "the fixture's islands happen to be contiguous in ascending key \
             order, so it cannot show that grouping reorders anything"
        );
    }

    #[test]
    fn island_grouping_allocates_nothing_after_the_first_pass() {
        let mut world = settled_columns(PERMUTATION_SEEDS[0]);
        for _ in 0..2 {
            world.collect_constraints();
        }
        assert!(world.constraints.len() > 1);

        let bytes = alloc_probe::bytes_allocated_by(|| {
            for _ in 0..16 {
                world.collect_constraints();
            }
        });
        assert_eq!(
            bytes, 0,
            "16 island passes over a steady contact set asked the allocator for \
             {bytes} bytes"
        );
    }

    #[test]
    fn manifold_update_allocates_nothing_after_the_first_pass() {
        let mut world = settled_columns(PERMUTATION_SEEDS[0]);
        for _ in 0..2 {
            world.update_manifolds();
        }
        assert!(
            world.manifolds.len() > 1,
            "the fixture holds too few contacts to exercise the eviction pass"
        );

        let bytes = alloc_probe::bytes_allocated_by(|| {
            for _ in 0..16 {
                world.update_manifolds();
            }
        });
        assert_eq!(
            bytes, 0,
            "16 manifold passes over a steady contact set asked the allocator \
             for {bytes} bytes"
        );
    }

    #[test]
    fn r4_wall_bias_pushes_toward_the_near_face() {
        use crate::euclidean_r4::{sphere_body_r4, tesseract_vertices};
        use glam::Vec4;
        for x in [-0.092, -0.09, -0.088] {
            let mut world = World::new(EuclideanR4);
            crate::euclidean_r4::register_default_narrowphase(&mut world.narrowphase);
            world.push_body(
                RigidBody::fixed(
                    Vec4::ZERO,
                    Collider::ConvexPolytope4D {
                        vertices: tesseract_vertices(2.0)
                            .into_iter()
                            .map(|v| v * Vec4::new(0.05, 2.0, 2.0, 2.0))
                            .collect(),
                    },
                    1.0,
                    &EuclideanR4,
                )
                .unwrap(),
            );
            let ball = world.push_body(
                sphere_body_r4(Vec4::new(x, 0.0, 0.0, 0.0), Vec4::ZERO, 0.1, 1.0).unwrap(),
            );
            world.step(1.0 / 240.0);
            assert!(world.bodies[ball].velocity.x < 0.0, "x={x}");
        }
    }
}
