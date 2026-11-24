use std::collections::{BTreeMap, HashMap};

use loam_time::par::{self, Executor};
use loam_time::StateHash;

use crate::body::{BodyArena, BodyDef, BodyId, RigidBody};
use crate::collider::Collider;
use crate::dirty::{DirtyBodies, DirtyDrain};
use crate::edit::EditError;
use crate::field_contact::FieldNarrowphase;
use crate::geometry::GeometryStore;
use crate::integrator::{integrate_body, PhysicsSpace, VectorOps};
use crate::manifold::{Manifold, DEFAULT_PGS_ITERS, RESTITUTION_THRESHOLD};
use crate::narrowphase::Narrowphase;
use crate::response::Contact;
use crate::state::WorldState;

mod broadphase;
mod islands;
mod solve;

use broadphase::Broadphase;
use islands::Islands;
use solve::IslandSolver;
pub use solve::ISLANDS_PER_SOLVE_WORKER;

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

const WAKE_APPROACH_SPEED: f32 = 0.01;

/// Measured with scoped threads; box pairs gain from 2560 pairs, sphere pairs break even.
pub const PAIRS_PER_NARROWPHASE_WORKER: usize = 1024;

const STALE_CONSTRAINT_KEY: &str = "constraint buffer outlived its manifold";
const STALE_MANIFOLD_BODY: &str = "manifold key names a body that is gone";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepCounters {
    pub index_visits: u32,
    pub distance_evals: u32,
    pub candidates: u32,
    pub contacts: u32,
}

#[derive(Clone, Copy)]
struct ConstraintUnit {
    island: BodyId,
    key: PairKey,
    dense: (usize, usize),
}

#[cfg_attr(feature = "persist", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FieldId(u32);

struct FieldEntry {
    anchor: BodyId,
    field: Box<dyn loam_shape::field::DistanceField>,
}

pub struct World<S: PhysicsSpace> {
    space: S,
    pub(crate) bodies: BodyArena<S>,
    gravity: Option<S::Vector>,
    pub narrowphase: Narrowphase<S>,
    pub field_narrowphase: FieldNarrowphase<S>,
    fields: Vec<FieldEntry>,
    field_bindings: Vec<(BodyId, FieldId)>,
    geometry: GeometryStore,
    pub(crate) manifolds: HashMap<PairKey, Manifold<S>>,
    manifold_order: Vec<PairKey>,
    manifold_pool: Vec<Manifold<S>>,
    pending_contacts: Vec<(PairKey, Contact<S>)>,
    pair_contacts: Vec<(PairKey, Option<Contact<S>>)>,
    pgs_iters: usize,
    time: f32,
    dirty: DirtyBodies,
    pair_order: Vec<PairKey>,
    constraints: Vec<ConstraintUnit>,
    touched_pairs: Vec<PairKey>,
    counters: StepCounters,
    broadphase: Broadphase,
    islands: Islands,
    solver: IslandSolver<S>,
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    const fn assert_no_allocation<T: Copy>() {}
    #[cfg(feature = "r2")]
    {
        assert_send_sync::<World<loam_math::EuclideanR2>>();
        assert_no_allocation::<RigidBody<loam_math::EuclideanR2>>();
    }
    #[cfg(feature = "r3")]
    {
        assert_send_sync::<World<loam_math::EuclideanR3>>();
        assert_no_allocation::<RigidBody<loam_math::EuclideanR3>>();
    }
    #[cfg(feature = "r4")]
    {
        assert_send_sync::<World<loam_math::EuclideanR4>>();
        assert_no_allocation::<RigidBody<loam_math::EuclideanR4>>();
    }
};

impl<S: PhysicsSpace> World<S> {
    pub fn new(space: S) -> Self {
        Self {
            space,
            bodies: BodyArena::new(),
            gravity: None,
            narrowphase: Narrowphase::new(),
            field_narrowphase: FieldNarrowphase::new(),
            fields: Vec::new(),
            field_bindings: Vec::new(),
            geometry: GeometryStore::default(),
            manifolds: HashMap::new(),
            manifold_order: Vec::new(),
            manifold_pool: Vec::new(),
            pending_contacts: Vec::new(),
            pair_contacts: Vec::new(),
            pgs_iters: DEFAULT_PGS_ITERS,
            time: 0.0,
            dirty: DirtyBodies::default(),
            pair_order: Vec::new(),
            constraints: Vec::new(),
            touched_pairs: Vec::new(),
            counters: StepCounters::default(),
            broadphase: Broadphase::default(),
            islands: Islands::default(),
            solver: IslandSolver::default(),
        }
    }

    /// From the last `step` or `broadphase_into`.
    pub fn counters(&self) -> StepCounters {
        self.counters
    }

    pub fn bodies(&self) -> &BodyArena<S> {
        &self.bodies
    }

    pub fn space(&self) -> &S {
        &self.space
    }

    pub fn gravity(&self) -> Option<S::Vector> {
        self.gravity
    }

    pub fn set_gravity(&mut self, gravity: Option<S::Vector>) -> Result<(), EditError> {
        if gravity.is_some_and(|gravity| !self.space.valid_vector(gravity)) {
            return Err(EditError::InvalidGravity);
        }
        self.gravity = gravity;
        Ok(())
    }

    pub fn solver_iterations(&self) -> usize {
        self.pgs_iters
    }

    pub fn set_solver_iterations(&mut self, iterations: usize) {
        self.pgs_iters = iterations;
    }

    pub fn time(&self) -> f32 {
        self.time
    }

    pub fn body(&self, id: BodyId) -> Option<&RigidBody<S>> {
        self.bodies.get(id)
    }

    pub fn manifolds(&self) -> impl ExactSizeIterator<Item = (&PairKey, &Manifold<S>)> {
        self.manifold_order
            .iter()
            .map(|key| (key, &self.manifolds[key]))
    }

    pub fn push_body(&mut self, body: BodyDef<S>) -> BodyId {
        let row = body.into_row(&mut self.geometry, &self.space);
        let id = self.bodies.spawn(row);
        self.dirty.mark(id);
        id
    }

    /// `anchor` must be static; it leaves every collision group so the broadphase never pairs it, and it owns the field's contacts.
    pub fn insert_field(
        &mut self,
        anchor: BodyId,
        field: Box<dyn loam_shape::field::DistanceField>,
    ) -> Result<FieldId, EditError> {
        let Some(body) = self.bodies.get_mut(anchor) else {
            return Err(EditError::StaleHandle);
        };
        if !body.is_static() {
            return Err(EditError::DynamicFieldAnchor);
        }
        body.collision_group = 0;
        body.collision_mask = 0;
        self.drop_contacts_of(anchor);
        let id = FieldId(self.fields.len() as u32);
        self.fields.push(FieldEntry { anchor, field });
        Ok(id)
    }

    /// Every step queries the field for `body`; a contact lands in the manifold keyed by the body and the field's anchor.
    pub fn bind_field(&mut self, body: BodyId, field: FieldId) -> Result<(), EditError> {
        let Some(entry) = self.fields.get(field.0 as usize) else {
            return Err(EditError::StaleHandle);
        };
        if entry.anchor == body {
            return Err(EditError::AnchorBindsOwnField);
        }
        if self.bodies.get(body).is_none() {
            return Err(EditError::StaleHandle);
        }
        if let Err(at) = self.field_bindings.binary_search(&(body, field)) {
            self.field_bindings.insert(at, (body, field));
        }
        Ok(())
    }

    pub fn field_bindings(&self) -> &[(BodyId, FieldId)] {
        &self.field_bindings
    }

    pub fn geometry(&self) -> &GeometryStore {
        &self.geometry
    }

    /// Pops the newest of the last two released shapes so its buffer can be reused.
    pub fn reclaim_geometry(&mut self) -> Option<Collider> {
        self.geometry.take_released()
    }

    pub fn collider(&self, body: &RigidBody<S>) -> Option<&Collider> {
        self.geometry.get(body.collider())
    }

    /// Also removes every manifold the body takes part in.
    pub fn despawn_body(&mut self, id: BodyId) -> Result<(), EditError> {
        if self.fields.iter().any(|field| field.anchor == id) {
            return Err(EditError::FieldAnchorRemoval);
        }
        let Some(removed) = self.bodies.despawn(id) else {
            return Err(EditError::StaleHandle);
        };
        self.field_bindings.retain(|&(body, _)| body != id);
        self.geometry.release(removed.collider());
        self.drop_contacts_of(id);
        self.dirty.forget(id);
        Ok(())
    }

    /// Drops the body's contacts.
    pub fn set_pose(
        &mut self,
        id: BodyId,
        position: S::Point,
        orientation: S::Iso,
    ) -> Result<(), EditError> {
        if self.bodies.get(id).is_none() {
            return Err(EditError::StaleHandle);
        }
        if !self.space.valid_point(position) || !self.space.valid_orientation(orientation) {
            return Err(EditError::NotFinite);
        }
        let body = &mut self.bodies[id];
        body.position = position;
        body.orientation = orientation;
        body.wake();
        self.drop_contacts_of(id);
        self.dirty.mark(id);
        Ok(())
    }

    pub fn set_velocity(
        &mut self,
        id: BodyId,
        velocity: S::Vector,
        angular_velocity: S::AngVel,
    ) -> Result<(), EditError> {
        if self.bodies.get(id).is_none() {
            return Err(EditError::StaleHandle);
        }
        if !self.space.valid_vector(velocity)
            || !self.space.valid_angular_velocity(angular_velocity)
        {
            return Err(EditError::NotFinite);
        }
        let body = &mut self.bodies[id];
        body.velocity = velocity;
        body.angular_velocity = angular_velocity;
        body.wake();
        self.dirty.mark(id);
        Ok(())
    }

    pub fn set_restitution(&mut self, id: BodyId, restitution: f32) -> Result<(), EditError> {
        let Some(body) = self.bodies.get_mut(id) else {
            return Err(EditError::StaleHandle);
        };
        if !restitution.is_finite() || restitution < 0.0 {
            return Err(EditError::InvalidRestitution);
        }
        body.restitution = restitution;
        self.drop_contacts_of(id);
        Ok(())
    }

    pub fn set_collision_filter(
        &mut self,
        id: BodyId,
        group: u32,
        mask: u32,
    ) -> Result<(), EditError> {
        if self.fields.iter().any(|field| field.anchor == id) && (group != 0 || mask != 0) {
            return Err(EditError::InvalidFieldBinding);
        }
        let Some(body) = self.bodies.get_mut(id) else {
            return Err(EditError::StaleHandle);
        };
        body.collision_group = group;
        body.collision_mask = mask;
        self.drop_contacts_of(id);
        Ok(())
    }

    /// Drops the body's contacts.
    pub fn set_mass_properties(
        &mut self,
        id: BodyId,
        mass: f32,
        inertia: S::Inertia,
    ) -> Result<(), EditError> {
        if mass > 0.0 && self.fields.iter().any(|field| field.anchor == id) {
            return Err(EditError::DynamicFieldAnchor);
        }
        if self.bodies.get(id).is_none() {
            return Err(EditError::StaleHandle);
        }
        if !self.space.valid_inertia(inertia) {
            return Err(EditError::InvalidInertia);
        }
        self.bodies[id].set_mass_properties(mass, inertia)?;
        self.drop_contacts_of(id);
        self.dirty.mark(id);
        Ok(())
    }

    /// Drops the body's contacts.
    pub fn set_collider(
        &mut self,
        id: BodyId,
        collider: Collider,
        inertia: S::Inertia,
    ) -> Result<(), EditError> {
        if self.bodies.get(id).is_none() {
            self.geometry.stash(collider);
            return Err(EditError::StaleHandle);
        }
        if !self.space.valid_inertia(inertia) {
            self.geometry.stash(collider);
            return Err(EditError::InvalidInertia);
        }
        self.bodies[id].replace_collider(&mut self.geometry, &self.space, collider, inertia)?;
        self.drop_contacts_of(id);
        self.dirty.mark(id);
        Ok(())
    }

    pub fn apply_impulse(&mut self, id: BodyId, impulse: S::Vector) -> Result<(), EditError> {
        if self.bodies.get(id).is_none() {
            return Err(EditError::StaleHandle);
        }
        if !self.space.valid_vector(impulse) {
            return Err(EditError::NotFinite);
        }
        self.bodies[id].apply_impulse(impulse);
        self.dirty.mark(id);
        Ok(())
    }

    pub fn apply_impulse_at_point(
        &mut self,
        id: BodyId,
        impulse: S::Vector,
        point: S::Point,
    ) -> Result<(), EditError> {
        if self.bodies.get(id).is_none() {
            return Err(EditError::StaleHandle);
        }
        if !self.space.valid_vector(impulse) || !self.space.valid_point(point) {
            return Err(EditError::NotFinite);
        }
        let space = &self.space;
        self.bodies[id].apply_impulse_at_point(space, impulse, point);
        self.dirty.mark(id);
        Ok(())
    }

    pub fn sleep_body(&mut self, id: BodyId) -> Result<(), EditError> {
        let Some(body) = self.bodies.get_mut(id) else {
            return Err(EditError::StaleHandle);
        };
        body.sleep();
        self.dirty.mark(id);
        Ok(())
    }

    pub fn wake_body(&mut self, id: BodyId) -> Result<(), EditError> {
        let Some(body) = self.bodies.get_mut(id) else {
            return Err(EditError::StaleHandle);
        };
        body.wake();
        self.dirty.mark(id);
        Ok(())
    }

    fn drop_contacts_of(&mut self, id: BodyId) {
        let mut write = 0;
        for read in 0..self.manifold_order.len() {
            let key = self.manifold_order[read];
            if key.0 == id || key.1 == id {
                if let Some(manifold) = self.manifolds.remove(&key) {
                    self.manifold_pool.push(manifold);
                }
            } else {
                self.manifold_order[write] = key;
                write += 1;
            }
        }
        self.manifold_order.truncate(write);
    }

    /// Yields each body spawned, integrated, edited, or restored since the last drain, skipping despawned ones.
    pub fn drain_dirty(&mut self) -> DirtyDrain<'_, S> {
        self.dirty.drain(&self.bodies)
    }

    pub fn snapshot(&self) -> WorldState<S> {
        WorldState {
            bodies: self.bodies.clone(),
            geometry: self.geometry.clone(),
            manifolds: self
                .manifolds()
                .map(|(&key, manifold)| (key, manifold.clone()))
                .collect(),
            time: self.time,
            field_bindings: self.field_bindings.clone(),
            field_anchors: self.fields.iter().map(|entry| entry.anchor).collect(),
            registrations: self.narrowphase.registrations().to_vec(),
            field_registrations: self.field_narrowphase.registrations().to_vec(),
        }
    }

    pub fn check_restore(&self, state: &WorldState<S>) -> Result<(), EditError>
    where
        S::AngVel: PartialEq,
    {
        if self
            .gravity
            .is_some_and(|gravity| !self.space.valid_vector(gravity))
        {
            return Err(EditError::InvalidGravity);
        }
        let field_registrations = self.field_narrowphase.registrations();
        if self.narrowphase.registrations() != state.registrations
            || field_registrations.len() != state.field_registrations.len()
            || state
                .field_registrations
                .iter()
                .any(|kind| !field_registrations.contains(kind))
        {
            return Err(EditError::RegistrationMismatch);
        }
        if self.fields.len() != state.field_anchors.len()
            || self
                .fields
                .iter()
                .zip(&state.field_anchors)
                .any(|(entry, &anchor)| entry.anchor != anchor)
        {
            return Err(EditError::FieldMismatch);
        }
        if !state.time.is_finite() || state.time < 0.0 {
            return Err(EditError::InvalidTime);
        }
        state.bodies.validate(&self.space, &state.geometry)?;
        for (&anchor, entry) in state.field_anchors.iter().zip(&self.fields) {
            let Some(body) = state.bodies.get(anchor) else {
                return Err(EditError::InvalidFieldBinding);
            };
            if entry.anchor != anchor
                || !body.is_static()
                || body.collision_group != 0
                || body.collision_mask != 0
            {
                return Err(EditError::InvalidFieldBinding);
            }
        }
        if state
            .field_bindings
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(EditError::InvalidFieldBinding);
        }
        for &(body, field) in &state.field_bindings {
            let Some(entry) = self.fields.get(field.0 as usize) else {
                return Err(EditError::InvalidFieldBinding);
            };
            if body == entry.anchor || state.bodies.get(body).is_none() {
                return Err(EditError::InvalidFieldBinding);
            }
        }
        for (&key, manifold) in &state.manifolds {
            let (Some(body_a), Some(body_b)) = (state.bodies.get(key.0), state.bodies.get(key.1))
            else {
                return Err(EditError::InvalidManifold);
            };
            if key.0 >= key.1 || manifold.body_a != key.0 || manifold.body_b != key.1 {
                return Err(EditError::InvalidManifold);
            }
            let field_pair = state.field_bindings.iter().any(|&(body, field)| {
                let anchor = self.fields[field.0 as usize].anchor;
                canonical_pair(body, anchor) == key
                    && state.bodies.get(body).is_some_and(|body| {
                        state.field_registrations.contains(&body.collider().kind())
                    })
            });
            let registered_pair = self.narrowphase.registrations().iter().any(|&(a, b)| {
                let pair = (body_a.collider().kind(), body_b.collider().kind());
                (a, b) == pair || (b, a) == pair
            });
            let body_pair = registered_pair
                && !(body_a.is_static() && body_b.is_static())
                && body_a.collision_mask & body_b.collision_group != 0
                && body_b.collision_mask & body_a.collision_group != 0;
            if (!field_pair && !body_pair) || !manifold.is_valid(&self.space, body_a, body_b) {
                return Err(EditError::InvalidManifold);
            }
        }
        Ok(())
    }

    /// Refuses, unchanged, whatever `check_restore` refuses.
    pub fn restore(&mut self, state: &WorldState<S>) -> Result<(), EditError>
    where
        S::AngVel: PartialEq,
    {
        self.check_restore(state)?;
        self.bodies = state.bodies.clone();
        self.geometry = state.geometry.clone();
        let required_pool = state
            .manifolds
            .len()
            .max(self.manifold_pool.len() + self.manifold_order.len());
        if self.manifold_pool.capacity() < required_pool {
            self.manifold_pool
                .reserve_exact(required_pool - self.manifold_pool.len());
        }
        for key in self.manifold_order.drain(..) {
            if let Some(manifold) = self.manifolds.remove(&key) {
                self.manifold_pool.push(manifold);
            }
        }
        for (&key, manifold) in &state.manifolds {
            let restored = if let Some(mut restored) = self.manifold_pool.pop() {
                restored.clone_from(manifold);
                restored
            } else {
                manifold.clone()
            };
            self.manifolds.insert(key, restored);
            self.manifold_order.push(key);
        }
        self.time = state.time;
        self.field_bindings.clear();
        self.field_bindings.extend_from_slice(&state.field_bindings);
        self.dirty.mark_every(&self.bodies);
        self.pair_order.clear();
        self.constraints.clear();
        self.touched_pairs.clear();
        self.pending_contacts.clear();
        self.broadphase.clear();
        self.islands.clear();
        self.solver.clear();
        Ok(())
    }

    /// `dt` is in seconds, and the result is the same for any `executor`.
    pub fn step(&mut self, dt: f32, executor: &dyn Executor) -> Result<(), EditError> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(EditError::InvalidTimeStep);
        }
        let next_time = self.time + dt;
        if !next_time.is_finite() {
            return Err(EditError::InvalidTimeStep);
        }
        self.counters = StepCounters::default();
        // Test approach before gravity adds support-load velocity.
        self.wake_approaching_contacts();
        self.apply_forces(dt);
        self.integrate(dt);
        self.update_manifolds(executor);
        self.collect_constraints();
        self.prepare_solve();
        self.warm_start();
        self.solve(dt, executor);

        self.time = next_time;
        Ok(())
    }

    fn apply_forces(&mut self, dt: f32) {
        let Some(acceleration) = self.gravity else {
            return;
        };
        for body in self.bodies.iter_mut() {
            if body.inv_mass() != 0.0 {
                body.velocity = body.velocity + acceleration * dt;
            }
        }
    }

    fn integrate(&mut self, dt: f32) {
        for i in 0..self.bodies.len() {
            if self.bodies[i].inv_mass() == 0.0 {
                continue;
            }
            integrate_body(&self.space, &mut self.bodies[i], dt);
            let id = self.bodies.id_at(i);
            self.dirty.mark(id);
        }
    }

    fn update_manifolds(&mut self, executor: &dyn Executor) {
        self.broadphase.fill(
            &self.bodies,
            &self.geometry,
            &self.space,
            &mut self.pair_order,
            &mut self.counters,
        );
        self.touched_pairs.clear();
        self.pending_contacts.clear();
        self.pair_contacts.clear();
        self.pair_contacts
            .extend(self.pair_order.iter().map(|&key| (key, None)));
        self.test_pairs(executor);

        for &(key, found) in &self.pair_contacts {
            let (i, j) = dense_pair(&self.bodies, key);
            let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
            if let Some(manifold) = self.manifolds.get_mut(&key) {
                manifold.refresh(&self.space, a, b);
            }
            let Some(contact) = found else {
                continue;
            };
            if !self.manifolds.contains_key(&key)
                && a.is_sleeping() != b.is_sleeping()
                && !a.is_static()
                && !b.is_static()
            {
                a.wake();
                b.wake();
            }
            self.pending_contacts.push((key, contact));
        }

        self.field_manifolds();

        self.touched_pairs
            .extend(self.pending_contacts.iter().map(|(key, _)| *key));
        self.touched_pairs.sort_unstable();
        self.touched_pairs.dedup();
        self.counters.contacts = self.pending_contacts.len() as u32;
        self.recycle_untouched_manifolds();
        for index in 0..self.pending_contacts.len() {
            let (key, contact) = self.pending_contacts[index];
            self.ensure_manifold(key, contact.restitution);
            let (i, j) = dense_pair(&self.bodies, key);
            let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
            self.manifolds
                .get_mut(&key)
                .expect(STALE_CONSTRAINT_KEY)
                .add_or_update(&self.space, a, b, contact);
        }
    }

    // Each pair writes only its own slot, so the merge that follows sees pair order for any thread count.
    fn test_pairs(&mut self, executor: &dyn Executor) {
        let test = |chunk: &mut [(PairKey, Option<Contact<S>>)]| {
            for (key, contact) in chunk {
                let a = self.bodies.get(key.0).expect(STALE_MANIFOLD_BODY);
                let b = self.bodies.get(key.1).expect(STALE_MANIFOLD_BODY);
                *contact = self.narrowphase.test(a, b, &self.geometry, &self.space);
            }
        };
        let slots = &mut self.pair_contacts;
        let workers = slots.len() / PAIRS_PER_NARROWPHASE_WORKER;
        if workers < 2 {
            test(slots);
        } else {
            let chunk = slots.len().div_ceil(workers);
            par::for_each_chunk(executor, slots, chunk, test);
        }
    }

    fn ensure_manifold(&mut self, key: PairKey, restitution: f32) {
        if self.manifolds.contains_key(&key) {
            return;
        }
        let manifold = if let Some(mut manifold) = self.manifold_pool.pop() {
            manifold.reset(key.0, key.1, restitution);
            manifold
        } else {
            let required = self.manifolds.len() + 1;
            if self.manifold_pool.capacity() < required {
                self.manifold_pool
                    .reserve_exact(required - self.manifold_pool.len());
            }
            Manifold::new(key.0, key.1, restitution)
        };
        self.manifolds.insert(key, manifold);
    }

    fn recycle_untouched_manifolds(&mut self) {
        for &key in &self.manifold_order {
            if self.touched_pairs.binary_search(&key).is_err() {
                if let Some(manifold) = self.manifolds.remove(&key) {
                    self.manifold_pool.push(manifold);
                }
            }
        }
        self.manifold_order.clear();
        self.manifold_order.extend_from_slice(&self.touched_pairs);
    }

    fn wake_approaching_contacts(&mut self) {
        for key in &self.manifold_order {
            let manifold = &self.manifolds[key];
            let (Some(a), Some(b)) = (
                self.bodies.get(manifold.body_a),
                self.bodies.get(manifold.body_b),
            ) else {
                continue;
            };
            if a.is_static() || b.is_static() || a.is_sleeping() == b.is_sleeping() {
                continue;
            }
            let approaching = manifold.points.iter().any(|point| {
                let relative = self.space.velocity_at_point(b, point.world_point)
                    - self.space.velocity_at_point(a, point.world_point);
                VectorOps::dot(relative, point.normal) < -WAKE_APPROACH_SPEED
            });
            if approaching {
                let id = if a.is_sleeping() {
                    manifold.body_a
                } else {
                    manifold.body_b
                };
                if let Some(body) = self.bodies.get_mut(id) {
                    body.wake();
                }
            }
        }
    }

    fn field_manifolds(&mut self) {
        for index in 0..self.field_bindings.len() {
            let (body, field) = self.field_bindings[index];
            let Some(entry) = self.fields.get(field.0 as usize) else {
                continue;
            };
            let anchor = entry.anchor;
            let (Some(row), Some(_)) = (self.bodies.get(body), self.bodies.get(anchor)) else {
                continue;
            };
            let Ok(found) =
                self.field_narrowphase
                    .test(row, &self.geometry, entry.field.as_ref(), &self.space)
            else {
                continue;
            };
            if !found.separation.is_finite() || !found.error.is_finite() || found.error < 0.0 {
                continue;
            }
            let certified_separation = (found.separation as f64 + found.error as f64).next_up();
            if certified_separation >= 0.0 || -certified_separation > f32::MAX as f64 {
                continue;
            }
            let mut penetration = (-certified_separation) as f32;
            if penetration as f64 > -certified_separation {
                penetration = penetration.next_down();
            }
            let key = canonical_pair(body, anchor);
            let normal = if key.0 == anchor {
                found.normal
            } else {
                -found.normal
            };
            let contact = Contact {
                normal,
                point: found.witness,
                penetration,
                restitution: (self.bodies[body].restitution + self.bodies[anchor].restitution)
                    * 0.5,
            };
            let (i, j) = dense_pair(&self.bodies, key);
            let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
            if let Some(manifold) = self.manifolds.get_mut(&key) {
                manifold.refresh(&self.space, a, b);
            }
            self.pending_contacts.push((key, contact));
        }
    }

    // Manifold membership must stay fixed until the solve ends.
    fn collect_constraints(&mut self) {
        self.islands
            .fill(&self.bodies, self.manifold_order.iter().copied());
        self.constraints.clear();
        self.constraints
            .extend(self.manifold_order.iter().map(|&key| {
                let dense = dense_pair(&self.bodies, key);
                ConstraintUnit {
                    island: self.islands.constraint_island(&self.bodies, dense),
                    key,
                    dense,
                }
            }));
        self.constraints
            .sort_unstable_by_key(|unit| (unit.island, unit.key));
    }

    // Restitution uses velocities from before the warm-start impulse.
    fn prepare_solve(&mut self) {
        for unit in &self.constraints {
            debug_assert_eq!(unit.dense, dense_pair(&self.bodies, unit.key));
            let (i, j) = unit.dense;
            let manifold = self
                .manifolds
                .get_mut(&unit.key)
                .expect(STALE_CONSTRAINT_KEY);
            let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
            for cp in &mut manifold.points {
                let v_rel = self.space.velocity_at_point(b, cp.world_point)
                    - self.space.velocity_at_point(a, cp.world_point);
                let v_n = VectorOps::dot(v_rel, cp.normal);

                cp.velocity_bias = if v_n < -RESTITUTION_THRESHOLD {
                    manifold.restitution * v_n
                } else {
                    0.0
                };

                cp.tangent_impulse = 0.0;
                cp.tangent_dir = VectorOps::zero();
            }
        }
    }

    fn warm_start(&mut self) {
        for unit in &self.constraints {
            debug_assert_eq!(unit.dense, dense_pair(&self.bodies, unit.key));
            let (i, j) = unit.dense;
            let manifold = self.manifolds.get(&unit.key).expect(STALE_CONSTRAINT_KEY);
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

    fn solve(&mut self, dt: f32, executor: &dyn Executor) {
        self.solver
            .gather(&self.bodies, &self.manifolds, &self.constraints);
        self.solver.solve(&self.space, self.pgs_iters, dt, executor);
        self.solver.scatter(&mut self.bodies, &mut self.manifolds);
    }

    /// Replaces `pairs` with the sorted candidates: bounding-ball overlaps under a certified bound, every masked non-static pair otherwise.
    pub fn broadphase_into(&mut self, pairs: &mut Vec<PairKey>) {
        self.counters = StepCounters::default();
        self.broadphase.fill(
            &self.bodies,
            &self.geometry,
            &self.space,
            pairs,
            &mut self.counters,
        );
    }

    /// Hashes contact keys, point counts, and normal impulses in key order.
    pub fn hash_contacts(&self, hash: &mut StateHash) {
        for (key, manifold) in self.manifolds() {
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
        let mut components = Islands::default();
        components.fill(&self.bodies, self.manifold_order.iter().copied());

        let mut by_id: BTreeMap<BodyId, Island> = BTreeMap::new();
        for &key in &self.manifold_order {
            if self.bodies[key.0].inv_mass() == 0.0 && self.bodies[key.1].inv_mass() == 0.0 {
                continue;
            }
            let id = components.constraint_island(&self.bodies, dense_pair(&self.bodies, key));
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
}

fn dense_pair<S: PhysicsSpace>(bodies: &BodyArena<S>, key: PairKey) -> (usize, usize) {
    (
        bodies.dense_index(key.0).expect(STALE_MANIFOLD_BODY),
        bodies.dense_index(key.1).expect(STALE_MANIFOLD_BODY),
    )
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

#[cfg(all(test, feature = "r2", feature = "r3", feature = "r4"))]
mod tests {
    use std::alloc::System;

    use loam_time::par::Sequential;

    use super::*;
    use crate::determinism_fixture::{
        record_flick_chamber_tape, replay_flick_chamber_tape, sample_body_r3, REPLAY_SEED,
    };
    use crate::euclidean_r3::{
        box_body, halfspace_body_r3, register_default_narrowphase, sphere_body_r3,
    };
    use glam::Vec3;
    use loam_math::{Bivector3, EuclideanR3};
    use loam_time::alloc::{bytes_allocated_by, CountingAllocator};
    use loam_time::Tape;

    const PERMUTATION_SEEDS: [u64; 4] = [1, 0x9e37_79b9_7f4a_7c15, 0xdead_beef_cafe_f00d, 424_242];

    #[global_allocator]
    static COUNTING_ALLOCATOR: CountingAllocator<System> = CountingAllocator::new(System);

    #[test]
    fn a_tape_replays_to_its_recorded_checkpoints_and_a_tampered_one_does_not() {
        let recorded = record_flick_chamber_tape(REPLAY_SEED);
        let decoded = Tape::decode(&recorded.encode()).expect("own encoding decodes");
        assert_eq!(decoded, recorded);
        assert_eq!(replay_flick_chamber_tape(&decoded), decoded.checkpoints());

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

    const SPHERE_RADIUS: f32 = 0.5;
    const GRAVITY_Y: f32 = -9.8;

    mod solver_contracts {
        use glam::Vec4;
        use loam_math::EuclideanR4;

        use super::*;
        use crate::euclidean_r4::{halfspace4_body_r4, sphere_body_r4};

        fn kinetic_energy<S: PhysicsSpace<Inertia = f32>>(body: &RigidBody<S>, omega: f32) -> f32
        where
            S::Vector: VectorOps,
        {
            0.5 * body.mass() * VectorOps::length_squared(body.velocity)
                + 0.5 * body.inertia * omega * omega
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
        {
            let energy_before = kinetic_energy(
                &world.bodies[faller],
                angular_speed(world.bodies[faller].angular_velocity),
            );
            let steps = (4.0 * REBOUND_GAP / (REBOUND_APPROACH * REBOUND_DT)).ceil() as usize;
            for _ in 0..steps {
                world.step(REBOUND_DT, &Sequential).unwrap();
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
        fn a_perfectly_elastic_rebound_conserves_kinetic_energy_in_r3_and_r4() {
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

        fn assert_friction_brakes_a_slide_toward_rolling<S>(
            mut world: World<S>,
            slider: BodyId,
            slide: S::Vector,
            up: S::Vector,
        ) where
            S: PhysicsSpace,
            S::Vector: VectorOps,
        {
            for _ in 0..SLIDE_STEPS {
                world.step(SLIDE_DT, &Sequential).unwrap();
            }

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
        fn friction_brakes_a_slide_and_spins_it_toward_rolling_in_r3_and_r4() {
            let mut world = World::new(EuclideanR3);
            register_default_narrowphase(&mut world.narrowphase);
            world
                .set_gravity(Some(Vec3::new(0.0, GRAVITY_Y, 0.0)))
                .unwrap();
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

            assert_friction_brakes_a_slide_toward_rolling(world, sphere, Vec3::X, Vec3::Y);

            let mut world = World::new(EuclideanR4);
            crate::euclidean_r4::register_default_narrowphase(&mut world.narrowphase);
            world
                .set_gravity(Some(Vec4::new(0.0, GRAVITY_Y, 0.0, 0.0)))
                .unwrap();
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

            assert_friction_brakes_a_slide_toward_rolling(world, sphere, Vec4::X, Vec4::Y);
        }
    }

    const ISLAND_X: [f32; 4] = [-4.0, 0.0, 4.0, 8.0];

    fn settled_islands(dt: f32, settle_steps: usize) -> (World<EuclideanR3>, BodyId, Vec<BodyId>) {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world
            .set_gravity(Some(Vec3::new(0.0, GRAVITY_Y, 0.0)))
            .unwrap();

        let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());
        world.bodies[floor].restitution = 0.0;
        let mut spheres = Vec::with_capacity(ISLAND_X.len());
        for x in ISLAND_X {
            let id = world.push_body(island_sphere(x));
            world.bodies[id].restitution = 0.0;
            spheres.push(id);
        }

        for _ in 0..settle_steps {
            world.step(dt, &Sequential).unwrap();
        }
        (world, floor, spheres)
    }

    fn island_sphere(x: f32) -> BodyDef<EuclideanR3> {
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
    fn an_unrelated_despawn_or_spawn_leaves_a_surviving_island_bit_identical() {
        let dt = 1.0 / 240.0;
        let settle_steps = 400;
        let (mut world, _, spheres) = settled_islands(dt, settle_steps);
        let (mut control, _, control_spheres) = settled_islands(dt, settle_steps);

        let doomed = spheres[1];
        let keeper = spheres[3];
        let keeper_position_before = world.bodies.dense_index(keeper).unwrap();

        assert!(world.despawn_body(doomed).is_ok());

        assert_ne!(
            world.bodies.dense_index(keeper).unwrap(),
            keeper_position_before,
            "despawn moved no surviving body, so this test never reached the compaction case"
        );

        for _ in 0..60 {
            world.step(dt, &Sequential).unwrap();
            control.step(dt, &Sequential).unwrap();
        }

        assert_eq!(
            body_state(&world, keeper),
            body_state(&control, control_spheres[3]),
            "an unrelated despawn perturbed a surviving island"
        );

        let (mut world, floor, spheres) = settled_islands(dt, settle_steps);
        let (mut control, _, control_spheres) = settled_islands(dt, settle_steps);

        let keeper = spheres[0];
        let newcomer = world.push_body(island_sphere(12.0));
        world.bodies[newcomer].restitution = 0.0;

        for _ in 0..60 {
            world.step(dt, &Sequential).unwrap();
            control.step(dt, &Sequential).unwrap();
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
    }

    #[test]
    fn a_despawn_leaves_no_contact_handle_or_dirty_row_for_the_body_that_takes_its_slot() {
        let dt = 1.0 / 240.0;
        let (mut world, floor, spheres) = settled_islands(dt, 400);
        let doomed = spheres[1];
        let names_doomed = |world: &World<EuclideanR3>| {
            world
                .manifolds
                .keys()
                .any(|&(a, b)| a == doomed || b == doomed)
        };
        assert!(world.manifolds.contains_key(&(floor, doomed)));
        assert!(world
            .set_velocity(doomed, Vec3::new(0.0, 3.0, 0.0), Bivector3::ZERO)
            .is_ok());

        assert!(world.despawn_body(doomed).is_ok());
        assert!(
            !names_doomed(&world),
            "a contact outlived the body it names"
        );
        assert_eq!(
            world.despawn_body(doomed),
            Err(EditError::StaleHandle),
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
        let published: Vec<BodyId> = world.drain_dirty().map(|(id, _)| id).collect();
        assert!(
            !published.contains(&doomed),
            "a removed row was published for mirroring"
        );
        assert!(
            published.contains(&reborn),
            "the row that took the removed slot was never published"
        );

        world.step(dt, &Sequential).unwrap();
        assert!(
            world.manifolds.contains_key(&(floor, reborn)),
            "the respawned body made no contact"
        );
        assert!(
            !names_doomed(&world),
            "a manifold keyed on the despawned body came back with the slot"
        );
    }

    const EDIT_DT: f32 = 1.0 / 240.0;
    const EDIT_SETTLE_STEPS: usize = 400;

    #[test]
    fn a_rejected_edit_gravity_or_time_step_leaves_the_world_untouched() {
        let (mut world, floor, spheres) = settled_islands(EDIT_DT, EDIT_SETTLE_STEPS);
        let id = spheres[0];
        let before = body_state(&world, id);
        let mass = world.bodies[id].mass();
        let inv_mass = world.bodies[id].inv_mass();
        let inertia = world.bodies[id].inertia;
        let orientation = world.bodies[id].orientation;
        let stale = BodyId::forge(u32::MAX, 0);
        let _ = world.drain_dirty().count();

        assert_eq!(
            world.set_mass_properties(id, f32::NAN, 1.0),
            Err(EditError::InvalidMass)
        );
        assert_eq!(
            world.set_mass_properties(id, 3.0, f32::NAN),
            Err(EditError::InvalidInertia)
        );
        assert_eq!(
            world.set_restitution(id, -1.0),
            Err(EditError::InvalidRestitution)
        );
        assert_eq!(
            world.set_collider(
                id,
                Collider::Box3 {
                    half_extents: Vec3::ONE
                },
                2.0
            ),
            Err(EditError::UnsupportedCollider)
        );
        assert_eq!(
            world.set_collider(id, Collider::sphere_at_origin(0.5), f32::INFINITY),
            Err(EditError::InvalidInertia)
        );
        assert_eq!(
            world.set_pose(id, Vec3::splat(f32::NAN), orientation),
            Err(EditError::NotFinite)
        );
        let spun = loam_math::Iso3 {
            rotation: glam::Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0),
            translation: Vec3::ZERO,
        };
        assert_eq!(
            world.set_pose(id, Vec3::ZERO, spun),
            Err(EditError::NotFinite)
        );
        let zero_rotation = loam_math::Iso3 {
            rotation: glam::Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
            translation: Vec3::ZERO,
        };
        assert_eq!(
            world.set_pose(id, Vec3::ZERO, zero_rotation),
            Err(EditError::NotFinite)
        );
        assert_eq!(
            world.set_velocity(id, Vec3::ZERO, Bivector3::new(f32::NAN, 0.0, 0.0)),
            Err(EditError::NotFinite)
        );
        assert_eq!(
            world.set_velocity(stale, Vec3::ZERO, Bivector3::ZERO),
            Err(EditError::StaleHandle)
        );
        let half_space = Collider::HalfSpace {
            normal: Vec3::Y,
            offset: 0.0,
        };
        assert!(
            world.space.supports_collider(half_space.kind()),
            "the space rejects half-spaces outright, so this cannot tell the two refusals apart"
        );
        assert_eq!(
            world.set_collider(id, half_space, 1.0),
            Err(EditError::DynamicHalfSpace)
        );
        assert_eq!(
            world.set_mass_properties(floor, 3.0, 1.0),
            Err(EditError::DynamicHalfSpace)
        );
        assert_eq!(world.bodies[floor].mass(), 0.0);
        assert_eq!(
            world.drain_dirty().count(),
            0,
            "a rejected edit published a pose change"
        );

        assert_eq!(body_state(&world, id), before);
        assert_eq!(world.bodies[id].mass(), mass);
        assert_eq!(world.bodies[id].inv_mass(), inv_mass);
        assert_eq!(world.bodies[id].inertia, inertia);

        let mut world4 = World::new(loam_math::EuclideanR4);
        let id4 = world4.push_body(
            crate::euclidean_r4::sphere_body_r4(glam::Vec4::ZERO, glam::Vec4::ZERO, 0.5, 1.0)
                .unwrap(),
        );
        let before4 = world4.bodies[id4].orientation;
        let scale = std::f32::consts::FRAC_1_SQRT_2;
        let invalid4 = loam_math::Iso4Flat {
            rotation: loam_math::Rotor4 {
                s: scale,
                xy: 0.0,
                xz: 0.0,
                xw: 0.0,
                yz: 0.0,
                yw: 0.0,
                zw: 0.0,
                xyzw: scale,
            },
            translation: glam::Vec4::ZERO,
        };
        assert_eq!(
            world4.set_pose(id4, glam::Vec4::ZERO, invalid4),
            Err(EditError::NotFinite)
        );
        assert_eq!(world4.bodies[id4].orientation, before4);

        let mut world = World::new(EuclideanR3);
        let body = world.push_body(sphere_body_r3(Vec3::X, Vec3::Y, 0.5, 1.0).unwrap());
        let before = world.state_hash(sample_body_r3);

        assert_eq!(
            world.set_gravity(Some(Vec3::splat(f32::NAN))),
            Err(EditError::InvalidGravity)
        );
        assert_eq!(world.gravity(), None);
        for dt in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
            assert_eq!(world.step(dt, &Sequential), Err(EditError::InvalidTimeStep));
            assert!(world.body(body).is_some());
            assert_eq!(world.state_hash(sample_body_r3), before);
            assert_eq!(world.time(), 0.0);
        }

        world.step(0.0, &Sequential).unwrap();
        assert!(world.body(body).is_some());
        assert_eq!(world.state_hash(sample_body_r3), before);
        assert_eq!(world.time(), 0.0);
    }

    #[test]
    fn the_dirty_drain_publishes_every_pose_change_and_no_sleeping_body_that_stayed_put() {
        let (mut world, _, spheres) = settled_islands(EDIT_DT, 4);
        let _ = world.drain_dirty().count();
        world.step(EDIT_DT, &Sequential).unwrap();
        let mut mirror: BTreeMap<BodyId, Vec3> = BTreeMap::new();
        for (id, body) in world.drain_dirty() {
            mirror.insert(id, body.position);
        }
        assert_eq!(mirror.len(), spheres.len());
        for id in &spheres {
            assert_eq!(mirror.get(id), Some(&world.bodies[*id].position));
        }

        let (mut world, _, spheres) = settled_islands(EDIT_DT, EDIT_SETTLE_STEPS);
        let id = spheres[0];
        assert!(world.sleep_body(id).is_ok());
        assert!(world.drain_dirty().count() > 0);

        world.step(EDIT_DT, &Sequential).unwrap();
        assert!(
            !world.drain_dirty().any(|(dirty, _)| dirty == id),
            "a sleeping body was published as if it had moved"
        );

        let orientation = world.bodies[id].orientation;
        let elsewhere = Vec3::new(40.0, 9.0, 0.0);
        assert!(world.set_pose(id, elsewhere, orientation).is_ok());
        assert!(
            world.drain_dirty().any(|(dirty, _)| dirty == id),
            "the teleport never reached the dirty set"
        );
        assert_eq!(world.bodies[id].position, elsewhere);

        let (mut world, _, spheres) = settled_islands(EDIT_DT, EDIT_SETTLE_STEPS);
        let id = spheres[0];
        assert!(world.sleep_body(id).is_ok());
        let settled = world.bodies[id].position;
        let state = world.snapshot();
        let orientation = world.bodies[id].orientation;
        assert!(world.set_pose(id, elsewhere, orientation).is_ok());
        assert!(world.sleep_body(id).is_ok());
        let _ = world.drain_dirty().count();

        assert!(world.restore(&state).is_ok());
        assert_eq!(world.bodies[id].position, settled);
        assert!(world.bodies[id].is_sleeping());
        assert!(
            world.drain_dirty().any(|(dirty, _)| dirty == id),
            "a restore moved a sleeping body without publishing it"
        );
    }

    #[test]
    fn a_snapshot_restores_into_a_rebuilt_world_or_rewinds_its_own_with_the_contact_cache() {
        let (mut world, floor, spheres) = settled_islands(EDIT_DT, EDIT_SETTLE_STEPS);
        let key = (floor, spheres[0]);
        let settled = normal_impulses(&world, key);
        assert!(
            settled.iter().any(|&jn| jn > 0.0),
            "the fixture carries no warm start, so a restore has no contact cache to reproduce"
        );
        let state = world.snapshot();

        let mut unregistered = World::new(EuclideanR3);
        assert_eq!(
            unregistered.restore(&state),
            Err(EditError::RegistrationMismatch),
            "restore accepted a world whose dispatch table cannot serve the snapshot"
        );
        assert_eq!(unregistered.bodies.len(), 0);

        let mut rebuilt = World::new(EuclideanR3);
        register_default_narrowphase(&mut rebuilt.narrowphase);
        rebuilt.set_gravity(world.gravity()).unwrap();
        assert!(rebuilt.restore(&state).is_ok());

        assert_eq!(
            rebuilt.state_hash(sample_body_r3),
            world.state_hash(sample_body_r3)
        );

        for _ in 0..60 {
            world.step(EDIT_DT, &Sequential).unwrap();
            rebuilt.step(EDIT_DT, &Sequential).unwrap();
        }
        assert_eq!(
            rebuilt.state_hash(sample_body_r3),
            world.state_hash(sample_body_r3),
            "the restored world solved with a different dispatch table"
        );

        assert!(world
            .apply_impulse(spheres[0], Vec3::new(0.0, 20.0, 0.0))
            .is_ok());
        for _ in 0..30 {
            world.step(EDIT_DT, &Sequential).unwrap();
        }
        assert!(
            !world.manifolds.contains_key(&key),
            "the launched sphere kept its contact, so the restore has nothing to undo"
        );
        assert!(world.restore(&state).is_ok());
        assert_eq!(normal_impulses(&world, key), settled);
        assert_eq!(world.time, state.time);
    }

    fn stacked_pair_world(doomed_first: bool) -> (World<EuclideanR3>, BodyId, BodyId, BodyId) {
        const LOWER_HALF_EXTENT: f32 = 0.5;
        const UPPER_HALF_EXTENT: f32 = 0.35;
        const DOOMED_X: f32 = -6.0;

        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world
            .set_gravity(Some(Vec3::new(0.0, GRAVITY_Y, 0.0)))
            .unwrap();

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

    #[test]
    fn gravity_does_not_wake_a_sleeping_support_until_its_neighbor_is_driven() {
        let (mut world, lower, upper, distant) = stacked_pair_world(false);
        world.despawn_body(distant).unwrap();
        let dt = 1.0 / 240.0;
        for _ in 0..600 {
            world.step(dt, &Sequential).unwrap();
        }
        world.sleep_body(lower).unwrap();
        for _ in 0..120 {
            world.step(dt, &Sequential).unwrap();
            assert!(world.bodies[lower].is_sleeping());
        }
        assert!(world.manifolds.contains_key(&canonical_pair(lower, upper)));
        world
            .set_velocity(upper, -Vec3::Y, Bivector3::ZERO)
            .unwrap();
        world.step(dt, &Sequential).unwrap();
        assert!(!world.bodies[lower].is_sleeping());
    }

    fn despawned_pair_world(
        doomed_first: bool,
        dt: f32,
    ) -> (World<EuclideanR3>, BodyId, BodyId, PairKey) {
        const SETTLE_STEPS: usize = 40;

        let (mut world, lower, upper, doomed) = stacked_pair_world(doomed_first);
        for _ in 0..SETTLE_STEPS {
            world.step(dt, &Sequential).unwrap();
        }

        let key = canonical_pair(lower, upper);
        assert!(
            world.manifolds.contains_key(&key),
            "the two dynamic bodies never settled into contact"
        );
        assert!(world.despawn_body(doomed).is_ok());

        let (i, j) = dense_pair(&world.bodies, key);
        assert_eq!(
            i > j,
            doomed_first,
            "the pair is stored at {i}, {j}, which is not the order this fixture \
             was built to produce"
        );
        (world, lower, upper, key)
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
                world.step(dt, &Sequential).unwrap();
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
        world
            .set_gravity(Some(Vec3::new(0.0, GRAVITY_Y, 0.0)))
            .unwrap();
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
            assert!(world.despawn_body(*doomed).is_ok());
        }
        world
    }

    #[test]
    fn broadphase_culls_only_pairs_the_narrowphase_would_reject() {
        for seed in PERMUTATION_SEEDS {
            let mut world = random_scene(seed, 40, 3.0);
            let mut emitted = Vec::new();
            for step in 0..8 {
                world.broadphase_into(&mut emitted);
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
                            world.geometry(),
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
                world.step(1.0 / 240.0, &Sequential).unwrap();
            }
        }
    }

    #[test]
    fn a_warmed_world_steps_and_publishes_without_allocating() {
        const DT: f32 = 1.0 / 240.0;
        let mut world = settled_columns(PERMUTATION_SEEDS[0]);
        for _ in 0..2 {
            world.step(DT, &Sequential).unwrap();
            assert!(world.drain_dirty().count() > 0);
        }
        assert!(world.constraints.len() > 1, "the fixture solves nothing");
        let bytes = bytes_allocated_by(|| {
            for _ in 0..16 {
                world.step(DT, &Sequential).unwrap();
                assert!(world.drain_dirty().count() > 0);
            }
        })
        .expect("the counting allocator is installed");
        assert_eq!(
            bytes, 0,
            "16 steps and dirty drains over a settled contact set asked the allocator for \
             {bytes} bytes"
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
        world
            .set_gravity(Some(Vec3::new(0.0, GRAVITY_Y, 0.0)))
            .unwrap();
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
            assert!(world.despawn_body(column[0]).is_ok());
        }
        for _ in 0..ISLAND_SETTLE_STEPS {
            world.step(1.0 / 240.0, &Sequential).unwrap();
        }
        world
    }
}
