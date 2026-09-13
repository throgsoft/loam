use std::collections::{BTreeMap, HashMap};
use std::ops::{Add, Mul};

use loam_time::par;
use loam_time::StateHash;

use crate::body::{BodyArena, BodyDef, BodyId, RigidBody};
use crate::collider::Collider;
use crate::collision::VectorOps;
use crate::dirty::{DirtyBodies, DirtyDrain};
use crate::edit::EditError;
use crate::field_contact::FieldNarrowphase;
use crate::geometry::GeometryStore;
use crate::integrator::{integrate_body, BroadphaseBound, PhysicsSpace};
use crate::manifold::{
    ContactPoint, Manifold, BAUMGARTE_BETA, DEFAULT_PGS_ITERS, MAX_LINEAR_CORRECTION,
    PENETRATION_SLOP, RESTITUTION_THRESHOLD,
};
use crate::narrowphase::Narrowphase;
use crate::response::Contact;
use crate::response::FRICTION_COEFF;
use crate::state::WorldState;

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

pub const DEFAULT_SOLVER_TOLERANCE: f32 = 1e-3;

const WAKE_APPROACH_SPEED: f32 = 0.01;

/// Measured, not derived; the sweep is in docs/PERF.md.
pub const ISLANDS_PER_SOLVE_WORKER: usize = 256;

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

fn bounding_radius(collider: Option<&Collider>) -> f32 {
    let Some(collider) = collider else {
        return 0.0;
    };
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

/// `residual` is the largest normal-impulse change of the last sweep; `converged` compares it to `World::solver_tolerance`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveReport {
    pub residual: f32,
    pub converged: bool,
}

impl Default for SolveReport {
    fn default() -> Self {
        Self {
            residual: 0.0,
            converged: true,
        }
    }
}

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
    residual: f32,
}

impl<S: PhysicsSpace> Default for IslandSolve<S> {
    fn default() -> Self {
        Self {
            bodies: Vec::new(),
            dense: Vec::new(),
            units: Vec::new(),
            points: Vec::new(),
            residual: 0.0,
        }
    }
}

const SCATTERED_NOWHERE: u32 = u32::MAX;

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
    /// Positions of `key.0` and `key.1`, in the key's order.
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
    /// PGS convergence depends on constraint order.
    pub(crate) manifolds: HashMap<PairKey, Manifold<S>>,
    manifold_order: Vec<PairKey>,
    manifold_pool: Vec<Manifold<S>>,
    pending_contacts: Vec<(PairKey, Contact<S>)>,
    pgs_iters: usize,
    time: f32,
    dirty: DirtyBodies,
    pair_order: Vec<PairKey>,
    constraints: Vec<ConstraintUnit>,
    broadphase_intervals: Vec<RadialInterval>,
    broadphase_active: Vec<u32>,
    touched_pairs: Vec<PairKey>,
    island_parent: Vec<u32>,
    island_labels: Vec<BodyId>,
    counters: StepCounters,
    /// Same unit as [`SolveReport::residual`]: impulse, mass times speed.
    solver_tolerance: f32,
    report: SolveReport,
    scratch: Vec<IslandSolve<S>>,
    scratch_islands: usize,
    scratch_local: Vec<u32>,
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
            pgs_iters: DEFAULT_PGS_ITERS,
            time: 0.0,
            dirty: DirtyBodies::default(),
            pair_order: Vec::new(),
            constraints: Vec::new(),
            broadphase_intervals: Vec::new(),
            broadphase_active: Vec::new(),
            touched_pairs: Vec::new(),
            island_parent: Vec::new(),
            island_labels: Vec::new(),
            counters: StepCounters::default(),
            solver_tolerance: DEFAULT_SOLVER_TOLERANCE,
            report: SolveReport::default(),
            scratch: Vec::new(),
            scratch_islands: 0,
            scratch_local: Vec::new(),
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

    pub fn solver_tolerance(&self) -> f32 {
        self.solver_tolerance
    }

    pub fn set_solver_tolerance(&mut self, tolerance: f32) -> Result<(), EditError> {
        if tolerance.is_nan() || tolerance < 0.0 {
            return Err(EditError::InvalidSolverTolerance);
        }
        self.solver_tolerance = tolerance;
        Ok(())
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

    pub fn manifold(&self, key: PairKey) -> Option<&Manifold<S>> {
        self.manifolds.get(&key)
    }

    pub fn solve_report(&self) -> SolveReport {
        self.report
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
        let Self {
            space,
            bodies,
            geometry,
            ..
        } = self;
        bodies[id].replace_collider(geometry, space, collider, inertia)?;
        self.drop_contacts_of(id);
        self.dirty.mark(id);
        Ok(())
    }

    pub fn apply_impulse(&mut self, id: BodyId, impulse: S::Vector) -> Result<(), EditError>
    where
        S::Vector: Add<Output = S::Vector> + Mul<f32, Output = S::Vector>,
    {
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
    ) -> Result<(), EditError>
    where
        S::Vector: Add<Output = S::Vector> + Mul<f32, Output = S::Vector>,
    {
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

    pub fn sleep_body(&mut self, id: BodyId) -> Result<(), EditError>
    where
        S::Vector: Default,
    {
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
        let Self { bodies, dirty, .. } = self;
        dirty.drain(bodies)
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
        S::Vector: VectorOps,
        S::AngVel: PartialEq,
        S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    {
        if self
            .gravity
            .is_some_and(|gravity| !self.space.valid_vector(gravity))
        {
            return Err(EditError::InvalidGravity);
        }
        if self.solver_tolerance.is_nan() || self.solver_tolerance < 0.0 {
            return Err(EditError::InvalidSolverTolerance);
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
        S::Vector: VectorOps,
        S::AngVel: PartialEq,
        S::Point: Copy + std::ops::Sub<Output = S::Vector>,
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
        self.broadphase_intervals.clear();
        self.broadphase_active.clear();
        self.island_parent.clear();
        self.island_labels.clear();
        self.scratch_islands = 0;
        self.report = SolveReport::default();
        Ok(())
    }

    /// `dt` is in seconds.
    pub fn step(&mut self, dt: f32) -> Result<(), EditError>
    where
        S::Vector: VectorOps,
        S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    {
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
        self.update_manifolds();
        self.collect_constraints();
        self.prepare_solve(dt);
        self.warm_start();
        self.solve();

        self.time = next_time;
        Ok(())
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
            if self.bodies[i].inv_mass() == 0.0 {
                continue;
            }
            integrate_body(&self.space, &mut self.bodies[i], dt);
            let id = self.bodies.id_at(i);
            self.dirty.mark(id);
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
            &self.geometry,
            &self.space,
            &mut self.broadphase_intervals,
            &mut self.broadphase_active,
            &mut pairs,
            &mut self.counters,
        );
        let mut touched = std::mem::take(&mut self.touched_pairs);
        touched.clear();
        let mut pending = std::mem::take(&mut self.pending_contacts);
        pending.clear();

        for &key in &pairs {
            let (i, j) = self.dense_pair(key);
            let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
            // Refresh anchors before the narrowphase merges contacts.
            if let Some(manifold) = self.manifolds.get_mut(&key) {
                manifold.refresh(&self.space, a, b);
            }
            let Some(contact) = self.narrowphase.test(a, b, &self.geometry, &self.space) else {
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
            pending.push((key, contact));
        }

        self.field_manifolds(&mut pending);

        touched.extend(pending.iter().map(|(key, _)| *key));
        touched.sort_unstable();
        touched.dedup();
        self.counters.contacts = pending.len() as u32;
        self.recycle_untouched_manifolds(&touched);
        for &(key, contact) in &pending {
            self.ensure_manifold(key, contact.restitution);
            let Self {
                bodies,
                manifolds,
                space,
                ..
            } = self;
            let i = bodies
                .dense_index(key.0)
                .unwrap_or_else(|| panic!("{STALE_MANIFOLD_BODY}"));
            let j = bodies
                .dense_index(key.1)
                .unwrap_or_else(|| panic!("{STALE_MANIFOLD_BODY}"));
            let (a, b) = split_two_mut(bodies.dense_mut(), i, j);
            manifolds
                .get_mut(&key)
                .unwrap_or_else(|| panic!("{STALE_CONSTRAINT_KEY}"))
                .add_or_update(space, a, b, contact);
        }
        self.touched_pairs = touched;
        self.pending_contacts = pending;
        self.pair_order = pairs;
    }

    fn ensure_manifold(&mut self, key: PairKey, restitution: f32)
    where
        S::Vector: VectorOps,
    {
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

    fn recycle_untouched_manifolds(&mut self, touched: &[PairKey]) {
        for &key in &self.manifold_order {
            if touched.binary_search(&key).is_err() {
                if let Some(manifold) = self.manifolds.remove(&key) {
                    self.manifold_pool.push(manifold);
                }
            }
        }
        self.manifold_order.clear();
        self.manifold_order.extend_from_slice(touched);
    }

    fn wake_approaching_contacts(&mut self)
    where
        S::Vector: VectorOps,
    {
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

    fn field_manifolds(&mut self, pending: &mut Vec<(PairKey, Contact<S>)>)
    where
        S::Vector: VectorOps,
        S::Point: Copy + std::ops::Sub<Output = S::Vector>,
    {
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
            let (i, j) = self.dense_pair(key);
            let (a, b) = split_two_mut(self.bodies.dense_mut(), i, j);
            if let Some(manifold) = self.manifolds.get_mut(&key) {
                manifold.refresh(&self.space, a, b);
            }
            pending.push((key, contact));
        }
    }

    // Manifold membership must stay fixed until the solve ends.
    fn collect_constraints(&mut self) {
        let mut parent = std::mem::take(&mut self.island_parent);
        let mut labels = std::mem::take(&mut self.island_labels);
        Self::fill_islands(
            &self.bodies,
            self.manifold_order.iter().copied(),
            &mut parent,
            &mut labels,
        );
        let mut units = std::mem::take(&mut self.constraints);
        units.clear();
        units.extend(self.manifold_order.iter().map(|&key| {
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
        self.gather_islands();
        let Self {
            space,
            scratch,
            scratch_islands,
            pgs_iters,
            ..
        } = self;
        let iterations = *pgs_iters;
        let space = &*space;
        let islands = &mut scratch[..*scratch_islands];
        let workers = islands.len() / ISLANDS_PER_SOLVE_WORKER;
        if workers < 2 {
            for island in islands.iter_mut() {
                solve_island(space, iterations, island);
            }
        } else {
            let chunk = islands.len().div_ceil(workers);
            par::for_each_chunk(islands, chunk, |chunk| {
                for island in chunk {
                    solve_island(space, iterations, island);
                }
            });
        }
        self.scatter_islands();
    }

    fn gather_islands(&mut self) {
        self.scratch_local.clear();
        self.scratch_local
            .resize(self.bodies.len(), SCATTERED_NOWHERE);
        self.scratch_islands = 0;
        let mut current = None;
        for unit in &self.constraints {
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
            let a = gather_body(island, &mut self.scratch_local, &self.bodies, i);
            let b = gather_body(island, &mut self.scratch_local, &self.bodies, j);
            let manifold = self
                .manifolds
                .get(&unit.key)
                .unwrap_or_else(|| panic!("{STALE_CONSTRAINT_KEY}"));
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

    fn scatter_islands(&mut self) {
        let mut residual = 0.0_f32;
        for island in &self.scratch[..self.scratch_islands] {
            residual = residual.max(island.residual);
            for (local, &dense) in island.dense.iter().enumerate() {
                if dense != SCATTERED_NOWHERE {
                    self.bodies[dense as usize] = island.bodies[local];
                }
            }
            for unit in &island.units {
                let manifold = self
                    .manifolds
                    .get_mut(&unit.key)
                    .unwrap_or_else(|| panic!("{STALE_CONSTRAINT_KEY}"));
                let first = unit.first as usize;
                manifold
                    .points
                    .copy_from_slice(&island.points[first..first + unit.count as usize]);
            }
        }
        self.report = SolveReport {
            residual,
            converged: residual <= self.solver_tolerance,
        };
    }

    /// Replaces `pairs` with the sorted candidates: bounding-ball overlaps under a certified bound, every masked non-static pair otherwise.
    pub fn broadphase_into(&mut self, pairs: &mut Vec<PairKey>) {
        self.counters = StepCounters::default();
        Self::fill_broadphase(
            &self.bodies,
            &self.geometry,
            &self.space,
            &mut self.broadphase_intervals,
            &mut self.broadphase_active,
            pairs,
            &mut self.counters,
        );
    }

    // Cohen, Lin, Manocha, Ponamgi, I-COLLIDE, 1995, sec. 3; sweep radial distances.
    fn fill_broadphase(
        bodies: &BodyArena<S>,
        geometry: &GeometryStore,
        space: &S,
        intervals: &mut Vec<RadialInterval>,
        active: &mut Vec<u32>,
        pairs: &mut Vec<PairKey>,
        counters: &mut StepCounters,
    ) {
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
            let radius = bounding_radius(geometry.get(body.collider()));
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
        let mut parent = Vec::new();
        let mut labels = Vec::new();
        Self::fill_islands(
            &self.bodies,
            self.manifold_order.iter().copied(),
            &mut parent,
            &mut labels,
        );

        let mut by_id: BTreeMap<BodyId, Island> = BTreeMap::new();
        for &key in &self.manifold_order {
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
fn solve_island<S>(space: &S, iterations: usize, island: &mut IslandSolve<S>)
where
    S: PhysicsSpace,
    S::Vector: VectorOps,
{
    let IslandSolve {
        bodies,
        units,
        points,
        residual,
        ..
    } = island;
    *residual = 0.0;
    for _ in 0..iterations {
        *residual = 0.0;
        for unit in units.iter() {
            let (a, b) = split_two_mut(bodies, unit.a as usize, unit.b as usize);
            let first = unit.first as usize;
            for cp in &mut points[first..first + unit.count as usize] {
                *residual = residual.max(solve_normal_then_tangent(space, a, b, cp));
            }
        }
    }
}

fn solve_normal_then_tangent<S>(
    space: &S,
    a: &mut RigidBody<S>,
    b: &mut RigidBody<S>,
    cp: &mut ContactPoint<S>,
) -> f32
where
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

    let mut change = 0.0;
    if k_n > 0.0 {
        let dj = -(v_n + cp.velocity_bias) / k_n;
        let new_acc = (cp.normal_impulse + dj).max(0.0);
        let actual = new_acc - cp.normal_impulse;
        cp.normal_impulse = new_acc;
        change = actual.abs();
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
    change
}

#[cfg(all(test, feature = "r2", feature = "r3", feature = "r4"))]
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
    use glam::{Quat, Vec3};
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
        assert!(compacted.despawn_body(doomed).is_ok());

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
            world.step(MULTI_ISLAND_DT).unwrap();
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
        world
            .set_gravity(Some(Vec3::new(0.0, GRAVITY_Y, 0.0)))
            .unwrap();

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
            world.step(dt).unwrap();
        }
        world
    }

    #[test]
    fn a_one_iteration_budget_separates_a_resting_contact_from_a_fast_impact() {
        const DT: f32 = 1.0 / 240.0;
        let mut resting = settled_sphere_stack(DT, 240);
        resting.set_solver_iterations(1);
        resting.step(DT).unwrap();
        let settled = resting.solve_report();
        assert!(
            resting.counters().contacts > 0,
            "the resting stack lost its contacts"
        );
        assert!(settled.converged, "resting residual {}", settled.residual);

        let mut struck = settled_sphere_stack(DT, 240);
        struck.set_solver_iterations(1);
        let top = struck.bodies.id_at(2);
        struck
            .set_velocity(top, Vec3::new(0.0, -20.0, 0.0), Bivector3::ZERO)
            .unwrap();
        struck.step(DT).unwrap();
        let impact = struck.solve_report();
        assert!(!impact.converged, "impact residual {}", impact.residual);
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

        fn floor_r2() -> BodyDef<EuclideanR2> {
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
                world.step(REBOUND_DT).unwrap();
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
                world.step(SLIDE_DT).unwrap();
                for (_, manifold) in world.manifolds() {
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
            world.set_gravity(Some(Vec2::new(0.0, GRAVITY_Y))).unwrap();
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

            assert_tangent_impulse_stays_inside_the_coulomb_cone(world, sphere, Vec3::X, Vec3::Y);
        }

        #[test]
        fn tangent_impulse_stays_inside_the_coulomb_cone_r4() {
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
                .manifolds()
                .map(|(_, manifold)| manifold)
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
                warm.manifolds()
                    .map(|(_, manifold)| manifold)
                    .flat_map(|m| &m.points)
                    .any(|cp| cp.normal_impulse > 0.0),
                "fixture carries no warm-start payload"
            );

            clear_warm_start(&mut cold_converged);
            clear_warm_start(&mut cold_default);
            cold_converged.set_solver_iterations(400);

            warm.step(STACK_DT).unwrap();
            cold_converged.step(STACK_DT).unwrap();
            cold_default.step(STACK_DT).unwrap();

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
            world.set_gravity(Some(Vec2::new(0.0, GRAVITY_Y))).unwrap();

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
                world.step(STACK_DT).unwrap();
            }
            world
        }

        fn settled_sphere_stack_r4() -> World<EuclideanR4> {
            let mut world = World::new(EuclideanR4);
            crate::euclidean_r4::register_default_narrowphase(&mut world.narrowphase);
            world
                .set_gravity(Some(Vec4::new(0.0, GRAVITY_Y, 0.0, 0.0)))
                .unwrap();

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
                world.step(STACK_DT).unwrap();
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
            let mut geometry = crate::geometry::GeometryStore::default();
            let mut a = sphere_body(Vec2::ZERO, Vec2::ZERO, 1.0, 0.0)
                .unwrap()
                .into_row(&mut geometry, &EuclideanR2);
            let mut b = sphere_body(Vec2::ZERO, Vec2::new(slip, 2.0), 1.0, 1.0)
                .unwrap()
                .into_row(&mut geometry, &EuclideanR2);
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
            world.step(dt).unwrap();
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

        assert!(world.despawn_body(doomed).is_ok());

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
            world.step(dt).unwrap();
            control.step(dt).unwrap();
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
            world.step(dt).unwrap();
            control.step(dt).unwrap();
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

        assert!(world.despawn_body(doomed).is_ok());
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

        world.step(dt).unwrap();
        assert!(
            world.manifolds.contains_key(&(floor, reborn)),
            "the respawned body made no contact"
        );
        assert!(
            !world.manifolds.contains_key(&(floor, doomed)),
            "a manifold keyed on the despawned body came back with the slot"
        );
    }

    const EDIT_DT: f32 = 1.0 / 240.0;
    const EDIT_SETTLE_STEPS: usize = 400;

    #[test]
    fn a_rejected_edit_leaves_the_body_its_inertia_and_its_contacts_untouched() {
        let (mut world, floor, spheres) = settled_islands(EDIT_DT, EDIT_SETTLE_STEPS);
        let id = spheres[0];
        let key = (floor, id);
        let before = body_state(&world, id);
        let mass = world.bodies[id].mass();
        let inv_mass = world.bodies[id].inv_mass();
        let inertia = world.bodies[id].inertia;
        let orientation = world.bodies[id].orientation;
        let stale = BodyId::forge(u32::MAX, 0);
        let impulses = normal_impulses(&world, key);
        let _ = world.drain_dirty().count();
        assert!(
            impulses.iter().any(|&jn| jn > 0.0),
            "the fixture carries no warm start, so a preserved contact proves nothing"
        );

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
        while world.reclaim_geometry().is_some() {}
        assert_eq!(
            world.set_collider(id, Collider::sphere_at_origin(0.5), f32::INFINITY),
            Err(EditError::InvalidInertia)
        );
        assert!(
            world.reclaim_geometry().is_some(),
            "a rejected inertia swallowed the caller's collider"
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
        assert_eq!(
            world.drain_dirty().count(),
            0,
            "a rejected edit published a pose change"
        );

        assert_eq!(body_state(&world, id), before);
        assert_eq!(world.bodies[id].mass(), mass);
        assert_eq!(world.bodies[id].inv_mass(), inv_mass);
        assert_eq!(world.bodies[id].inertia, inertia);
        assert_eq!(normal_impulses(&world, key), impulses);

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
    }

    #[test]
    fn warm_start_impulses_survive_a_step_and_a_snapshot_restore() {
        let (mut world, floor, spheres) = settled_islands(EDIT_DT, EDIT_SETTLE_STEPS);
        let key = (floor, spheres[0]);
        let settled = normal_impulses(&world, key);
        assert!(settled.iter().any(|&jn| jn > 0.0));

        let state = world.snapshot();
        world.step(EDIT_DT).unwrap();

        let stepped = normal_impulses(&world, key);
        assert_eq!(
            stepped.len(),
            settled.len(),
            "a step dropped a contact slot"
        );
        for (after, before) in stepped.iter().zip(&settled) {
            assert!(
                *after > 0.0 && (after - before).abs() < 1e-3,
                "a step reset the accumulator: {before} became {after}"
            );
        }

        assert!(world
            .apply_impulse(spheres[0], Vec3::new(0.0, 20.0, 0.0))
            .is_ok());
        for _ in 0..30 {
            world.step(EDIT_DT).unwrap();
        }
        assert!(
            !world.manifolds.contains_key(&key),
            "the launched sphere kept its contact, so the restore has nothing to undo"
        );

        assert!(world.restore(&state).is_ok());
        assert_eq!(normal_impulses(&world, key), settled);
        assert_eq!(world.time, state.time);
    }

    #[test]
    fn invalid_snapshot_state_is_refused_before_the_world_changes() {
        let (mut world, floor, spheres) = settled_islands(EDIT_DT, EDIT_SETTLE_STEPS);
        let key = (floor, spheres[0]);
        let before = world.state_hash(sample_body_r3);
        let time = world.time;
        let impulses = normal_impulses(&world, key);
        let unchanged = |world: &World<EuclideanR3>| {
            assert_eq!(world.state_hash(sample_body_r3), before);
            assert_eq!(world.time, time);
            assert_eq!(normal_impulses(world, key), impulses);
        };

        let mut invalid_body = world.snapshot();
        invalid_body.bodies[spheres[0]].orientation.rotation = Quat::from_xyzw(0.0, 0.0, 0.0, 2.0);
        assert_eq!(world.restore(&invalid_body), Err(EditError::InvalidBody));
        unchanged(&world);

        let mut invalid_time = world.snapshot();
        invalid_time.time = f32::NAN;
        assert_eq!(world.restore(&invalid_time), Err(EditError::InvalidTime));
        unchanged(&world);

        let mut invalid_manifold = world.snapshot();
        invalid_manifold.manifolds.get_mut(&key).unwrap().points[0].normal = Vec3::splat(f32::NAN);
        assert_eq!(
            world.restore(&invalid_manifold),
            Err(EditError::InvalidManifold)
        );
        unchanged(&world);
    }

    #[test]
    fn invalid_gravity_and_time_steps_leave_body_state_and_time_unchanged() {
        let mut world = World::new(EuclideanR3);
        let body = world.push_body(sphere_body_r3(Vec3::X, Vec3::Y, 0.5, 1.0).unwrap());
        let before = world.state_hash(sample_body_r3);

        assert_eq!(
            world.set_gravity(Some(Vec3::splat(f32::NAN))),
            Err(EditError::InvalidGravity)
        );
        assert_eq!(world.gravity(), None);
        for dt in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
            assert_eq!(world.step(dt), Err(EditError::InvalidTimeStep));
            assert!(world.body(body).is_some());
            assert_eq!(world.state_hash(sample_body_r3), before);
            assert_eq!(world.time(), 0.0);
        }

        world.step(0.0).unwrap();
        assert!(world.body(body).is_some());
        assert_eq!(world.state_hash(sample_body_r3), before);
        assert_eq!(world.time(), 0.0);

        world.time = f32::MAX;
        assert_eq!(world.step(f32::MAX), Err(EditError::InvalidTimeStep));
        assert_eq!(world.state_hash(sample_body_r3), before);
        assert_eq!(world.time(), f32::MAX);
    }

    #[test]
    fn a_teleported_sleeping_body_publishes_its_pose_change() {
        let (mut world, _, spheres) = settled_islands(EDIT_DT, EDIT_SETTLE_STEPS);
        let id = spheres[0];
        assert!(world.sleep_body(id).is_ok());
        assert!(world.drain_dirty().count() > 0);

        world.step(EDIT_DT).unwrap();
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
    }

    #[test]
    fn a_restored_world_rebuilds_its_configuration_and_reproduces_the_snapshot() {
        let (mut world, floor, spheres) = settled_islands(EDIT_DT, EDIT_SETTLE_STEPS);
        let key = (floor, spheres[0]);
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
        assert_eq!(normal_impulses(&rebuilt, key), normal_impulses(&world, key));

        for _ in 0..60 {
            world.step(EDIT_DT).unwrap();
            rebuilt.step(EDIT_DT).unwrap();
        }
        assert_eq!(
            rebuilt.state_hash(sample_body_r3),
            world.state_hash(sample_body_r3),
            "the restored world solved with a different dispatch table"
        );
    }

    #[test]
    fn a_collider_replacement_drops_the_previous_shapes_inertia() {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        let id = world.push_body(box_body(Vec3::ZERO, Vec3::ZERO, Vec3::ONE, 2.0).unwrap());
        let previous = world.bodies[id].inertia;
        let replacement = 0.5;
        assert_ne!(
            previous, replacement,
            "the two shapes share an inertia, so the swap cannot be seen"
        );

        assert!(world
            .set_collider(id, Collider::sphere_at_origin(0.25), replacement)
            .is_ok());
        assert!(matches!(
            world.collider(&world.bodies[id]),
            Some(Collider::Sphere { .. })
        ));
        assert!(world
            .apply_impulse_at_point(id, Vec3::new(3.0, 0.0, 0.0), Vec3::new(0.0, 2.0, 0.0))
            .is_ok());

        assert_eq!(world.bodies[id].velocity, Vec3::new(1.5, 0.0, 0.0));
        assert_eq!(
            world.bodies[id].angular_velocity,
            Bivector3::new(-12.0, 0.0, 0.0)
        );
    }

    #[test]
    fn a_despawn_leaves_no_contact_or_dirty_row_naming_the_removed_body() {
        let (mut world, floor, spheres) = settled_islands(EDIT_DT, EDIT_SETTLE_STEPS);
        let doomed = spheres[1];
        assert!(world.manifolds.contains_key(&(floor, doomed)));
        assert!(world
            .set_velocity(doomed, Vec3::new(0.0, 3.0, 0.0), Bivector3::ZERO)
            .is_ok());

        assert!(world.despawn_body(doomed).is_ok());

        assert!(
            !world
                .manifolds
                .keys()
                .any(|&(a, b)| a == doomed || b == doomed),
            "a contact outlived the body it names"
        );
        let reborn = world.push_body(island_sphere(ISLAND_X[1]));
        assert_eq!(
            reborn.slot(),
            doomed.slot(),
            "the slot was not recycled, so this test is not exercising aliasing"
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

        world.step(EDIT_DT).unwrap();
        assert!(!world
            .manifolds
            .keys()
            .any(|&(a, b)| a == doomed || b == doomed));
    }

    #[test]
    fn bodies_that_share_a_hull_hold_one_copy_of_its_vertices() {
        use crate::euclidean_r4::{polytope_body_r4, tesseract_vertices};
        use glam::Vec4;
        use loam_math::EuclideanR4;

        let hull = tesseract_vertices(1.0);
        let mut world = World::new(EuclideanR4);
        let a =
            world.push_body(polytope_body_r4(Vec4::ZERO, Vec4::ZERO, hull.clone(), 1.0).unwrap());
        let b = world.push_body(polytope_body_r4(Vec4::X * 8.0, Vec4::ZERO, hull, 1.0).unwrap());

        assert_eq!(world.bodies[a].collider(), world.bodies[b].collider());
        assert_eq!(world.geometry().prepared(), 1);

        let vertices = |id: BodyId| match world.collider(&world.bodies[id]) {
            Some(Collider::ConvexPolytope4D { vertices }) => vertices.as_ptr(),
            other => panic!("the hull is not prepared: {other:?}"),
        };
        assert_eq!(
            vertices(a),
            vertices(b),
            "each body kept its own copy of the hull"
        );

        assert!(world.despawn_body(a).is_ok());
        assert_eq!(
            world.geometry().prepared(),
            1,
            "a shared hull was released while another body still used it"
        );
        assert!(world.despawn_body(b).is_ok());
        assert_eq!(
            world.geometry().prepared(),
            0,
            "the last body's hull outlived it"
        );
    }

    #[test]
    fn a_restored_sleeping_body_publishes_the_pose_the_restore_gave_it() {
        let (mut world, _, spheres) = settled_islands(EDIT_DT, EDIT_SETTLE_STEPS);
        let id = spheres[0];
        assert!(world.sleep_body(id).is_ok());
        let settled = world.bodies[id].position;

        let state = world.snapshot();
        let orientation = world.bodies[id].orientation;
        assert!(world
            .set_pose(id, Vec3::new(40.0, 9.0, 0.0), orientation)
            .is_ok());
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
    fn a_drained_row_is_readable_without_copying_the_ids_out_first() {
        let (mut world, _, spheres) = settled_islands(EDIT_DT, 4);
        let _ = world.drain_dirty().count();
        world.step(EDIT_DT).unwrap();

        let mut mirror: BTreeMap<BodyId, Vec3> = BTreeMap::new();
        for (id, body) in world.drain_dirty() {
            mirror.insert(id, body.position);
        }

        assert_eq!(mirror.len(), spheres.len());
        for id in &spheres {
            assert_eq!(mirror.get(id), Some(&world.bodies[*id].position));
        }
    }

    #[test]
    fn a_half_space_on_a_body_with_mass_is_not_reported_as_an_unsupported_collider() {
        let mut world = World::new(EuclideanR3);
        let id = world.push_body(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 0.5, 2.0).unwrap());
        let floor = Collider::HalfSpace {
            normal: Vec3::Y,
            offset: 0.0,
        };
        assert!(
            world.space.supports_collider(floor.kind()),
            "the space rejects half-spaces outright, so this cannot tell the two apart"
        );

        assert_eq!(
            world.set_collider(id, floor, 1.0),
            Err(EditError::DynamicHalfSpace)
        );
        assert_eq!(world.bodies[id].mass(), 2.0);

        let ground = world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());
        assert_eq!(
            world.set_mass_properties(ground, 3.0, 1.0),
            Err(EditError::DynamicHalfSpace)
        );
        assert_eq!(world.bodies[ground].mass(), 0.0);
    }

    #[test]
    fn registration_order_does_not_refuse_a_snapshot_of_the_same_pairs() {
        fn never(
            _a: &RigidBody<EuclideanR3>,
            _b: &RigidBody<EuclideanR3>,
            _geometry: &crate::geometry::GeometryStore,
            _space: &EuclideanR3,
        ) -> Option<crate::response::Contact<EuclideanR3>> {
            None
        }

        use crate::collider::ColliderKind;

        let pairs = [
            (ColliderKind::Sphere, ColliderKind::HalfSpace),
            (ColliderKind::Sphere, ColliderKind::ConvexPolytope3D),
            (ColliderKind::ConvexPolytope3D, ColliderKind::HalfSpace),
        ];
        assert_ne!(
            pairs.first(),
            pairs.last(),
            "the registration sequence is symmetric, so reversing it changes nothing"
        );

        let mut forward = World::new(EuclideanR3);
        for &(a, b) in &pairs {
            forward.narrowphase.register(a, b, never);
        }
        let mut reversed = World::new(EuclideanR3);
        for &(a, b) in pairs.iter().rev() {
            reversed.narrowphase.register(a, b, never);
        }
        forward.push_body(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 0.5, 1.0).unwrap());

        assert!(reversed.restore(&forward.snapshot()).is_ok());
        assert!(forward.restore(&reversed.snapshot()).is_ok());
        assert_eq!(reversed.bodies.len(), 1);
    }

    #[test]
    fn dirty_publication_allocates_nothing_on_a_warmed_world() {
        let (mut world, _, spheres) = settled_islands(EDIT_DT, 4);
        for _ in 0..2 {
            world.step(EDIT_DT).unwrap();
            assert!(world.drain_dirty().count() > 0);
        }

        let bytes = alloc_probe::bytes_allocated_by(|| {
            for _ in 0..16 {
                for &id in &spheres {
                    assert!(world.wake_body(id).is_ok());
                }
                assert_eq!(world.drain_dirty().count(), spheres.len());
            }
        });
        assert_eq!(
            bytes, 0,
            "16 dirty publications over a steady body set asked the allocator for {bytes} bytes"
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
            world.step(dt).unwrap();
        }
        world.sleep_body(lower).unwrap();
        for _ in 0..120 {
            world.step(dt).unwrap();
            assert!(world.bodies[lower].is_sleeping());
        }
        assert!(world.manifolds.contains_key(&canonical_pair(lower, upper)));
        world
            .set_velocity(upper, -Vec3::Y, Bivector3::ZERO)
            .unwrap();
        world.step(dt).unwrap();
        assert!(!world.bodies[lower].is_sleeping());
    }

    fn despawned_pair_world(
        doomed_first: bool,
        dt: f32,
    ) -> (World<EuclideanR3>, BodyId, BodyId, PairKey) {
        const SETTLE_STEPS: usize = 40;

        let (mut world, lower, upper, doomed) = stacked_pair_world(doomed_first);
        for _ in 0..SETTLE_STEPS {
            world.step(dt).unwrap();
        }

        let key = canonical_pair(lower, upper);
        assert!(
            world.manifolds.contains_key(&key),
            "the two dynamic bodies never settled into contact"
        );
        assert!(world.despawn_body(doomed).is_ok());

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
            world.step(dt).unwrap();

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
                world.step(dt).unwrap();
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

    fn all_pairs_reference(world: &World<EuclideanR3>) -> Vec<PairKey> {
        let mut pairs = Vec::new();
        let n = world.bodies.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let (a, b) = (&world.bodies[i], &world.bodies[j]);
                if a.inv_mass() == 0.0 && b.inv_mass() == 0.0 {
                    continue;
                }
                let reach = bounding_radius(world.collider(a)) + bounding_radius(world.collider(b));
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
                let mut emitted = Vec::new();
                for step in 0..8 {
                    let expected = all_pairs_reference(&world);
                    world.broadphase_into(&mut emitted);
                    assert_eq!(
                        emitted, expected,
                        "seed {seed}, {count} bodies, spread {spread}, step {step}"
                    );
                    ever_beyond_the_floor |= expected.len() > dynamic_body_count(&world);
                    world.step(1.0 / 240.0).unwrap();
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
                world.step(1.0 / 240.0).unwrap();
            }
        }
    }

    #[test]
    fn broadphase_emits_strictly_ascending_keys_under_disagreeing_storage_order() {
        for seed in PERMUTATION_SEEDS {
            let mut world = random_scene(seed, 40, 3.0);
            let disagrees = (1..world.bodies.len())
                .any(|dense| world.bodies.id_at(dense) < world.bodies.id_at(dense - 1));
            assert!(
                disagrees,
                "seed {seed}: storage order still agrees with handle order, so this \
                 scene cannot tell the two apart"
            );

            let mut pairs = Vec::new();
            world.broadphase_into(&mut pairs);
            assert!(pairs.len() > 1, "seed {seed}: too few pairs to be ordered");
            assert!(
                pairs.windows(2).all(|w| w[0] < w[1]),
                "seed {seed}: emission order is not strictly ascending in BodyId"
            );
        }
    }

    #[test]
    fn broadphase_prunes_the_quadratic_pair_set_at_scale() {
        let mut world = random_scene(PERMUTATION_SEEDS[1], 200, 20.0);
        let n = world.bodies.len();
        assert!(
            n >= 100,
            "the scale case needs at least 100 bodies, got {n}"
        );
        let all_pairs = n * (n - 1) / 2;
        let mut pairs = Vec::new();
        world.broadphase_into(&mut pairs);
        let emitted = pairs.len();
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
                world.geometry(),
                &world.space,
                &mut intervals,
                &mut active,
                &mut pairs,
                &mut StepCounters::default(),
            );
        }
        assert!(!pairs.is_empty(), "the fixture produced no pairs to emit");

        let bytes = alloc_probe::bytes_allocated_by(|| {
            for _ in 0..16 {
                World::fill_broadphase(
                    &world.bodies,
                    world.geometry(),
                    &world.space,
                    &mut intervals,
                    &mut active,
                    &mut pairs,
                    &mut StepCounters::default(),
                );
            }
        });
        assert_eq!(
            bytes, 0,
            "16 sweeps over a steady body set asked the allocator for {bytes} bytes"
        );
    }

    #[test]
    fn step_counters_report_the_sweep_work_of_a_four_body_line() {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        for x in [0.0, 0.75, 1.75, 5.0] {
            world.push_body(sphere_body_r3(Vec3::new(x, 0.0, 0.0), Vec3::ZERO, 0.5, 1.0).unwrap());
        }
        world.step(1.0 / 240.0).unwrap();
        assert_eq!(
            world.counters(),
            StepCounters {
                index_visits: 7,
                distance_evals: 6,
                candidates: 2,
                contacts: 1,
            }
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

        let mut pairs = Vec::new();
        world.broadphase_into(&mut pairs);
        assert_eq!(pairs, vec![canonical_pair(anchor, tangent)]);
    }

    #[test]
    fn coincident_point_colliders_are_a_candidate_pair() {
        let mut world = World::new(EuclideanR3);
        let a = world.push_body(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 0.0, 1.0).unwrap());
        let b = world.push_body(sphere_body_r3(Vec3::ZERO, Vec3::ZERO, 0.0, 1.0).unwrap());
        let mut pairs = Vec::new();
        world.broadphase_into(&mut pairs);
        assert_eq!(pairs, vec![canonical_pair(a, b)]);
    }

    #[test]
    fn bounding_radius_contains_every_posed_vertex_of_its_collider() {
        let half_extents = Vec3::new(0.5, 1.25, 0.25);
        let vertices = crate::euclidean_r3::box_vertices(half_extents);
        let radius = bounding_radius(Some(&Collider::ConvexPolytope3D {
            vertices: vertices.clone(),
        }));
        assert_eq!(radius, half_extents.length());

        let rotation = glam::Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7);
        for v in &vertices {
            let posed = rotation * *v;
            assert!(
                posed.length() <= radius + 1e-6,
                "posed vertex {posed} escaped the bounding radius {radius}"
            );
        }

        assert_eq!(
            bounding_radius(Some(&Collider::sphere_at_origin(0.75))),
            0.75
        );
        assert_eq!(
            bounding_radius(Some(&Collider::HalfSpace {
                normal: Vec3::Y,
                offset: 0.0
            })),
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
            world.step(1.0 / 240.0).unwrap();
        }
        world
    }

    #[test]
    fn island_ids_are_the_lowest_body_id_not_the_lowest_storage_position() {
        let mut orders_disagreed = 0usize;
        for seed in PERMUTATION_SEEDS {
            let mut world = settled_columns(seed);
            for step in 0..8 {
                world.step(1.0 / 240.0).unwrap();
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
                    body.inv_mass() == 0.0
                        && matches!(world.collider(body), Some(Collider::Sphere { .. }))
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
        for (&(a, b), _) in world.manifolds() {
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
                .manifolds()
                .map(|(&key, _)| key)
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
                world.step(1.0 / 240.0).unwrap();
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

        let ascending: Vec<PairKey> = world.manifolds().map(|(&key, _)| key).collect();
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
            world.step(MULTI_ISLAND_DT).unwrap();
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

        let ascending: Vec<PairKey> = world.manifolds().map(|(&key, _)| key).collect();
        assert_ne!(
            grouped, ascending,
            "the fixture's islands happen to be contiguous in ascending key \
             order, so it cannot show that grouping reorders anything"
        );
    }

    #[test]
    fn island_scratch_allocates_nothing_after_the_first_solve() {
        const DT: f32 = 1.0 / 240.0;
        let (mut world, _, _) = settled_islands(DT, 120);
        world.step(DT).unwrap();
        assert!(!world.constraints.is_empty(), "the fixture solves nothing");
        world.solve();

        let bytes = alloc_probe::bytes_allocated_by(|| {
            for _ in 0..16 {
                world.solve();
            }
        });
        assert_eq!(
            bytes, 0,
            "16 solves over a steady island set asked the allocator for {bytes} bytes"
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
    fn bounded_contact_churn_allocates_nothing_after_warmup() {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        let floor = world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());
        let mut active =
            world.push_body(sphere_body_r3(Vec3::Y * 0.49, Vec3::ZERO, 0.5, 1.0).unwrap());
        let mut inactive =
            world.push_body(sphere_body_r3(Vec3::Y * 4.0, Vec3::ZERO, 0.5, 1.0).unwrap());
        world.update_manifolds();
        for _ in 0..4 {
            world.despawn_body(inactive).unwrap();
            let next =
                world.push_body(sphere_body_r3(Vec3::Y * 4.0, Vec3::ZERO, 0.5, 1.0).unwrap());
            let active_orientation = world.bodies[active].orientation;
            let next_orientation = world.bodies[next].orientation;
            world
                .set_pose(active, Vec3::Y * 4.0, active_orientation)
                .unwrap();
            world
                .set_pose(next, Vec3::Y * 0.49, next_orientation)
                .unwrap();
            world.update_manifolds();
            inactive = active;
            active = next;
        }
        assert!(world.manifolds.contains_key(&canonical_pair(floor, active)));

        let state = world.snapshot();
        world.restore(&state).unwrap();
        let retained = world.manifold_pool.len();
        for _ in 0..4 {
            world.restore(&state).unwrap();
            assert_eq!(world.manifold_pool.len(), retained);
        }
        let first_measured_generation = active.generation();

        let mut bytes = 0;
        for _ in 0..16 {
            world.despawn_body(inactive).unwrap();
            let next =
                world.push_body(sphere_body_r3(Vec3::Y * 4.0, Vec3::ZERO, 0.5, 1.0).unwrap());
            let active_orientation = world.bodies[active].orientation;
            let next_orientation = world.bodies[next].orientation;
            world
                .set_pose(active, Vec3::Y * 4.0, active_orientation)
                .unwrap();
            world
                .set_pose(next, Vec3::Y * 0.49, next_orientation)
                .unwrap();
            bytes += alloc_probe::bytes_allocated_by(|| world.update_manifolds());
            inactive = active;
            active = next;
        }
        assert!(active.generation() > first_measured_generation);
        assert!(world.manifolds.contains_key(&canonical_pair(floor, active)));
        assert_eq!(
            bytes, 0,
            "16 new-generation recontacts asked the allocator for {bytes} bytes"
        );
    }

    #[test]
    fn posed_halfspaces_move_sphere_and_hull_contacts_in_r3_and_r4() {
        let mut r3 = World::new(EuclideanR3);
        register_default_narrowphase(&mut r3.narrowphase);
        let wall3 = r3.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());
        let sphere3 =
            r3.push_body(sphere_body_r3(Vec3::new(2.25, 3.0, -2.0), Vec3::ZERO, 0.5, 1.0).unwrap());
        let hull3 = r3.push_body(
            box_body(Vec3::new(2.1, 3.0, 2.0), Vec3::ZERO, Vec3::splat(0.2), 1.0).unwrap(),
        );
        let pose3 = loam_math::Iso3 {
            rotation: glam::Quat::from_rotation_arc(Vec3::Y, Vec3::X),
            translation: Vec3::ZERO,
        };
        r3.set_pose(wall3, Vec3::X * 2.0, pose3).unwrap();
        r3.update_manifolds();
        for body in [sphere3, hull3] {
            let manifold = r3.manifold(canonical_pair(wall3, body)).unwrap();
            assert!(manifold.points.iter().any(|point| point.normal.x > 0.99));
        }

        let mut r4 = World::new(loam_math::EuclideanR4);
        crate::euclidean_r4::register_default_narrowphase(&mut r4.narrowphase);
        let wall4 =
            r4.push_body(crate::euclidean_r4::halfspace4_body_r4(glam::Vec4::Y, 0.0).unwrap());
        let sphere4 = r4.push_body(
            crate::euclidean_r4::sphere_body_r4(
                glam::Vec4::new(2.25, 3.0, -2.0, 0.0),
                glam::Vec4::ZERO,
                0.5,
                1.0,
            )
            .unwrap(),
        );
        let hull4 = r4.push_body(
            crate::euclidean_r4::polytope_body_r4(
                glam::Vec4::new(2.1, 3.0, 2.0, 0.0),
                glam::Vec4::ZERO,
                crate::euclidean_r4::tesseract_vertices(0.2),
                1.0,
            )
            .unwrap(),
        );
        let pose4 = loam_math::Iso4Flat {
            rotation: loam_math::Rotor4::from_rotation_arc(glam::Vec4::Y, glam::Vec4::X),
            translation: glam::Vec4::ZERO,
        };
        r4.set_pose(wall4, glam::Vec4::X * 2.0, pose4).unwrap();
        r4.update_manifolds();
        for body in [sphere4, hull4] {
            let manifold = r4.manifold(canonical_pair(wall4, body)).unwrap();
            assert!(manifold.points.iter().any(|point| point.normal.x > 0.99));
        }
    }

    #[test]
    fn r4_wall_bias_pushes_toward_the_near_face() {
        use crate::euclidean_r4::{sphere_body_r4, tesseract_vertices};
        use glam::Vec4;
        use loam_math::EuclideanR4;
        for x in [-0.092, -0.09, -0.088] {
            let mut world = World::new(EuclideanR4);
            crate::euclidean_r4::register_default_narrowphase(&mut world.narrowphase);
            world.push_body(
                BodyDef::fixed(
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
            world.step(1.0 / 240.0).unwrap();
            assert!(world.bodies[ball].velocity.x < 0.0, "x={x}");
        }
    }
}
