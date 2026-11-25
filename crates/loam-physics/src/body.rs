use std::ops::{Deref, Index, IndexMut};

use loam_math::Bivector;
use loam_shape::{Shape, ShapeKind};

use crate::error::EditError;
use crate::geometry::{ColliderRef, GeometryStore};
use crate::space::{PhysicsSpace, VectorOps};
use crate::state::StateError;

pub struct RigidBody<S: PhysicsSpace> {
    pub position: S::Point,
    pub velocity: S::Vector,
    pub orientation: S::Iso,
    pub angular_velocity: S::AngVel,

    mass: f32,
    inv_mass: f32,
    sleeping: bool,
    pub inertia: S::Inertia,

    collider: ColliderRef,

    pub restitution: f32,

    pub collision_group: u32,
    pub collision_mask: u32,
}

impl<S: PhysicsSpace> Clone for RigidBody<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: PhysicsSpace> Copy for RigidBody<S> {}

pub struct BodyDef<S: PhysicsSpace> {
    position: S::Point,
    velocity: S::Vector,
    angular_velocity: S::AngVel,
    collider: Shape,
    mass: f32,
    inertia: S::Inertia,
    restitution: f32,
    collision_group: u32,
    collision_mask: u32,
    asleep: bool,
}

impl<S: PhysicsSpace> BodyDef<S> {
    pub fn new(
        position: S::Point,
        velocity: S::Vector,
        collider: Shape,
        mass: f32,
        inertia: S::Inertia,
        space: &S,
    ) -> Result<Self, EditError> {
        if !valid_mass(mass) {
            return Err(EditError::InvalidMass);
        }
        if !space.valid_point(position) || !space.valid_vector(velocity) {
            return Err(EditError::NotFinite);
        }
        if !space.valid_inertia(inertia) {
            return Err(EditError::InvalidInertia);
        }
        check_collider(space, &collider, mass)?;
        Ok(Self {
            position,
            velocity,
            angular_velocity: <S::AngVel as Bivector>::zero(),
            collider,
            mass,
            inertia,
            restitution: 0.2,
            collision_group: GROUP_DEFAULT,
            collision_mask: MASK_ALL,
            asleep: false,
        })
    }

    pub fn fixed(
        position: S::Point,
        collider: Shape,
        inertia: S::Inertia,
        space: &S,
    ) -> Result<Self, EditError> {
        Self::new(position, VectorOps::zero(), collider, 0.0, inertia, space)
    }

    pub fn mass(&self) -> f32 {
        self.mass
    }

    pub fn collider(&self) -> &Shape {
        &self.collider
    }

    pub fn restitution(mut self, restitution: f32) -> Result<Self, EditError> {
        if !restitution.is_finite() || restitution < 0.0 {
            return Err(EditError::InvalidRestitution);
        }
        self.restitution = restitution;
        Ok(self)
    }

    pub fn velocity(
        mut self,
        velocity: S::Vector,
        angular_velocity: S::AngVel,
        space: &S,
    ) -> Result<Self, EditError> {
        if !space.valid_vector(velocity) || !space.valid_angular_velocity(angular_velocity) {
            return Err(EditError::NotFinite);
        }
        if !self.asleep {
            self.velocity = velocity;
            self.angular_velocity = angular_velocity;
        }
        Ok(self)
    }

    pub fn collision_filter(mut self, group: u32, mask: u32) -> Self {
        self.collision_group = group;
        self.collision_mask = mask;
        self
    }

    /// A dynamic body spawns asleep and at rest whatever velocity the definition is given.
    pub fn asleep(mut self) -> Self {
        self.velocity = VectorOps::zero();
        self.angular_velocity = <S::AngVel as Bivector>::zero();
        self.asleep = true;
        self
    }

    pub(crate) fn into_row(self, geometry: &mut GeometryStore, space: &S) -> RigidBody<S> {
        let inv_mass = if self.mass > 0.0 {
            1.0 / self.mass
        } else {
            0.0
        };
        RigidBody {
            sleeping: self.asleep && self.mass > 0.0,
            collision_group: self.collision_group,
            collision_mask: self.collision_mask,
            position: self.position,
            velocity: self.velocity,
            orientation: space.iso_identity(),
            angular_velocity: self.angular_velocity,
            mass: self.mass,
            inv_mass,
            inertia: self.inertia,
            collider: geometry.prepare(self.collider),
            restitution: self.restitution,
        }
    }
}

pub const GROUP_DEFAULT: u32 = 1;

pub const MASK_ALL: u32 = u32::MAX;

impl<S: PhysicsSpace> RigidBody<S> {
    pub fn mass(&self) -> f32 {
        self.mass
    }

    /// Sleeping bodies have zero effective inverse mass until woken.
    pub fn inv_mass(&self) -> f32 {
        if self.sleeping {
            0.0
        } else {
            self.inv_mass
        }
    }

    pub fn is_static(&self) -> bool {
        self.mass == 0.0
    }

    pub fn is_sleeping(&self) -> bool {
        self.sleeping
    }

    pub fn sleep(&mut self) {
        if !self.is_static() {
            self.velocity = VectorOps::zero();
            self.angular_velocity = S::AngVel::zero();
            self.sleeping = true;
        }
    }

    pub fn wake(&mut self) {
        self.sleeping = false;
    }

    /// Invalid mass leaves the body unchanged.
    pub fn set_mass(&mut self, mass: f32) -> bool {
        self.set_mass_properties(mass, self.inertia).is_ok()
    }

    pub fn collider(&self) -> ColliderRef {
        self.collider
    }

    pub(crate) fn set_mass_properties(
        &mut self,
        mass: f32,
        inertia: S::Inertia,
    ) -> Result<(), EditError> {
        if !valid_mass(mass) {
            return Err(EditError::InvalidMass);
        }
        if mass > 0.0 && is_halfspace_kind(self.collider.kind()) {
            return Err(EditError::DynamicHalfSpace);
        }
        self.mass = mass;
        self.inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        self.inertia = inertia;
        self.wake();
        Ok(())
    }

    pub(crate) fn replace_collider(
        &mut self,
        geometry: &mut GeometryStore,
        space: &S,
        collider: Shape,
        inertia: S::Inertia,
    ) -> Result<(), EditError> {
        check_collider(space, &collider, self.mass)?;
        let replacement = geometry.prepare(collider);
        geometry.release(self.collider);
        self.collider = replacement;
        self.inertia = inertia;
        self.wake();
        Ok(())
    }

    // Baraff 1997, "Physically Based Modeling: Rigid Body Simulation", colliding contact.
    pub fn apply_impulse(&mut self, impulse: S::Vector) {
        if self.is_static() {
            return;
        }
        self.wake();
        self.velocity = self.velocity + impulse * self.inv_mass;
    }

    // Baraff 1997, "Physically Based Modeling: Rigid Body Simulation", colliding contact.
    pub fn apply_impulse_at_point(&mut self, space: &S, impulse: S::Vector, point: S::Point) {
        if self.is_static() {
            return;
        }
        self.wake();
        self.velocity = self.velocity + impulse * self.inv_mass;
        let lever = space.log(self.position, point);
        let torque = space.wedge(lever, impulse);
        self.angular_velocity =
            self.angular_velocity + space.apply_inv_inertia(self.inertia, torque);
    }
}

pub(crate) fn valid_mass(mass: f32) -> bool {
    mass.is_finite() && mass >= 0.0 && (mass == 0.0 || mass.recip().is_finite())
}

fn is_halfspace_kind(kind: ShapeKind) -> bool {
    matches!(kind, ShapeKind::HalfSpace | ShapeKind::HalfSpace4D)
}

fn check_collider<S: PhysicsSpace>(
    space: &S,
    collider: &Shape,
    mass: f32,
) -> Result<(), EditError> {
    if mass > 0.0 && is_halfspace_kind(collider.kind()) {
        return Err(EditError::DynamicHalfSpace);
    }
    if !space.supports_collider(collider.kind()) {
        return Err(EditError::UnsupportedCollider);
    }
    if let Shape::Sphere { center, .. } = collider {
        if *center != glam::Vec3::ZERO {
            return Err(EditError::OffCentreSphere);
        }
    }
    if !collider.is_well_formed() {
        return Err(EditError::MalformedCollider);
    }
    Ok(())
}

/// A body handle is valid only in its owning arena and is ordered by slot, then generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BodyId {
    slot: u32,
    generation: u32,
}

impl BodyId {
    pub fn slot(self) -> u32 {
        self.slot
    }

    pub fn generation(self) -> u32 {
        self.generation
    }

    #[cfg(test)]
    pub(crate) fn forge(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }
}

const STALE_HANDLE: &str = "BodyId refers to a despawned body";

#[derive(Clone)]
struct Slot {
    generation: u32,
    dense: Option<u32>,
}

/// Compaction preserves handles; callers must not exchange bodies between mutable references.
pub struct BodyArena<S: PhysicsSpace> {
    dense: Vec<RigidBody<S>>,
    ids: Vec<BodyId>,
    slots: Vec<Slot>,
    free: Vec<u32>,
}

impl<S: PhysicsSpace> Clone for BodyArena<S> {
    fn clone(&self) -> Self {
        Self {
            dense: self.dense.clone(),
            ids: self.ids.clone(),
            slots: self.slots.clone(),
            free: self.free.clone(),
        }
    }
}

impl<S: PhysicsSpace> Default for BodyArena<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: PhysicsSpace> BodyArena<S> {
    pub fn new() -> Self {
        Self {
            dense: Vec::new(),
            ids: Vec::new(),
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    pub fn spawn(&mut self, body: RigidBody<S>) -> BodyId {
        let dense = self.dense.len() as u32;
        let slot = match self.free.pop() {
            Some(slot) => {
                self.slots[slot as usize].dense = Some(dense);
                slot
            }
            None => {
                self.slots.push(Slot {
                    generation: 0,
                    dense: Some(dense),
                });
                (self.slots.len() - 1) as u32
            }
        };
        let id = BodyId {
            slot,
            generation: self.slots[slot as usize].generation,
        };
        self.dense.push(body);
        self.ids.push(id);
        id
    }

    pub fn despawn(&mut self, id: BodyId) -> Option<RigidBody<S>> {
        let dense = self.dense_index(id)?;
        let slot = &mut self.slots[id.slot as usize];
        slot.dense = None;
        if let Some(next) = slot.generation.checked_add(1) {
            slot.generation = next;
            self.free.push(id.slot);
        }

        let removed = self.dense.swap_remove(dense);
        self.ids.swap_remove(dense);
        if let Some(&moved) = self.ids.get(dense) {
            self.slots[moved.slot as usize].dense = Some(dense as u32);
        }
        Some(removed)
    }

    /// Valid only until the next [`Self::despawn`].
    pub fn dense_index(&self, id: BodyId) -> Option<usize> {
        let slot = self.slots.get(id.slot as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.dense.map(|dense| dense as usize)
    }

    pub fn id_at(&self, dense: usize) -> BodyId {
        self.ids[dense]
    }

    pub fn get(&self, id: BodyId) -> Option<&RigidBody<S>> {
        self.dense_index(id).map(|dense| &self.dense[dense])
    }

    pub fn get_mut(&mut self, id: BodyId) -> Option<&mut RigidBody<S>> {
        match self.dense_index(id) {
            Some(dense) => Some(&mut self.dense[dense]),
            None => None,
        }
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, RigidBody<S>> {
        self.dense.iter_mut()
    }

    pub(crate) fn dense_mut(&mut self) -> &mut [RigidBody<S>] {
        &mut self.dense
    }

    pub(crate) fn validate(&self, space: &S, geometry: &GeometryStore) -> Result<(), StateError>
    where
        S::AngVel: PartialEq,
    {
        if self.dense.len() != self.ids.len() {
            return Err(StateError::InvalidBodyArena);
        }
        let mut occupied = vec![false; self.slots.len()];
        for (dense, (&id, body)) in self.ids.iter().zip(&self.dense).enumerate() {
            let Ok(dense_index) = u32::try_from(dense) else {
                return Err(StateError::InvalidBodyArena);
            };
            let Some(slot) = self.slots.get(id.slot as usize) else {
                return Err(StateError::InvalidBodyArena);
            };
            if occupied[id.slot as usize]
                || slot.generation != id.generation
                || slot.dense != Some(dense_index)
            {
                return Err(StateError::InvalidBodyArena);
            }
            occupied[id.slot as usize] = true;
            let expected_inv_mass = if body.mass > 0.0 {
                body.mass.recip()
            } else {
                0.0
            };
            let Some(collider) = geometry.get(body.collider) else {
                return Err(StateError::InvalidGeometry);
            };
            if !valid_mass(body.mass)
                || body.inv_mass != expected_inv_mass
                || !space.valid_initial_state(body.position, body.velocity, body.inertia)
                || !space.valid_orientation(body.orientation)
                || !space.valid_angular_velocity(body.angular_velocity)
                || (body.sleeping
                    && (body.mass == 0.0
                        || body.velocity != S::Vector::zero()
                        || body.angular_velocity != S::AngVel::zero()))
                || !body.restitution.is_finite()
                || body.restitution < 0.0
                || body.collider.kind() != collider.kind()
                || check_collider(space, collider, body.mass).is_err()
            {
                return Err(StateError::InvalidBody);
            }
        }
        let mut free = vec![false; self.slots.len()];
        for &slot in &self.free {
            let Some(entry) = self.slots.get(slot as usize) else {
                return Err(StateError::InvalidBodyArena);
            };
            if free[slot as usize] || entry.dense.is_some() || occupied[slot as usize] {
                return Err(StateError::InvalidBodyArena);
            }
            free[slot as usize] = true;
        }
        for (slot, entry) in self.slots.iter().enumerate() {
            if occupied[slot] != entry.dense.is_some()
                || (!occupied[slot] && entry.generation != u32::MAX && !free[slot])
            {
                return Err(StateError::InvalidBodyArena);
            }
        }
        geometry.validate(self.dense.iter().map(|body| body.collider))
    }
}

impl<S: PhysicsSpace> Deref for BodyArena<S> {
    type Target = [RigidBody<S>];

    fn deref(&self) -> &Self::Target {
        &self.dense
    }
}

impl<S: PhysicsSpace> Index<BodyId> for BodyArena<S> {
    type Output = RigidBody<S>;

    fn index(&self, id: BodyId) -> &RigidBody<S> {
        self.get(id).unwrap_or_else(|| panic!("{STALE_HANDLE}"))
    }
}

impl<S: PhysicsSpace> IndexMut<BodyId> for BodyArena<S> {
    fn index_mut(&mut self, id: BodyId) -> &mut RigidBody<S> {
        self.get_mut(id).unwrap_or_else(|| panic!("{STALE_HANDLE}"))
    }
}

impl<S: PhysicsSpace> Index<usize> for BodyArena<S> {
    type Output = RigidBody<S>;

    fn index(&self, dense: usize) -> &RigidBody<S> {
        &self.dense[dense]
    }
}

impl<S: PhysicsSpace> IndexMut<usize> for BodyArena<S> {
    fn index_mut(&mut self, dense: usize) -> &mut RigidBody<S> {
        &mut self.dense[dense]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::World;
    use glam::{Vec3, Vec4};
    use loam_math::{Bivector3, Bivector4, EuclideanR3, EuclideanR4};

    fn row<S: PhysicsSpace>(def: BodyDef<S>, space: &S) -> RigidBody<S> {
        def.into_row(&mut GeometryStore::default(), space)
    }

    #[test]
    fn invalid_definition() {
        let sphere = |position, velocity, mass, inertia| {
            BodyDef::new(
                position,
                velocity,
                Shape::sphere_at_origin(0.5),
                mass,
                inertia,
                &EuclideanR3,
            )
            .err()
        };
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for (position, velocity, mass, inertia, cause) in [
                (Vec3::ZERO, Vec3::ZERO, invalid, 1.0, EditError::InvalidMass),
                (
                    Vec3::splat(invalid),
                    Vec3::ZERO,
                    1.0,
                    1.0,
                    EditError::NotFinite,
                ),
                (
                    Vec3::ZERO,
                    Vec3::splat(invalid),
                    1.0,
                    1.0,
                    EditError::NotFinite,
                ),
                (
                    Vec3::ZERO,
                    Vec3::ZERO,
                    1.0,
                    invalid,
                    EditError::InvalidInertia,
                ),
            ] {
                assert_eq!(sphere(position, velocity, mass, inertia), Some(cause));
            }
        }

        let dynamic_halfspace4 = BodyDef::new(
            Vec4::ZERO,
            Vec4::ZERO,
            Shape::HalfSpace4D {
                normal: Vec4::Y,
                offset: 0.0,
            },
            1.0,
            1.0,
            &EuclideanR4,
        );
        assert_eq!(dynamic_halfspace4.err(), Some(EditError::DynamicHalfSpace));
        assert_eq!(
            crate::euclidean_r4::halfspace4_body_r4(Vec4::ZERO, 0.0).err(),
            Some(EditError::MalformedCollider),
            "a half-space without a normal"
        );
        assert_eq!(
            crate::euclidean_r3::halfspace_body_r3(Vec3::ZERO, 0.0).err(),
            Some(EditError::MalformedCollider),
            "a half-space without a normal"
        );

        let mut world = World::new(EuclideanR3);
        let id = world.push_body(def_r3(Vec3::ZERO, 2.0, 1.0));
        for (collider, cause) in [
            (
                Shape::ConvexPolytope3D { vertices: vec![] },
                EditError::MalformedCollider,
            ),
            (
                Shape::Box3 {
                    half_extents: Vec3::ONE,
                },
                EditError::UnsupportedCollider,
            ),
            (
                Shape::sphere_at_origin(f32::NAN),
                EditError::MalformedCollider,
            ),
            (Shape::sphere_at(Vec3::X, 1.0), EditError::OffCentreSphere),
            (
                Shape::HalfSpace {
                    normal: Vec3::Y,
                    offset: 0.0,
                },
                EditError::DynamicHalfSpace,
            ),
        ] {
            let built = BodyDef::new(
                Vec3::ZERO,
                Vec3::ZERO,
                collider.clone(),
                2.0,
                1.0,
                &EuclideanR3,
            );
            assert_eq!(built.err(), Some(cause));
            assert_eq!(
                world.set_collider(id, collider, 1.0),
                Err(cause),
                "an edit and a definition disagree on the cause"
            );
            let kept = world.collider(&world.bodies[id]);
            assert!(matches!(kept, Some(Shape::Sphere { radius, .. }) if *radius == 0.5));
        }
    }

    #[test]
    fn sleeping_body() {
        let mut body = body_r3(Vec3::ZERO, 2.0, 1.0);
        body.velocity = Vec3::Y;
        body.angular_velocity = Bivector3::new(1.0, 0.0, 0.0);
        body.sleep();
        assert!(!body.is_static());
        assert_eq!(body.mass(), 2.0);
        assert_eq!(body.inv_mass(), 0.0);
        crate::integrate_body(&EuclideanR3, &mut body, 1.0);
        assert_eq!(body.position, Vec3::ZERO);
        body.apply_impulse(Vec3::X * 4.0);
        assert!(!body.is_sleeping());
        assert_eq!(body.inv_mass(), 0.5);
        assert_eq!(body.velocity, Vec3::X * 2.0);
        assert_eq!(body.angular_velocity, Bivector3::ZERO);
    }

    fn def_r3(position: Vec3, mass: f32, inertia: f32) -> BodyDef<EuclideanR3> {
        BodyDef::new(
            position,
            Vec3::ZERO,
            Shape::sphere_at_origin(0.5),
            mass,
            inertia,
            &EuclideanR3,
        )
        .unwrap()
    }

    fn body_r3(position: Vec3, mass: f32, inertia: f32) -> RigidBody<EuclideanR3> {
        row(def_r3(position, mass, inertia), &EuclideanR3)
    }

    fn body_r4(position: Vec4, mass: f32, inertia: f32) -> RigidBody<EuclideanR4> {
        row(
            BodyDef::new(
                position,
                Vec4::ZERO,
                Shape::sphere_at_origin(0.5),
                mass,
                inertia,
                &EuclideanR4,
            )
            .unwrap(),
            &EuclideanR4,
        )
    }

    #[test]
    fn impulse_at_point() {
        let position = Vec3::new(2.0, -1.0, 3.0);
        let impulse = Vec3::new(8.0, -4.0, 2.0);

        let mut at_point = body_r3(position, 4.0, 0.5);
        at_point.apply_impulse_at_point(&EuclideanR3, impulse, position);

        let mut linear = body_r3(position, 4.0, 0.5);
        linear.apply_impulse(impulse);

        assert_eq!(at_point.velocity, linear.velocity);
        assert_eq!(at_point.angular_velocity, Bivector3::ZERO);

        let mut body = body_r3(Vec3::ZERO, 2.0, 0.5);
        body.apply_impulse_at_point(
            &EuclideanR3,
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        );

        assert_eq!(body.velocity, Vec3::new(1.5, 0.0, 0.0));
        assert_eq!(body.angular_velocity, Bivector3::new(-12.0, 0.0, 0.0));

        let mut body = body_r4(Vec4::ZERO, 2.0, 0.5);
        body.apply_impulse_at_point(
            &EuclideanR4,
            Vec4::new(3.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 2.0),
        );

        assert_eq!(body.velocity, Vec4::new(1.5, 0.0, 0.0, 0.0));
        let expected = Bivector4 {
            xw: -12.0,
            ..Bivector4::ZERO
        };
        assert_eq!(body.angular_velocity, expected);

        let mut fixed = row(
            BodyDef::<EuclideanR3>::fixed(
                Vec3::ZERO,
                Shape::sphere_at_origin(0.5),
                1.0,
                &EuclideanR3,
            )
            .unwrap(),
            &EuclideanR3,
        );
        fixed.apply_impulse(Vec3::new(5.0, 5.0, 5.0));
        fixed.apply_impulse_at_point(
            &EuclideanR3,
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        assert_eq!(fixed.velocity, Vec3::ZERO);
        assert_eq!(fixed.angular_velocity, Bivector3::ZERO);
    }
}
