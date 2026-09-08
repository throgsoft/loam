//! The four Shapes-view render paths (SDF upload, section caps, wireframe
//! overlay, point sprites) source their pose here, so they cannot disagree
//! about where a body is.

use glam::{Vec2, Vec3, Vec4};
use loam_math::{Bivector4, EuclideanR4, Rotor, Rotor4};
use loam_physics::euclidean_r4::{
    ball4_inertia, register_default_narrowphase, regular_polytope4_inertia, sphere_body_r4,
};
use loam_physics::{Collider, World};
use loam_render::raymarch::RaymarchShape;

use crate::catalog::ShapeEntry;
use crate::spins::SlotSpins;
use crate::state::body_position;

#[cfg(test)]
const PHYSICS_DT: f32 = 1.0 / 60.0;

const BODY_MASS: f32 = 1.0;

#[cfg(test)]
const THROW_SPEED: f32 = 8.1;

const VELOCITY_DECAY_TAU: f32 = 0.6;

const REST_SPEED: f32 = 0.02;
const REST_ANGULAR_SPEED: f32 = 0.02;

// y-up NDC: the inverse of what `loam_camera::Camera::ray_from_ndc` consumes.
pub(crate) fn ndc_from_pixels(pixels: Vec2, viewport: (u32, u32)) -> Vec2 {
    let (width, height) = (viewport.0 as f32, viewport.1 as f32);
    Vec2::new(2.0 * pixels.x / width - 1.0, 1.0 - 2.0 * pixels.y / height)
}

// `Rotor4` multiplies left-first, so the world-frame physics rotor is the right factor.
pub(crate) fn composed_rotor(spin: Rotor4, orientation: Rotor4) -> Rotor4 {
    spin * orientation
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct BodyPose {
    pub(crate) position: Vec4,
    pub(crate) rotor: Rotor4,
}

impl BodyPose {
    // Applied after projection, so a Perspective4D divide never scales x.
    pub(crate) fn position_r3(&self) -> Vec3 {
        self.position.truncate()
    }

    // Off-origin frame: no caller may read a vertex `length()` as the circumradius.
    pub(crate) fn body_local(&self, canonical: Vec4, size: f32) -> Vec4 {
        size * self.rotor.apply(canonical) + Vec4::W * self.position.w
    }
}

#[derive(Copy, Clone, PartialEq)]
struct SyncedSlot {
    shape: RaymarchShape,
    spin: Rotor4,
    size: f32,
}

pub(crate) struct PlaygroundPhysics {
    pub(crate) world: World<EuclideanR4>,
    synced: Vec<Option<SyncedSlot>>,
    hull_scratch: Vec<Vec4>,
    spawn_scratch: Vec<loam_physics::RigidBody<EuclideanR4>>,
}

impl PlaygroundPhysics {
    pub(crate) fn new(slots: usize, radius: f32) -> Option<Self> {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        let mut physics = Self {
            world,
            synced: Vec::new(),
            hull_scratch: Vec::new(),
            spawn_scratch: Vec::new(),
        };
        physics.respawn(slots, radius)?;
        Some(physics)
    }

    pub(crate) fn respawn(&mut self, slots: usize, radius: f32) -> Option<()> {
        self.spawn_scratch.clear();
        for slot in 0..slots {
            let position = Vec4::from_array(body_position(slot, slots));
            self.spawn_scratch
                .push(sphere_body_r4(position, Vec4::ZERO, radius, BODY_MASS)?);
        }
        // A fresh arena restarts generations at 0 and would alias held handles.
        while let Some(last) = self.world.bodies.len().checked_sub(1) {
            let id = self.world.bodies.id_at(last);
            self.world.despawn_body(id);
        }
        for body in self.spawn_scratch.drain(..) {
            self.world.push_body(body);
        }
        self.synced.clear();
        Some(())
    }

    // The UI spin is baked into the hull: `PosedHull4` applies `orientation.rotation` alone.
    pub(crate) fn sync(&mut self, row: &[ShapeEntry], spins: &SlotSpins, size: f32) -> bool {
        if self.world.bodies.len() != row.len() && self.respawn(row.len(), size).is_none() {
            return false;
        }
        self.synced.resize(row.len(), None);
        let mut accepted = true;
        for (slot, entry) in row.iter().enumerate() {
            let spin = spins.rotor(slot);
            let desired = SyncedSlot {
                shape: entry.shape,
                spin,
                size,
            };
            if self.synced[slot] == Some(desired) {
                continue;
            }
            let body = &mut self.world.bodies[slot];
            let (collider, inertia) = if let Some(polytope) = entry.collider_polytope() {
                self.hull_scratch.clear();
                self.hull_scratch.extend(
                    polytope
                        .topology()
                        .vertices
                        .iter()
                        .map(|v| size * spin.apply(*v)),
                );
                (
                    Collider::ConvexPolytope4D {
                        vertices: std::mem::take(&mut self.hull_scratch),
                    },
                    regular_polytope4_inertia(polytope, body.mass(), size),
                )
            } else {
                (
                    Collider::sphere_at_origin(size),
                    ball4_inertia(body.mass(), size),
                )
            };
            let result = if inertia.is_finite() && inertia >= 0.0 {
                body.set_collider(&EuclideanR4, collider)
            } else {
                Err(collider)
            };
            let recycled = match result {
                Ok(previous) => {
                    body.inertia = inertia;
                    self.synced[slot] = Some(desired);
                    previous
                }
                Err(rejected) => {
                    accepted = false;
                    rejected
                }
            };
            if let Collider::ConvexPolytope4D { vertices } = recycled {
                self.hull_scratch = vertices;
            }
        }
        accepted
    }

    pub(crate) fn at_rest(&self) -> bool {
        self.world
            .bodies
            .iter()
            .all(|b| b.velocity == Vec4::ZERO && b.angular_velocity.magnitude_squared() == 0.0)
    }

    // Authored layouts can overlap; physics starts only after an interaction.
    pub(crate) fn tick(&mut self, dt: f32) {
        if self.at_rest() {
            return;
        }
        self.world.step(dt);
        self.damp((-dt / VELOCITY_DECAY_TAU).exp());
    }

    #[cfg(test)]
    pub(crate) fn step(&mut self, ticks: usize) {
        for _ in 0..ticks {
            self.tick(PHYSICS_DT);
        }
    }

    fn damp(&mut self, decay: f32) {
        for body in self.world.bodies.iter_mut() {
            body.velocity *= decay;
            if body.velocity.length_squared() < REST_SPEED * REST_SPEED {
                body.velocity = Vec4::ZERO;
            }
            body.angular_velocity = body.angular_velocity * decay;
            if body.angular_velocity.magnitude_squared() < REST_ANGULAR_SPEED * REST_ANGULAR_SPEED {
                body.angular_velocity = Bivector4::ZERO;
            }
        }
    }

    pub(crate) fn pose(&self, slot: usize, slots: usize, spin: Rotor4) -> BodyPose {
        assert_eq!(
            self.world.bodies.len(),
            slots,
            "physics world not synced to the rendered row"
        );
        let body = &self.world.bodies[slot];
        BodyPose {
            position: body.position,
            rotor: composed_rotor(spin, body.orientation.rotation),
        }
    }

    pub(crate) fn body_frame(
        &self,
        slot: usize,
        slots: usize,
        spin: Rotor4,
        canonical: &[Vec4],
        size: f32,
        out: &mut Vec<Vec4>,
    ) -> Vec3 {
        let pose = self.pose(slot, slots, spin);
        out.clear();
        out.extend(canonical.iter().map(|v| pose.body_local(*v, size)));
        pose.position_r3()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::{Bivector, Plane4};
    use loam_shape::polytope::Polytope4;

    const RADIUS: f32 = crate::consts::BODY_SIZE;

    fn rotor_at(plane: Plane4, angle: f32) -> Rotor4 {
        (plane.unit_bivector() * angle).exp().normalize()
    }

    fn row_of(shape: RaymarchShape, slots: usize) -> Vec<ShapeEntry> {
        let entry = *crate::catalog::SHAPE_CATALOG
            .iter()
            .find(|e| e.shape == shape)
            .expect("every RaymarchShape has a catalog entry");
        vec![entry; slots]
    }

    fn synced_row(
        shape: RaymarchShape,
        slots: usize,
        size: f32,
        spin: Rotor4,
    ) -> (PlaygroundPhysics, Vec<ShapeEntry>, SlotSpins) {
        let row = row_of(shape, slots);
        let spins = SlotSpins::uniform(slots, spin);
        let mut physics = PlaygroundPhysics::new(slots, size).unwrap();
        physics.sync(&row, &spins, size);
        (physics, row, spins)
    }

    fn sweep_spins() -> [Rotor4; 4] {
        [
            Rotor4::IDENTITY,
            rotor_at(Plane4::Xz, 1.1),
            rotor_at(Plane4::Xw, 0.6),
            rotor_at(Plane4::Xy, 0.7) * rotor_at(Plane4::Zw, 0.4),
        ]
    }

    #[test]
    fn rejected_respawn_and_sync_preserve_live_handles_geometry_and_cache() {
        let shape = RaymarchShape::Polytope(Polytope4::Tesseract);
        let (mut physics, row, spins) = synced_row(shape, 1, RADIUS, Rotor4::IDENTITY);
        let id = physics.world.bodies.id_at(0);
        let Collider::ConvexPolytope4D { vertices } = physics.world.bodies[id].collider() else {
            panic!("hull fixture")
        };
        let before = vertices.clone();
        let cache = physics.synced.clone();
        assert!(physics.respawn(2, f32::NAN).is_none());
        assert_eq!(physics.world.bodies.len(), 1);
        assert!(physics.world.bodies.get(id).is_some());
        assert!(!physics.sync(&row, &spins, f32::NAN));
        let Collider::ConvexPolytope4D { vertices } = physics.world.bodies[id].collider() else {
            panic!("rejected hull replaced valid hull")
        };
        assert_eq!(*vertices, before);
        assert!(physics.synced == cache);
        assert!(physics.sync(&row, &spins, RADIUS * 2.0));
        let Collider::ConvexPolytope4D { vertices } = physics.world.bodies[id].collider() else {
            panic!("hull fixture")
        };
        for (got, old) in vertices.iter().zip(before) {
            assert!((*got - old * 2.0).length() < 1e-6);
        }
    }

    #[test]
    fn overlapping_layout_at_rest_is_never_pushed_apart() {
        let slots = 4;
        let mut physics = PlaygroundPhysics::new(slots, crate::consts::BODY_X_SPACING).unwrap();
        physics.step(120);
        for slot in 0..slots {
            assert_eq!(
                physics
                    .pose(slot, slots, Rotor4::IDENTITY)
                    .position
                    .to_array(),
                body_position(slot, slots)
            );
        }
    }

    #[test]
    fn body_local_carries_the_body_w_into_the_slice_frame() {
        let v = Vec4::new(0.5, -0.25, 0.125, 0.75);
        let flat = BodyPose {
            position: Vec4::new(1.0, 0.9, 0.0, 0.0),
            rotor: Rotor4::IDENTITY,
        };
        assert_eq!(flat.body_local(v, RADIUS), RADIUS * v);
        assert_eq!(flat.position_r3(), Vec3::new(1.0, 0.9, 0.0));

        let lifted = BodyPose {
            position: Vec4::new(1.0, 0.9, 0.0, 0.25),
            rotor: Rotor4::IDENTITY,
        };
        assert_eq!(
            lifted.body_local(v, RADIUS),
            RADIUS * v + Vec4::new(0.0, 0.0, 0.0, 0.25)
        );
    }

    #[test]
    fn an_impulse_drives_its_own_slot_and_only_that_slot() {
        let slots = 3;
        let ticks = 30;
        let mut physics = PlaygroundPhysics::new(slots, RADIUS).unwrap();
        let impulse = Vec4::new(0.0, 0.0, 0.0, 2.0);
        physics.world.bodies[1].apply_impulse(impulse);
        assert!(!physics.at_rest());
        physics.step(ticks);

        let decay = (-PHYSICS_DT / VELOCITY_DECAY_TAU).exp();
        let travel = PHYSICS_DT * (1.0 - decay.powi(ticks as i32)) / (1.0 - decay);
        let expected = Vec4::from_array(body_position(1, slots)) + impulse * travel;
        let moved = physics.pose(1, slots, Rotor4::IDENTITY).position;
        assert!(
            (moved - expected).length() < 1e-5,
            "struck pose {moved} away from {expected}"
        );
        for slot in [0, 2] {
            assert_eq!(
                physics
                    .pose(slot, slots, Rotor4::IDENTITY)
                    .position
                    .to_array(),
                body_position(slot, slots),
                "untouched slot {slot} moved"
            );
        }
    }

    #[test]
    fn angular_impulse_composes_after_the_ui_spin() {
        let mut physics = PlaygroundPhysics::new(1, RADIUS).unwrap();
        let layout = Vec4::from_array(body_position(0, 1));
        physics.world.bodies[0].apply_impulse_at_point(
            &EuclideanR4,
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            layout + Vec4::W * 0.5,
        );
        physics.step(10);

        let orientation = physics.world.bodies[0].orientation.rotation;
        assert_ne!(
            orientation,
            Rotor4::IDENTITY,
            "off-centre impulse produced no rotation"
        );

        let spin = rotor_at(Plane4::Xy, 0.9);
        let composed = physics.pose(0, 1, spin).rotor;
        let v = Vec4::new(0.3, -0.2, 0.9, 0.1);
        let staged = orientation.apply(spin.apply(v));
        assert!(
            (composed.apply(v) - staged).length() < 1e-5,
            "composition order is not spin-then-physics"
        );
    }

    #[test]
    fn sync_respawns_only_when_the_slot_count_changes() {
        let shape = RaymarchShape::Polytope(Polytope4::Tesseract);
        let (mut physics, row, spins) = synced_row(shape, 3, RADIUS, Rotor4::IDENTITY);
        physics.world.bodies[0].apply_impulse(Vec4::new(0.0, 0.0, 0.0, 1.0));
        physics.step(10);
        let in_flight = physics.pose(0, 3, Rotor4::IDENTITY).position;

        physics.sync(&row, &spins, RADIUS);
        assert_eq!(
            physics.pose(0, 3, Rotor4::IDENTITY).position,
            in_flight,
            "same-count sync cancelled an impulse"
        );

        physics.sync(
            &row_of(shape, 4),
            &SlotSpins::uniform(4, Rotor4::IDENTITY),
            RADIUS,
        );
        assert!(physics.at_rest(), "respawn left motion behind");
        for slot in 0..4 {
            assert_eq!(
                physics.pose(slot, 4, Rotor4::IDENTITY).position.to_array(),
                body_position(slot, 4)
            );
        }
    }

    #[test]
    fn every_polychoron_collides_as_its_own_hull_and_the_smooth_solids_do_not() {
        for entry in crate::catalog::SHAPE_CATALOG {
            let (physics, ..) = synced_row(entry.shape, 1, RADIUS, Rotor4::IDENTITY);
            let expected_hull = Polytope4::ALL
                .iter()
                .any(|p| entry.shape == RaymarchShape::Polytope(*p));
            let got_hull = matches!(
                physics.world.bodies[0].collider(),
                Collider::ConvexPolytope4D { .. }
            );
            assert_eq!(
                got_hull,
                expected_hull,
                "{} collided as {:?}",
                entry.label,
                physics.world.bodies[0].collider()
            );
        }
    }

    #[test]
    fn the_hull_collider_is_the_shape_the_row_draws_under_its_ui_spin() {
        let orientation = rotor_at(Plane4::Yw, 0.8);
        for spin in sweep_spins() {
            for polytope in Polytope4::ALL {
                let (mut physics, ..) =
                    synced_row(RaymarchShape::Polytope(polytope), 1, RADIUS, spin);
                physics.world.bodies[0].orientation.rotation = orientation;
                let pose = physics.pose(0, 1, spin);
                let Collider::ConvexPolytope4D { vertices } = physics.world.bodies[0].collider()
                else {
                    panic!("{polytope:?} lost its hull");
                };
                let canonical = polytope.topology().vertices;
                assert_eq!(vertices.len(), canonical.len());
                for (local, v) in vertices.iter().zip(canonical) {
                    let collided = orientation.apply(*local);
                    let drawn = pose.body_local(*v, RADIUS);
                    assert!(
                        (collided - drawn).length() < 1e-5,
                        "{polytope:?} collides at {collided} and draws at {drawn}"
                    );
                }
            }
        }
    }

    #[test]
    fn spinning_hulls_reuse_storage_and_unchanged_rows_preserve_inertia() {
        let shape = RaymarchShape::Polytope(Polytope4::Cell24);
        let (mut physics, row, _) = synced_row(shape, 2, RADIUS, Rotor4::IDENTITY);
        let changes: Vec<_> = [1, 2, 3, 199]
            .into_iter()
            .map(|step| SlotSpins::uniform(2, rotor_at(Plane4::Xw, step as f32 * 0.03)))
            .collect();
        physics.sync(&row, &changes[0], RADIUS);
        let bytes = crate::alloc_probe::bytes_allocated_by(|| {
            for spins in &changes[1..] {
                physics.sync(&row, spins, RADIUS);
            }
        });
        assert_eq!(bytes, 0);

        let spins = SlotSpins::uniform(2, rotor_at(Plane4::Xw, 199.0 * 0.03));
        physics.world.bodies[0].inertia = 0.0;
        physics.sync(&row, &spins, RADIUS);
        assert_eq!(
            physics.world.bodies[0].inertia, 0.0,
            "unchanged row resynced"
        );
        physics.sync(&row, &spins, RADIUS * 1.5);
        assert!(
            physics.world.bodies[0].inertia > 0.0,
            "a size edit was skipped"
        );
    }

    fn facing_pair(
        shape: RaymarchShape,
        spin: Rotor4,
        separation: f32,
        lateral: f32,
    ) -> PlaygroundPhysics {
        let (mut physics, ..) = synced_row(shape, 2, RADIUS, spin);
        let origin = physics.world.bodies[0].position;
        physics.world.bodies[1].position = origin + Vec4::new(separation, lateral, 0.0, 0.0);
        physics
    }

    fn peak_struck_spin(shape: RaymarchShape, spin: Rotor4) -> f32 {
        let (mut physics, ..) = synced_row(shape, 2, RADIUS, spin);
        physics.world.bodies[0].apply_impulse(flick(1.0, RIGHT));
        let mut peak = 0.0_f32;
        for _ in 0..120 {
            physics.step(1);
            peak = peak.max(physics.world.bodies[1].angular_velocity.magnitude());
        }
        peak
    }

    #[test]
    fn a_head_on_hull_collision_spins_the_struck_body_where_a_ball_pair_cannot() {
        let spin = rotor_at(Plane4::Xz, 1.1);
        let peak = peak_struck_spin(RaymarchShape::Polytope(Polytope4::Pentatope), spin);
        assert!(peak > 0.0, "an asymmetric hull collision produced no spin");
        assert_eq!(
            peak_struck_spin(RaymarchShape::ThreeSphere, spin),
            0.0,
            "the smooth solids keep the ball collider, which has no lever to spin on"
        );
    }

    #[test]
    fn a_hull_collision_pushes_the_struck_body_off_the_w_zero_slice() {
        {
            let polytope = Polytope4::Pentatope;
            let shape = RaymarchShape::Polytope(polytope);
            let mut leaked = 0.0_f32;
            for spin in sweep_spins() {
                let (mut physics, ..) = synced_row(shape, 2, RADIUS, spin);
                physics.world.bodies[0].apply_impulse(flick(1.0, RIGHT));
                assert_eq!(
                    physics.world.bodies[0].velocity.w, 0.0,
                    "the impulse itself left the slice"
                );
                for _ in 0..120 {
                    physics.step(1);
                    leaked = leaked.max(physics.world.bodies[1].position.w.abs());
                }
            }
            assert!(
                leaked > 1e-3,
                "no spin in the sweep moved a struck {polytope:?} off the \
                 slice (best |w| = {leaked})"
            );
        }

        let (mut physics, ..) = synced_row(RaymarchShape::ThreeSphere, 2, RADIUS, Rotor4::IDENTITY);
        physics.world.bodies[0].apply_impulse(flick(1.0, RIGHT));
        physics.step(120);
        assert_eq!(physics.world.bodies[1].position.w, 0.0);
    }

    #[test]
    fn overlapping_boxes_separate_then_stop_drifting() {
        let mut physics = facing_pair(
            RaymarchShape::Polytope(Polytope4::Tesseract),
            Rotor4::IDENTITY,
            0.75 * RADIUS,
            0.0,
        );
        physics.world.bodies[1].apply_impulse(flick(0.0625, RIGHT));
        let mut touched = false;
        for _ in 0..400 {
            physics.step(1);
            touched |= !physics.world.manifolds.is_empty();
            if physics.at_rest() {
                break;
            }
        }
        assert!(touched && physics.at_rest());
        let separation = physics.world.bodies[1].position.x - physics.world.bodies[0].position.x;
        assert!(
            separation >= RADIUS - 1e-4,
            "boxes stopped with overlap {separation}"
        );
        let resting: Vec<_> = physics
            .world
            .bodies
            .iter()
            .map(|body| body.position)
            .collect();
        physics.step(2);
        assert!(physics
            .world
            .bodies
            .iter()
            .zip(&resting)
            .all(|(body, position)| body.position == *position));
    }

    fn tumbling(slots: usize) -> PlaygroundPhysics {
        let mut physics = PlaygroundPhysics::new(slots, RADIUS).unwrap();
        let layout = Vec4::from_array(body_position(1, slots));
        physics.world.bodies[1].apply_impulse_at_point(
            &EuclideanR4,
            Vec4::new(0.4, 0.0, 0.0, 1.2),
            layout + Vec4::W * 0.5,
        );
        physics.step(24);
        physics
    }

    #[test]
    fn body_frame_reports_the_live_pose_not_the_authored_spin() {
        let slots = 3;
        let physics = tumbling(slots);
        let spin = rotor_at(Plane4::Xy, 0.7);
        let size = 0.4;
        let canonical = [
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 0.6, -0.3, 0.2),
        ];

        let mut out = Vec::new();
        let origin = physics.body_frame(1, slots, spin, &canonical, size, &mut out);

        let body = &physics.world.bodies[1];
        let composed = composed_rotor(spin, body.orientation.rotation);
        assert_ne!(
            body.orientation.rotation,
            Rotor4::IDENTITY,
            "the impulse produced no rotation, so the pin below is vacuous"
        );
        assert_eq!(origin, body.position.truncate());
        assert_ne!(
            origin,
            Vec4::from_array(body_position(1, slots)).truncate(),
            "R³ translate still reads the static layout"
        );
        for (i, v) in canonical.iter().enumerate() {
            assert_eq!(
                out[i],
                size * composed.apply(*v) + Vec4::W * body.position.w
            );
            assert_ne!(
                out[i],
                size * spin.apply(*v),
                "frame vertex {i} still reads the authored spin alone"
            );
        }
    }

    #[test]
    #[should_panic(expected = "physics world not synced to the rendered row")]
    fn pose_rejects_a_row_the_world_was_not_synced_to() {
        let physics = PlaygroundPhysics::new(3, RADIUS).unwrap();
        physics.pose(0, 4, Rotor4::IDENTITY);
    }

    const RIGHT: Vec3 = Vec3::X;
    const UP: Vec3 = Vec3::Y;

    // `m · speed · direction`: `apply_impulse` divides by the same mass.
    fn flick(fraction: f32, direction: Vec3) -> Vec4 {
        (direction * (fraction * THROW_SPEED * BODY_MASS)).extend(0.0)
    }

    #[test]
    fn an_impulse_advances_the_world_and_returns_it_to_the_at_rest_fixpoint() {
        let mut physics = PlaygroundPhysics::new(1, RADIUS).unwrap();
        let layout = Vec4::from_array(body_position(0, 1));
        physics.world.bodies[0].apply_impulse(flick(1.0, RIGHT));
        assert!(!physics.at_rest(), "an impulse left the world at rest");

        physics.step(6);
        let moved = physics.pose(0, 1, Rotor4::IDENTITY).position;
        assert!(
            (moved - layout).length() > 0.1,
            "six ticks of a full-power impulse moved the body only {}",
            (moved - layout).length()
        );

        // Decay from THROW_SPEED to REST_SPEED takes ~3.7 s.
        physics.step(600);
        assert!(physics.at_rest(), "the throw never decayed back to rest");
        let settled = physics.pose(0, 1, Rotor4::IDENTITY).position;
        physics.step(2);
        assert_eq!(
            physics.pose(0, 1, Rotor4::IDENTITY).position,
            settled,
            "a settled body kept drifting"
        );
        physics.world.bodies[0].apply_impulse(flick(0.8, UP));
        assert!(!physics.at_rest());
        physics.step(6);
        assert!(physics.pose(0, 1, Rotor4::IDENTITY).position.y > settled.y + 0.1);
    }

    #[test]
    fn a_full_speed_impulse_transfers_momentum_to_the_neighbour_it_hits() {
        let slots = 2;
        let mut physics = PlaygroundPhysics::new(slots, RADIUS).unwrap();
        let target_layout = Vec4::from_array(body_position(1, slots));
        physics.world.bodies[0].apply_impulse(flick(1.0, RIGHT));
        physics.step(12);

        let thrower = physics.world.bodies[0].velocity;
        let target = physics.world.bodies[1].velocity;
        assert!(
            target.x > 1.0,
            "the neighbour was left at {target}: the impulse passed through it"
        );
        assert!(
            target.x > thrower.x,
            "the thrower kept more speed ({thrower}) than the body it hit ({target})"
        );
        let moved = physics.pose(1, slots, Rotor4::IDENTITY).position - target_layout;
        assert!(moved.x > 0.0, "the neighbour never left its layout");
    }

    #[test]
    fn ndc_from_pixels_centres_the_viewport_and_flips_y() {
        let viewport = (800, 600);
        assert_eq!(
            ndc_from_pixels(Vec2::new(400.0, 300.0), viewport),
            Vec2::ZERO
        );
        assert_eq!(
            ndc_from_pixels(Vec2::ZERO, viewport),
            Vec2::new(-1.0, 1.0),
            "window top-left is NDC (-1, +1)"
        );
        assert_eq!(
            ndc_from_pixels(Vec2::new(800.0, 600.0), viewport),
            Vec2::new(1.0, -1.0)
        );
    }
}
