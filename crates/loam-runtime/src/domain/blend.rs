use loam_math::Space;

use super::{DomainSpace, Pose};
use crate::entity::Entity;
use crate::store::{Store, Version};

pub(crate) fn presented_alpha(alpha: f32) -> f32 {
    if alpha.is_nan() {
        return 1.0;
    }
    alpha.clamp(0.0, 1.0)
}

struct TickPose<S: Space> {
    entity: Entity,
    pose: Pose<S>,
    before: Version,
    after: Version,
}

impl<S: Space> Clone for TickPose<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: Space> Copy for TickPose<S> {}

pub(super) struct Blend<S: Space> {
    before_tick: Vec<TickPose<S>>,
    moving: Vec<Entity>,
    generation: u64,
    presented: f32,
}

impl<S: DomainSpace> Blend<S> {
    pub(super) fn new() -> Self {
        Self {
            before_tick: Vec::new(),
            moving: Vec::new(),
            generation: 0,
            presented: 1.0,
        }
    }

    pub(super) fn presented(&self) -> f32 {
        self.presented
    }

    pub(super) fn present(&mut self, alpha: f32) {
        self.presented = alpha;
    }

    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) fn moving(&self) -> &[Entity] {
        &self.moving
    }

    pub(super) fn snap(&mut self, entity: Entity, poses: &Store<Pose<S>>) {
        let Some(dense) = poses.dense_index(entity) else {
            return;
        };
        let Some(pose) = poses.rows().get(dense).copied() else {
            return;
        };
        if let Some(entry) = self
            .before_tick
            .get_mut(dense)
            .filter(|entry| entry.entity == entity)
        {
            entry.pose = pose;
        }
    }

    #[inline]
    pub(super) fn drawn(
        &self,
        space: &S,
        poses: &Store<Pose<S>>,
        entity: Entity,
        alpha: f32,
    ) -> Option<Pose<S>> {
        let dense = poses.dense_index(entity)?;
        let pose = *poses.rows().get(dense)?;
        if alpha >= 1.0 || !poses.is_tracked() {
            return Some(pose);
        }
        Some(self.blended(space, poses, entity, dense, pose, alpha))
    }

    #[inline(never)]
    fn blended(
        &self,
        space: &S,
        poses: &Store<Pose<S>>,
        entity: Entity,
        dense: usize,
        pose: Pose<S>,
        alpha: f32,
    ) -> Pose<S> {
        let now = poses
            .dense_versions()
            .and_then(|(_, versions)| versions.get(dense));
        match self.before_tick.get(dense) {
            Some(entry)
                if entry.entity == entity
                    && now == Some(&entry.after)
                    && entry.after != entry.before =>
            {
                space.blend(&entry.pose, &pose, alpha)
            }
            _ => pose,
        }
    }

    pub(super) fn begin(&mut self, poses: &Store<Pose<S>>) {
        self.before_tick.clear();
        if !poses.is_tracked() {
            return;
        }
        let Some((_, versions)) = poses.dense_versions() else {
            return;
        };
        self.before_tick
            .extend(
                poses
                    .iter()
                    .zip(versions)
                    .map(|((entity, pose), version)| TickPose {
                        entity,
                        pose: *pose,
                        before: *version,
                        after: *version,
                    }),
            );
    }

    pub(super) fn end(&mut self, poses: &Store<Pose<S>>) {
        self.moving.clear();
        let Some((keys, versions)) = poses.dense_versions() else {
            return;
        };
        for ((entry, key), version) in self.before_tick.iter_mut().zip(keys).zip(versions) {
            if *version != entry.before && entry.entity.key() == *key {
                entry.after = *version;
                self.moving.push(entry.entity);
            }
        }
        self.generation = self.generation.wrapping_add(1);
    }

    // Runs before the pose store swaps its last row into the released one, so the blend entry follows that row.
    pub(super) fn release(&mut self, entity: Entity, poses: &Store<Pose<S>>) {
        let Some(dense) = poses.dense_index(entity) else {
            return;
        };
        let last = poses.len() - 1;
        let moved = self
            .before_tick
            .get(last)
            .copied()
            .filter(|entry| Some(entry.entity) == poses.entity_at(last));
        if let (Some(moved), Some(slot)) = (moved, self.before_tick.get_mut(dense)) {
            *slot = moved;
        }
    }

    pub(super) fn clear(&mut self) {
        self.before_tick.clear();
        self.moving.clear();
    }
}

#[cfg(test)]
mod tests {
    use loam_math::{EuclideanR4, Rotor4};
    use loam_time::par::Sequential;

    use crate::assets::{Material, PreparedGeometry};
    use crate::command::SpawnBundle;
    use crate::domain::{DomainBuilder, DomainError, DomainHandle, DomainSpace, Instance, Pose};
    use crate::entity::Entity;
    use crate::phase::{Ctx, Phase, Tick};
    use crate::publication::Publication;
    use crate::session::{Session, SimConfig};
    use crate::store::LogCapacity;
    use crate::view::{Section4, Vec4, ViewSpec};

    crate::stores! {
        #[derive(Default)]
        pub struct Probe {
            tags: Store<u8>,
        }
    }

    const SEGMENTS: usize = 3;
    const EYE_AT: Vec4 = Vec4::new(0.0, 0.0, 0.0, 2.0);
    const OBJECT_AT: Vec4 = Vec4::new(0.0, 0.0, -4.0, 0.0);

    fn lines() -> PreparedGeometry {
        PreparedGeometry::Lines4 {
            segments: (0..SEGMENTS)
                .map(|i| {
                    let x = 0.1 + i as f32 * 0.1;
                    [[x, 0.0, 0.0, 0.0], [-x, 0.0, 0.0, 0.0]]
                })
                .collect(),
        }
    }

    fn drawn_session(
        step: fn(Tick) -> Option<(f32, bool)>,
    ) -> (Session<Probe>, DomainHandle<EuclideanR4>, Entity) {
        let mut session = Session::new(Probe::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let prepared = session.prepare(lines());
        let material = session.add_material(Material::flat([1.0; 4]));
        let root = session.views().root_id();
        let mover = session.dispatch(|dispatch| {
            let eye = dispatch
                .spawn(SpawnBundle::new().at(r4, Pose::at(EYE_AT)))
                .unwrap();
            let mover = dispatch
                .spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(OBJECT_AT))
                        .instance(Instance::new(prepared, material)),
                )
                .unwrap();
            dispatch
                .domains
                .typed(r4)
                .unwrap()
                .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))
                .unwrap();
            mover
        });
        session.system(
            Phase::Simulation,
            "move",
            move |ctx: Ctx<'_, Probe>| -> Result<(), DomainError> {
                let Some((x, teleport)) = step(ctx.step.tick) else {
                    return Ok(());
                };
                let domain = ctx.domains.typed(r4)?;
                let pose = Pose::at(OBJECT_AT + Vec4::X * x);
                if teleport {
                    domain.teleport(mover, pose)
                } else {
                    domain.set_pose(mover, pose)
                }
            },
        );
        (session, r4, mover)
    }

    fn drawn_x(session: &mut Session<Probe>, publication: &mut Publication<()>, alpha: f32) -> f32 {
        session.publish(publication, alpha).unwrap();
        publication.views[0].records.instances.rows()[0]
            .pose
            .coordinates[0]
            - OBJECT_AT.x
    }

    #[test]
    fn a_body_is_drawn_between_its_last_two_ticks_and_a_stop_or_jump_is_not_smeared() {
        let (mut session, _, _) = drawn_session(|tick| Some((tick.0 as f32 + 1.0, false)));
        let mut publication = Publication::default();
        session.tick(&Sequential).unwrap();
        assert_eq!(drawn_x(&mut session, &mut publication, 0.25), 0.25);
        assert_eq!(drawn_x(&mut session, &mut publication, 0.75), 0.75);
        assert_eq!(drawn_x(&mut session, &mut publication, 1.0), 1.0);
        session.tick(&Sequential).unwrap();
        assert_eq!(drawn_x(&mut session, &mut publication, 0.5), 1.5);

        let (mut session, _, _) = drawn_session(|tick| (tick.0 == 0).then_some((1.0, false)));
        let mut publication = Publication::default();
        session.tick(&Sequential).unwrap();
        assert_eq!(drawn_x(&mut session, &mut publication, 0.5), 0.5);
        session.tick(&Sequential).unwrap();
        assert_eq!(drawn_x(&mut session, &mut publication, 0.5), 1.0);

        let (mut session, r4, mover) = drawn_session(|_| Some((10.0, true)));
        let mut publication = Publication::default();
        session.tick(&Sequential).unwrap();
        assert_eq!(drawn_x(&mut session, &mut publication, 0.25), 10.0);
        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .set_pose(mover, Pose::at(OBJECT_AT + Vec4::X * 5.0))
            .unwrap();
        assert_eq!(drawn_x(&mut session, &mut publication, 0.5), 5.0);
    }

    #[test]
    fn drawn_pose_and_image_of_read_the_pose_the_publication_drew() {
        let (mut session, r4, mover) = drawn_session(|tick| Some((tick.0 as f32 + 1.0, false)));
        let mut publication = Publication::default();
        session.tick(&Sequential).unwrap();
        session.publish(&mut publication, 0.3).unwrap();
        let published = &publication.views[0];
        let row = &published.records.instances.rows()[0];
        let drawn = session.domains().read(r4).unwrap().drawn_pose(mover, 0.3);
        assert_eq!(
            drawn.map(|pose| EuclideanR4.chart_pose(&pose)),
            Some(row.pose)
        );
        let domain = session.domains().get(r4.id()).unwrap();
        assert_eq!(
            domain.image_of(published.target.view, mover),
            Some(row.image_point)
        );
    }

    #[test]
    fn a_body_swapped_into_a_despawned_row_keeps_its_blend() {
        let (mut session, r4, doomed) = drawn_session(|_| None);
        let prepared = session.prepare(lines());
        let material = session.add_material(Material::flat([1.0; 4]));
        let swapped = session
            .dispatch(|dispatch| {
                dispatch.spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(OBJECT_AT))
                        .instance(Instance::new(prepared, material)),
                )
            })
            .unwrap();
        session.system(
            Phase::Simulation,
            "rise",
            move |ctx: Ctx<'_, Probe>| -> Result<(), DomainError> {
                let y = ctx.step.tick.0 as f32 + 1.0;
                ctx.domains
                    .typed(r4)?
                    .set_pose(swapped, Pose::at(OBJECT_AT + Vec4::Y * y))
            },
        );
        let mut publication = Publication::default();
        let drawn_y = |session: &mut Session<Probe>, publication: &mut Publication<()>, alpha| {
            session.publish(publication, alpha).unwrap();
            let rows = publication.views[0].records.instances.rows();
            let record = rows.iter().find(|record| record.entity == swapped).unwrap();
            record.pose.coordinates[1] - OBJECT_AT.y
        };
        session.tick(&Sequential).unwrap();
        assert_eq!(drawn_y(&mut session, &mut publication, 0.25), 0.25);
        session
            .dispatch(|dispatch| dispatch.despawn(doomed))
            .unwrap();
        assert_eq!(drawn_y(&mut session, &mut publication, 0.75), 0.75);
    }

    #[test]
    fn a_rotor_sign_flip_between_ticks_is_drawn_without_a_turn() {
        let (mut session, r4, mover) = drawn_session(|_| None);
        let turned = Rotor4::from_rotation_arc(Vec4::X, Vec4::new(1.0, 1.0, 0.0, 1.0).normalize());
        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .set_frame(mover, turned)
            .unwrap();
        session.system(
            Phase::Simulation,
            "flip",
            move |ctx: Ctx<'_, Probe>| -> Result<(), DomainError> {
                let domain = ctx.domains.typed(r4)?;
                let frame = domain
                    .poses()
                    .get(mover)
                    .ok_or(DomainError::Stale(mover))?
                    .frame;
                let minus = Rotor4 {
                    s: -1.0,
                    ..Rotor4::IDENTITY
                };
                domain.set_frame(mover, frame * minus)
            },
        );
        let mut publication = Publication::default();
        session.tick(&Sequential).unwrap();
        session.publish(&mut publication, 0.5).unwrap();
        let drawn = publication.views[0].records.instances.rows()[0].pose.frame;
        let expected = EuclideanR4.chart_pose(&Pose {
            point: OBJECT_AT,
            frame: turned,
        });
        for (column, expected) in drawn.iter().zip(expected.frame) {
            for (value, expected) in column.iter().zip(expected) {
                assert!((value - expected).abs() < 1e-5, "{drawn:?}");
            }
        }
    }
}
