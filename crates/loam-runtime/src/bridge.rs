use loam_shape::field::FieldKind;

use crate::domain::{DomainError, DomainId};
use crate::entity::Entity;
use crate::store::StoreError;
use crate::view::{ImageRay, ImageSpaceId, Placement, Vec3, ViewId};

/// Linked from its anchor to the source view's eye; releasing either entity unlinks it and unplaces `image` inside that despawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bridge {
    pub source: DomainId,
    pub view: ViewId,
    pub image: ImageSpaceId,
}

pub struct BridgeSpec {
    pub anchor: Entity,
    pub into: ImageSpaceId,
    pub source: DomainId,
    pub view: ViewId,
    pub placement: Placement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BridgeError {
    Stale(Entity),
    UnknownDomain(DomainId),
    UnknownView(ViewId),
    UnknownImage(ImageSpaceId),
    Nonlinear(&'static str),
    NoRayLift(&'static str),
    NoStepBound(&'static str, FieldKind),
    Domain(DomainError),
    Link(StoreError),
}

impl From<DomainError> for BridgeError {
    fn from(error: DomainError) -> Self {
        Self::Domain(error)
    }
}

impl From<StoreError> for BridgeError {
    fn from(error: StoreError) -> Self {
        Self::Link(error)
    }
}

pub(crate) fn step_bound(kind: FieldKind) -> bool {
    matches!(
        kind,
        FieldKind::ExactDistance | FieldKind::ConservativeBound
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragError {
    NoPick,
    NotGrabbed,
    NoLift(&'static str),
    /// The pointer ray runs parallel to the drag plane, so no point on it is nearer than another.
    Ambiguous(&'static str),
    Domain(DomainError),
}

impl From<DomainError> for DragError {
    fn from(error: DomainError) -> Self {
        Self::Domain(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drag {
    pub entity: Entity,
    pub domain: DomainId,
    pub view: ViewId,
    pub image: ImageSpaceId,
    pub(crate) plane: [f32; 3],
    pub(crate) normal: [f32; 3],
    pub(crate) center: [f32; 3],
    pub(crate) at: [f32; 3],
    pub(crate) time: f64,
    pub(crate) velocity: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DragRelease {
    pub entity: Entity,
    pub domain: DomainId,
    pub image: ImageSpaceId,
    /// In the view's image space, per second.
    pub velocity: [f32; 3],
}

const PLANE_EPSILON: f32 = 1e-6;

impl Drag {
    pub(crate) fn meet(&self, ray: &ImageRay) -> Option<[f32; 3]> {
        let normal = Vec3::from(self.normal);
        let along = Vec3::from(ray.direction).dot(normal);
        if along.abs() <= PLANE_EPSILON {
            return None;
        }
        Some(ray.at((Vec3::from(self.plane) - Vec3::from(ray.origin)).dot(normal) / along))
    }

    pub(crate) fn moved(&self, at: [f32; 3]) -> [f32; 3] {
        (Vec3::from(self.center) + Vec3::from(at) - Vec3::from(self.plane)).to_array()
    }

    pub(crate) fn sample(&mut self, at: [f32; 3], time: f64) {
        let elapsed = time - self.time;
        if elapsed > 0.0 {
            self.velocity = ((Vec3::from(at) - Vec3::from(self.at)) / elapsed as f32).to_array();
            self.time = time;
        }
        self.at = at;
    }

    pub(crate) fn released(&self) -> DragRelease {
        DragRelease {
            entity: self.entity,
            domain: self.domain,
            image: self.image,
            velocity: self.velocity,
        }
    }
}

#[cfg(test)]
mod tests {
    use loam_math::{EuclideanR4, Iso3, Iso4Flat};

    use super::*;
    use crate::command::SpawnBundle;
    use crate::domain::{DomainBuilder, Instance, Pose};
    use crate::session::{Material, PreparedGeometry, Session, SimConfig};
    use crate::store::tests::alloc_probe::bytes_allocated_by;
    use crate::store::LogCapacity;
    use crate::view::{Rigid, Section4, Vec4, ViewSpec};

    crate::stores! {
        #[derive(Default)]
        pub struct Probe {
            tags: Store<u8>,
        }
    }

    #[test]
    fn a_warmed_pick_through_two_bridges_allocates() {
        let mut session = Session::new(Probe::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let stub = session.prepare(PreparedGeometry::Lines4 {
            segments: vec![[[0.5, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0]]],
        });
        let material = session.add_material(Material::flat([1.0; 4]));
        let root = session.views().root();
        let (near, far, anchor) = session.dispatch(|d| {
            let eye = d
                .spawn(SpawnBundle::new().at(r4, Pose(Iso4Flat::IDENTITY)))
                .unwrap();
            d.spawn(
                SpawnBundle::new()
                    .at(
                        r4,
                        Pose(Iso4Flat::from_translation(Vec4::new(1.0, 0.0, -4.0, 0.0))),
                    )
                    .instance(Instance::new(stub, material)),
            )
            .unwrap();
            let anchor = d.spawn(SpawnBundle::new()).unwrap();
            let domain = d.domains.typed(r4).unwrap();
            let near = domain.add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }));
            let far = domain.add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }));
            (near, far, anchor)
        });
        let placement = Placement::Rigid(Rigid {
            pose: Iso3::from_translation(Vec3::new(0.0, 0.0, -0.5)),
            scale: 0.25,
        });
        let outer = session
            .bridge(BridgeSpec {
                anchor,
                into: root,
                source: r4.id(),
                view: near,
                placement,
            })
            .unwrap();
        let middle = session.bridges().get(outer).unwrap().data.image;
        session
            .bridge(BridgeSpec {
                anchor,
                into: middle,
                source: r4.id(),
                view: far,
                placement,
            })
            .unwrap();
        for _ in 0..8 {
            assert!(session.pick([0.4330127, 0.0]).is_some());
        }

        let bytes = bytes_allocated_by(|| {
            for _ in 0..16 {
                assert!(session.pick([0.4330127, 0.0]).is_some());
            }
        });
        assert_eq!(
            bytes, 0,
            "16 warmed picks through two bridges asked the allocator for {bytes} bytes"
        );
    }
}
