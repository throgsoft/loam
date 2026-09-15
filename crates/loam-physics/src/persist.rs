use std::ops::Sub;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::collider::ColliderKind;
use crate::collision::VectorOps;
use crate::edit::EditError;
use crate::integrator::PhysicsSpace;
use crate::state::WorldState;
use crate::world::World;

pub const PERSIST_SCHEMA: &str = "loam.physics.world";

pub const PERSIST_VERSION: u32 = 3;

pub const PERSIST_CODEC: &str = "ron";

#[derive(Serialize, Deserialize)]
struct Header {
    schema: String,
    version: u32,
    codec: String,
    registrations: Vec<(ColliderKind, ColliderKind)>,
    field_registrations: Vec<ColliderKind>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "S::Point: Serialize, S::Vector: Serialize, S::Iso: Serialize, S::AngVel: Serialize, S::Inertia: Serialize",
    deserialize = "S::Point: Deserialize<'de>, S::Vector: Deserialize<'de>, S::Iso: Deserialize<'de>, S::AngVel: Deserialize<'de>, S::Inertia: Deserialize<'de>"
))]
struct Persisted<S: PhysicsSpace> {
    header: Header,
    state: WorldState<S>,
}

#[derive(Debug, Error)]
pub enum PersistError {
    #[error("the text is not a loam physics document: {0}")]
    Decode(ron::error::SpannedError),
    #[error("the world could not be encoded: {0}")]
    Encode(ron::Error),
    #[error("the schema is {found}, not {PERSIST_SCHEMA}")]
    Schema { found: String },
    #[error("the schema version is {found}, not {PERSIST_VERSION}")]
    Version { found: u32 },
    #[error("the codec is {found}, not {PERSIST_CODEC}")]
    Codec { found: String },
    #[error("the saved world state is invalid: {0}")]
    State(#[source] EditError),
}

impl<S> World<S>
where
    S: PhysicsSpace,
    S::Point: Serialize + DeserializeOwned + Copy + Sub<Output = S::Vector>,
    S::Vector: Serialize + DeserializeOwned + VectorOps,
    S::Iso: Serialize + DeserializeOwned,
    S::AngVel: Serialize + DeserializeOwned + PartialEq,
    S::Inertia: Serialize + DeserializeOwned,
{
    pub fn save(&self) -> Result<String, PersistError> {
        let document = Persisted {
            header: Header {
                schema: PERSIST_SCHEMA.to_owned(),
                version: PERSIST_VERSION,
                codec: PERSIST_CODEC.to_owned(),
                registrations: self.narrowphase.registrations().to_vec(),
                field_registrations: self.field_narrowphase.registrations().to_vec(),
            },
            state: self.snapshot(),
        };
        ron::to_string(&document).map_err(PersistError::Encode)
    }

    /// Decodes and checks the whole document before touching the world; a refusal leaves it unchanged.
    pub fn load(&mut self, text: &str) -> Result<(), PersistError> {
        let document: Persisted<S> = ron::from_str(text).map_err(PersistError::Decode)?;
        let header = document.header;
        if header.schema != PERSIST_SCHEMA {
            return Err(PersistError::Schema {
                found: header.schema,
            });
        }
        if header.version != PERSIST_VERSION {
            return Err(PersistError::Version {
                found: header.version,
            });
        }
        if header.codec != PERSIST_CODEC {
            return Err(PersistError::Codec {
                found: header.codec,
            });
        }
        let mut state = document.state;
        state.registrations = header.registrations;
        state.field_registrations = header.field_registrations;
        self.restore(&state).map_err(PersistError::State)
    }
}

#[cfg(all(test, feature = "r3"))]
mod tests {
    use glam::Vec3;
    use loam_math::EuclideanR3;

    use super::*;
    use crate::determinism_fixture::sample_body_r3;
    use crate::euclidean_r3::{halfspace_body_r3, register_default_narrowphase, sphere_body_r3};

    #[test]
    fn a_saved_pose_changed_without_its_contact_cache_is_refused_atomically() {
        let mut world = World::new(EuclideanR3);
        register_default_narrowphase(&mut world.narrowphase);
        world.set_gravity(Some(Vec3::NEG_Y * 9.8)).unwrap();
        world.push_body(halfspace_body_r3(Vec3::Y, 0.0).unwrap());
        let ball = world
            .push_body(sphere_body_r3(Vec3::new(0.0, 0.5, 0.0), Vec3::ZERO, 0.5, 1.0).unwrap());
        for _ in 0..120 {
            world.step(1.0 / 240.0).unwrap();
        }
        assert!(world.manifolds().len() > 0);
        let saved = world.save().unwrap();
        let mut document: Persisted<EuclideanR3> = ron::from_str(&saved).unwrap();
        document.state.bodies[ball].position = Vec3::X * 10.0;
        let altered = ron::to_string(&document).unwrap();
        let before = world.state_hash(sample_body_r3);
        let time = world.time();
        let manifolds = world.manifolds().len();

        assert!(matches!(
            world.load(&altered),
            Err(PersistError::State(EditError::InvalidManifold))
        ));
        assert!(world.body(ball).is_some());
        assert_eq!(world.state_hash(sample_body_r3), before);
        assert_eq!(world.time(), time);
        assert_eq!(world.manifolds().len(), manifolds);
    }
}
