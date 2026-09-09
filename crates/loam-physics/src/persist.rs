use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::collider::ColliderKind;
use crate::integrator::PhysicsSpace;
use crate::state::WorldState;
use crate::world::World;

pub const PERSIST_SCHEMA: &str = "loam.physics.world";

pub const PERSIST_VERSION: u32 = 1;

pub const PERSIST_CODEC: &str = "ron";

#[derive(Serialize, Deserialize)]
struct Header {
    schema: String,
    version: u32,
    codec: String,
    registrations: Vec<(ColliderKind, ColliderKind)>,
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
    #[error("the saved narrowphase registrations are not this world's")]
    Registrations,
}

impl<S> World<S>
where
    S: PhysicsSpace,
    S::Point: Serialize + DeserializeOwned,
    S::Vector: Serialize + DeserializeOwned,
    S::Iso: Serialize + DeserializeOwned,
    S::AngVel: Serialize + DeserializeOwned,
    S::Inertia: Serialize + DeserializeOwned,
{
    pub fn save(&self) -> Result<String, PersistError> {
        let document = Persisted {
            header: Header {
                schema: PERSIST_SCHEMA.to_owned(),
                version: PERSIST_VERSION,
                codec: PERSIST_CODEC.to_owned(),
                registrations: self.narrowphase.registrations().to_vec(),
            },
            state: self.snapshot(),
        };
        ron::to_string(&document).map_err(PersistError::Encode)
    }

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
        self.restore(&state)
            .map_err(|_| PersistError::Registrations)
    }
}
