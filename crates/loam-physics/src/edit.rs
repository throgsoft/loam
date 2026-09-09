use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditError {
    StaleHandle,
    NotFinite,
    InvalidMass,
    InvalidInertia,
    UnsupportedCollider,
    RegistrationMismatch,
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::StaleHandle => "the body handle is stale or belongs to another world",
            Self::NotFinite => "the value is not finite",
            Self::InvalidMass => "the mass is negative, not finite, or too small to invert",
            Self::InvalidInertia => "the inertia is negative, not finite, or too small to invert",
            Self::UnsupportedCollider => "the space cannot use this collider",
            Self::RegistrationMismatch => "the narrowphase registrations are not the snapshot's",
        };
        f.write_str(text)
    }
}

impl std::error::Error for EditError {}
