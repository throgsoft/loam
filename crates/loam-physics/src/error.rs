use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditError {
    StaleHandle,
    NotFinite,
    InvalidGravity,
    InvalidTimeStep,
    InvalidMass,
    InvalidInertia,
    InvalidRestitution,
    UnsupportedCollider,
    MalformedCollider,
    OffCentreSphere,
    FieldAnchorFilter,
    DynamicHalfSpace,
    DynamicFieldAnchor,
    FieldAnchorRemoval,
    AnchorBindsOwnField,
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::StaleHandle => "the body handle is stale",
            Self::NotFinite => "the value is not finite",
            Self::InvalidGravity => "gravity is not finite",
            Self::InvalidTimeStep => "the time step is negative or not finite",
            Self::InvalidMass => "the mass is negative, not finite, or too small to invert",
            Self::InvalidInertia => "the inertia is negative, not finite, or too small to invert",
            Self::InvalidRestitution => "the restitution is negative or not finite",
            Self::UnsupportedCollider => "the space cannot use this collider",
            Self::MalformedCollider => "the collider shape is malformed",
            Self::OffCentreSphere => "a sphere collider needs a zero centre",
            Self::FieldAnchorFilter => "a field anchor body keeps an empty collision filter",
            Self::DynamicHalfSpace => "a half-space collider needs zero mass",
            Self::DynamicFieldAnchor => "a field anchor body needs zero mass",
            Self::FieldAnchorRemoval => "a field anchor body cannot be removed",
            Self::AnchorBindsOwnField => "a field's anchor cannot bind that field to itself",
        };
        f.write_str(text)
    }
}

impl std::error::Error for EditError {}
