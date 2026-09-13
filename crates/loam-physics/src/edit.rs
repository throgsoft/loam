use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditError {
    StaleHandle,
    NotFinite,
    InvalidGravity,
    InvalidTimeStep,
    InvalidTime,
    InvalidSolverTolerance,
    InvalidGrabConfig,
    InvalidStepTravel,
    InvalidMass,
    InvalidInertia,
    InvalidRestitution,
    InvalidBodyArena,
    InvalidBody,
    InvalidGeometry,
    InvalidFieldBinding,
    InvalidManifold,
    UnsupportedCollider,
    DynamicHalfSpace,
    DynamicFieldAnchor,
    FieldAnchorRemoval,
    AnchorBindsOwnField,
    FieldMismatch,
    /// The world's narrowphase registrations differ from the snapshot's.
    RegistrationMismatch,
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::StaleHandle => "the body handle is stale",
            Self::NotFinite => "the value is not finite",
            Self::InvalidGravity => "gravity is not finite",
            Self::InvalidTimeStep => "the time step is negative or not finite",
            Self::InvalidTime => "the world time is negative or not finite",
            Self::InvalidSolverTolerance => "the solver tolerance is negative or NaN",
            Self::InvalidGrabConfig => "the grab configuration is invalid",
            Self::InvalidStepTravel => "the adaptive step travel is not positive",
            Self::InvalidMass => "the mass is negative, not finite, or too small to invert",
            Self::InvalidInertia => "the inertia is negative, not finite, or too small to invert",
            Self::InvalidRestitution => "the restitution is negative or not finite",
            Self::InvalidBodyArena => "the body arena is inconsistent",
            Self::InvalidBody => "a body has invalid state",
            Self::InvalidGeometry => "the geometry store is inconsistent",
            Self::InvalidFieldBinding => "a field binding is invalid",
            Self::InvalidManifold => "a contact manifold is invalid",
            Self::UnsupportedCollider => "the space cannot use this collider",
            Self::DynamicHalfSpace => "a half-space collider needs zero mass",
            Self::DynamicFieldAnchor => "a field anchor body needs zero mass",
            Self::FieldAnchorRemoval => "a field anchor body cannot be removed",
            Self::AnchorBindsOwnField => "a field's anchor cannot bind that field to itself",
            Self::FieldMismatch => "the world's field anchors are not the snapshot's",
            Self::RegistrationMismatch => "the narrowphase registrations are not the snapshot's",
        };
        f.write_str(text)
    }
}

impl std::error::Error for EditError {}
