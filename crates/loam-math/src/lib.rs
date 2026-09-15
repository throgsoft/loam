//! Geometry uses f32; transcendental results can differ across targets.

pub mod bivector;
pub mod blended;
pub mod euclidean;
pub mod euclidean_r2;
pub mod euclidean_r4;
pub mod hyperbolic;
pub mod quotient;
pub mod rasterizable;
pub mod sectionable;
pub mod space;
pub mod spherical;
pub mod spherical_embedded;

pub use glam::Mat3;

pub use bivector::{
    Bivector, Bivector2, Bivector3, Bivector4, Plane4, Rotor, Rotor2, Rotor3, Rotor4,
};
pub use blended::{
    BlendedSpace, BlendingField, ConformallyFlat, LinearBlendX, BLENDED_E3_H3_WGSL_RESIDUAL,
};
pub use euclidean::{EuclideanR3, Iso3};
pub use euclidean_r2::{EuclideanR2, Iso2};
pub use euclidean_r4::{EuclideanR4, Iso4Flat};
pub use hyperbolic::{HyperbolicH3, Iso3H, H3_MAX_ARC, POINCARE_R2_MAX};
pub use quotient::{CoveringSpace, FlatTorus3, LensSpace, QuotientSpace};
pub use rasterizable::{Projection, RasterizableSpace, STEREOGRAPHIC_POLE_EPSILON};
pub use sectionable::{WPlane, EDGE_PARALLEL_EPSILON, SLICE_PERTURBATION_EPSILON};
pub use space::{IsometryGroup, Space, WgslAccuracy, WgslSpace, FLAT_CHART_MAX_ARC};
pub use spherical::{Iso4, SphericalS3};
pub use spherical_embedded::SphericalS3Embedded;
