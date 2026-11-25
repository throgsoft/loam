pub use loam_math::{
    Bivector, Bivector4, EuclideanR4, Plane4, Projection, RasterizableSpace, Rotor, Rotor4,
    SphericalS3Embedded, Vec2, Vec3, Vec4, WPlane,
};
pub use loam_runtime::{
    stores, ActionId, AppCommand, CameraPair, Command, Ctx, CursorPolicy, DepthEnvelope, Dispatch,
    DomainBuilder, DomainError, DomainHandle, DomainId, DomainRay, DomainSpace, Domains,
    EdgeShading, Entity, Eye, FreeCamera, Growth, ImageRay, Input, Instance, LogCapacity, Material,
    MaterialId, Moment, Number, Orbit, Order, Outcome, Overlay, PaletteId, Phase, PointRecord,
    Pointer, PointerButton, PointerPhase, Pose, PreparedGeometry, PreparedId, Projection4,
    Publication, Rejection, Rig, Section4, SectionCut, SegmentRecord, Session, Setting, SimConfig,
    SpawnBundle, Store, Stores, TextRecord, TypedDomain, Version, ViewId, ViewMapping, ViewSpec,
    Views, CAMERA_RIG, DOMAIN_STEP, MOVE_BACKWARD, MOVE_DOWN, MOVE_FORWARD, MOVE_LEFT, MOVE_RIGHT,
    MOVE_UP,
};
#[cfg(feature = "physics")]
pub use loam_runtime::{GrabConfig, GrabHold, Physics, PhysicsConfig, RestConfig, ThrowConfig};
pub use loam_shape::polytope::{
    polytope_section_faces_append, vertex_color_by_position, Polytope4, Polytope4Topology,
    SectionScratch,
};
pub use loam_shape::projected_edges::stereographic_view_point;
pub use loam_shape::{FieldShape, LineMesh, Shape, TriangleMesh};
pub use loam_time::par::Sequential;
pub use loam_time::{Ease, Track};

#[cfg(feature = "physics")]
pub use loam_physics::body::MASK_ALL;
#[cfg(feature = "physics")]
pub use loam_physics::euclidean_r4::{
    flush_halfspace4_body_r4, halfspace4_body_r4, polytope_body_r4, regular_polytope4_inertia,
    regular_polytope_body_r4,
};
#[cfg(feature = "physics")]
pub use loam_physics::manifold::PENETRATION_SLOP;
#[cfg(feature = "physics")]
pub use loam_physics::{BodyDef, EditError, RigidBody};

#[cfg(feature = "render")]
pub use loam_render::hypergimbal::{Hypergimbal, RingDrag, RingStyle};
#[cfg(feature = "render")]
pub use loam_render::hyperslice::{shape_id, BodyUniform};
#[cfg(feature = "render")]
pub use loam_render::{
    root_view_projection, FragmentShading, FramePass, HyperslicePass, RaymarchPass, Section,
    SkyGroundPass, TriangleFeed, Viewport,
};

#[cfg(feature = "text")]
pub use loam_text::ab_glyph::FontRef;
#[cfg(feature = "text")]
pub use loam_text::glyph::{
    append_field_prism, layout_word, DistanceField2D, GlyphParams, GlyphSolid,
};

#[cfg(feature = "app")]
pub use loam_app::{
    launch, launch_with, Args, Bindings, CaptureControl, CaptureFormat, CaptureRequest,
    CaptureStage, CaptureUnavailable, CommandSender, Environment, HostConfig, HostError, InputHook,
    Key, LaunchMode, PaletteMode, PublishedHook, Redraw, SessionApp, Submit, WasmConfig,
};
