pub mod composite;
pub mod depth;
pub mod device;
pub mod gpu_timer;
pub mod hypergimbal;
pub mod hyperslice;
pub mod line;
pub mod pass;
pub mod point;
pub mod present;
pub mod raymarch;
pub mod shader;
pub mod sky_ground;
pub mod triangle;
pub mod viewport;

pub use depth::DepthBuffer;
pub use hyperslice::HyperslicePass;
pub use line::{LinePass, LineRasterNode, LineRasterUniforms};
pub use pass::{
    ColorLoad, FrameFormat, FramePass, FrameTarget, GpuTime, PassError, PassExecutionError,
    PassPhase, PassSchedule, PassStage, Section,
};
pub use point::{PointInstance, PointPass, PointRasterNode, PointRasterUniforms};
pub use present::Presenter;
pub use raymarch::{RayMarchNode, RayMarchUniforms, RaymarchPass};
pub use sky_ground::{SkyGroundNode, SkyGroundPass, SkyGroundUniforms};
pub use triangle::{
    FragmentShading, MaterialPipeline, MaterialSpec, TriangleFeed, TriangleRasterNode,
    TriangleRasterUniforms, TriangleVertex,
};
pub use viewport::Viewport;

/// `ReversedZ` follows the [`depth`] contract; `StandardZ` serves a standard perspective matrix.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DepthConvention {
    StandardZ,
    ReversedZ,
}

impl DepthConvention {
    pub fn compare(self, standard_z: wgpu::CompareFunction) -> wgpu::CompareFunction {
        match self {
            Self::StandardZ => standard_z,
            Self::ReversedZ => depth::DEPTH_COMPARE,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum DepthMode {
    Off,
    ReadWrite { format: wgpu::TextureFormat },
    ReadOnly { format: wgpu::TextureFormat },
}

impl DepthMode {
    pub fn format(&self) -> Option<wgpu::TextureFormat> {
        match self {
            DepthMode::Off => None,
            DepthMode::ReadWrite { format } | DepthMode::ReadOnly { format } => Some(*format),
        }
    }

    pub fn is_active(&self) -> bool {
        !matches!(self, DepthMode::Off)
    }

    pub fn writes(&self) -> bool {
        matches!(self, DepthMode::ReadWrite { .. })
    }
}
