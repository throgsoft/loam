pub mod composite;
pub mod depth;
pub mod device;
pub mod gizmo;
pub mod gpu_timer;
pub mod hypergimbal;
pub mod lattice;
pub mod line_raster;
pub mod line_raster_static_r4;
pub mod point_raster;
pub mod present;
pub mod raymarch;
pub mod shader;
pub mod sky_ground;
pub mod triangle_raster;
pub mod view;

pub use depth::DepthBuffer;
pub use lattice::Viewport;
pub use line_raster::{LineRasterNode, LineRasterUniforms};
pub use line_raster_static_r4::{LineRasterStaticR4Node, LineRasterStaticR4Uniforms};
pub use point_raster::{PointRasterNode, PointRasterUniforms};
pub use present::Presenter;
pub use raymarch::{RayMarchNode, RayMarchUniforms};
pub use sky_ground::{Ground, SkyGroundNode, SkyGroundUniforms};
pub use triangle_raster::{
    FragmentShading, TriangleRasterNode, TriangleRasterUniforms, TriangleVertex,
};

/// `ReversedZ` follows the [`view`] depth contract; `StandardZ` serves a standard perspective matrix.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DepthConvention {
    StandardZ,
    ReversedZ,
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

#[cfg(test)]
pub(crate) fn depth_passes(compare: wgpu::CompareFunction, incoming: f32, stored: f32) -> bool {
    use wgpu::CompareFunction;
    match compare {
        CompareFunction::Never => false,
        CompareFunction::Less => incoming < stored,
        CompareFunction::Equal => incoming == stored,
        CompareFunction::LessEqual => incoming <= stored,
        CompareFunction::Greater => incoming > stored,
        CompareFunction::NotEqual => incoming != stored,
        CompareFunction::GreaterEqual => incoming >= stored,
        CompareFunction::Always => true,
    }
}
