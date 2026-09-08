use loam_math::Projection;
use loam_render::raymarch::RaymarchShape;
use loam_shape::polytope::Polytope4;

use crate::catalog::ShapeEntry;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub(crate) enum SurfaceMode {
    #[default]
    Raster,
    Sdf,
    Off,
}

impl SurfaceMode {
    pub(crate) fn from_token(token: &str) -> Option<Self> {
        match token {
            "raster" => Some(SurfaceMode::Raster),
            "sdf" => Some(SurfaceMode::Sdf),
            "off" => Some(SurfaceMode::Off),
            _ => None,
        }
    }

    pub(crate) fn uses_sdf_for_polychora(self) -> bool {
        matches!(self, SurfaceMode::Sdf)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct SectionLayer {
    pub(crate) perimeter: bool,
    /// `0.0` skips the fill; `(0, 1)` is translucent; `1.0` writes depth.
    pub(crate) surface_alpha: f32,
}

impl SectionLayer {
    pub(crate) fn fill_visible(self) -> bool {
        self.surface_alpha > 0.0
    }
}

impl SectionLayer {
    pub(crate) const CROSS_SECTION_DEFAULT: SectionLayer = SectionLayer {
        perimeter: true,
        surface_alpha: 1.0,
    };
    pub(crate) const PROJECTED_CAP_DEFAULT: SectionLayer = SectionLayer {
        perimeter: false,
        surface_alpha: 0.0,
    };
}

// The physical slice uses drop-w even when the wireframe projection changes.
pub(crate) fn section_layer_projection(
    is_cross_section: bool,
    projection: Projection<4>,
) -> Projection<4> {
    if is_cross_section {
        Projection::Identity
    } else {
        projection
    }
}

// The extended kernels remain disabled until browser rendering is verified.
pub(crate) fn row_blocks_sdf(row: &[ShapeEntry]) -> bool {
    row.iter().any(|e| {
        matches!(
            e.shape,
            RaymarchShape::Polytope(Polytope4::Cell120 | Polytope4::Cell600)
        )
    })
}
