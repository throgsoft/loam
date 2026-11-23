use crate::session::{MaterialId, PaletteId, PreparedId};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    pub geometry: PreparedId,
    pub material: MaterialId,
    pub shading: EdgeShading,
    pub line_width_px: Option<f32>,
    pub line_opacity: Option<f32>,
    /// The perimeter's line material; `None` publishes no section for a geometry that could be cut.
    pub section: Option<MaterialId>,
}

impl Instance {
    pub fn new(geometry: PreparedId, material: MaterialId) -> Self {
        Self {
            geometry,
            material,
            shading: EdgeShading::Material,
            line_width_px: None,
            line_opacity: None,
            section: None,
        }
    }

    pub fn shaded(mut self, shading: EdgeShading) -> Self {
        self.shading = shading;
        self
    }

    pub fn line_style(mut self, width_px: f32, opacity: f32) -> Self {
        self.line_width_px = Some(width_px);
        self.line_opacity = Some(opacity);
        self
    }

    pub fn sectioned(mut self, perimeter: MaterialId) -> Self {
        self.section = Some(perimeter);
        self
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum EdgeShading {
    #[default]
    Material,
    Palette(PaletteId),
    Depth {
        back: [f32; 4],
        front: [f32; 4],
        extent: f32,
    },
}
