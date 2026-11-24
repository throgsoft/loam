use loam_shape::polytope::Polytope4Topology;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PreparedId(u32);

impl PreparedId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialId(u32);

impl MaterialId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PaletteId(u32);

impl PaletteId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PreparedGeometry {
    Lines4 {
        segments: Vec<[[f32; 4]; 2]>,
    },
    Lines3 {
        segments: Vec<[[f32; 3]; 2]>,
    },
    /// Edges for the wireframe and cells for the section, at canonical coordinates times `scale`.
    Polytope4 {
        polytope: loam_shape::polytope::Polytope4,
        scale: f32,
    },
    /// A solid that only a view drawing fields shows, with no edges or section.
    Field {
        shape: loam_shape::FieldShape,
        size: f32,
    },
}

impl PreparedGeometry {
    pub fn edges_of(topology: &Polytope4Topology, scale: f32) -> Self {
        let segments = topology
            .edges
            .iter()
            .map(|&[i, j]| {
                [
                    (topology.vertices[i as usize] * scale).to_array(),
                    (topology.vertices[j as usize] * scale).to_array(),
                ]
            })
            .collect();
        Self::Lines4 { segments }
    }

    /// Chart distance from the origin to the farthest vertex.
    pub fn bounding_radius(&self) -> f32 {
        fn farthest<'a>(points: impl Iterator<Item = &'a [f32]>) -> f32 {
            points
                .map(|point| point.iter().map(|c| c * c).sum::<f32>().sqrt())
                .fold(0.0, f32::max)
        }
        match self {
            Self::Lines4 { segments } => {
                farthest(segments.iter().flatten().map(|point| point.as_slice()))
            }
            Self::Lines3 { segments } => {
                farthest(segments.iter().flatten().map(|point| point.as_slice()))
            }
            Self::Polytope4 { polytope, scale } => {
                loam_shape::FieldShape::Polytope(*polytope).radius() * scale
            }
            Self::Field { shape, size } => shape.radius() * size,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Material {
    Lines { color: [f32; 4], width_px: f32 },
    Flat { color: [f32; 4] },
}

impl Material {
    pub fn lines(color: [f32; 4], width_px: f32) -> Self {
        Self::Lines { color, width_px }
    }

    pub fn flat(color: [f32; 4]) -> Self {
        Self::Flat { color }
    }
}

#[derive(Clone, Copy)]
pub struct Library<'a> {
    pub(crate) geometry: &'a [PreparedGeometry],
    pub(crate) materials: &'a [Material],
    pub(crate) palettes: &'a [Vec<[f32; 4]>],
}

impl Library<'_> {
    pub(crate) fn line_style(&self, material: MaterialId) -> ([f32; 4], f32) {
        match self.materials.get(material.index()) {
            Some(Material::Lines { color, width_px }) => (*color, *width_px),
            Some(Material::Flat { color }) => (*color, 1.0),
            None => ([1.0; 4], 1.0),
        }
    }

    pub(crate) fn palette(&self, palette: PaletteId) -> &[[f32; 4]] {
        self.palettes
            .get(palette.index())
            .map_or(&[][..], Vec::as_slice)
    }
}

#[derive(Default)]
pub(crate) struct Assets {
    prepared: Vec<PreparedGeometry>,
    materials: Vec<Material>,
    palettes: Vec<Vec<[f32; 4]>>,
}

impl Assets {
    pub(crate) fn prepare(&mut self, geometry: PreparedGeometry) -> PreparedId {
        self.prepared.push(geometry);
        PreparedId((self.prepared.len() - 1) as u32)
    }

    pub(crate) fn prepared(&self, id: PreparedId) -> Option<&PreparedGeometry> {
        self.prepared.get(id.index())
    }

    pub(crate) fn add_palette(&mut self, colors: Vec<[f32; 4]>) -> PaletteId {
        self.palettes.push(colors);
        PaletteId((self.palettes.len() - 1) as u32)
    }

    pub(crate) fn add_material(&mut self, material: Material) -> MaterialId {
        self.materials.push(material);
        MaterialId((self.materials.len() - 1) as u32)
    }

    pub(crate) fn geometry(&self) -> &[PreparedGeometry] {
        &self.prepared
    }

    pub(crate) fn library(&self) -> Library<'_> {
        Library {
            geometry: &self.prepared,
            materials: &self.materials,
            palettes: &self.palettes,
        }
    }
}
