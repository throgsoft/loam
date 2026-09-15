use loam::runtime::{EdgeShading, PaletteId};
use loam::shape::polytope::{vertex_color_by_position, Polytope4Topology};

// Matches the depth cue in the LineRasterStaticR4 shader.
pub(crate) const W_DEPTH_BACK: [f32; 4] = [0.30, 0.42, 0.58, 1.0];

pub(crate) const W_DEPTH_FRONT: [f32; 4] = [1.00, 0.78, 0.45, 1.0];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ColorMode {
    #[default]
    VertexGradient,
    UniqueEdge,
    WDepth,
}

impl ColorMode {
    pub(crate) const ALL: [ColorMode; 3] = [
        ColorMode::VertexGradient,
        ColorMode::UniqueEdge,
        ColorMode::WDepth,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            ColorMode::VertexGradient => "vertex",
            ColorMode::UniqueEdge => "edge",
            ColorMode::WDepth => "w-depth",
        }
    }

    pub(crate) fn from_token(token: &str) -> Option<Self> {
        ColorMode::ALL.into_iter().find(|mode| mode.name() == token)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Shades {
    pub(crate) gradient: PaletteId,
    pub(crate) unique: PaletteId,
    pub(crate) extent: f32,
}

impl Shades {
    pub(crate) fn of(self, mode: ColorMode) -> EdgeShading {
        match mode {
            ColorMode::VertexGradient => EdgeShading::Palette(self.gradient),
            ColorMode::UniqueEdge => EdgeShading::Palette(self.unique),
            ColorMode::WDepth => EdgeShading::Depth {
                back: W_DEPTH_BACK,
                front: W_DEPTH_FRONT,
                extent: self.extent,
            },
        }
    }
}

fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> [f32; 3] {
    let sixth = hue.fract() * 6.0;
    let chroma = value * saturation;
    let second = chroma * (1.0 - (sixth % 2.0 - 1.0).abs());
    let base = value - chroma;
    let (r, g, b) = match sixth.floor() as i32 % 6 {
        0 => (chroma, second, 0.0),
        1 => (second, chroma, 0.0),
        2 => (0.0, chroma, second),
        3 => (0.0, second, chroma),
        4 => (second, 0.0, chroma),
        _ => (chroma, 0.0, second),
    };
    [r + base, g + base, b + base]
}

fn palette_color(index: usize) -> [f32; 4] {
    const PHI_INV: f32 = 0.618_034;
    let hue = ((index as f32) * PHI_INV).fract();
    let saturation = 0.78 + 0.18 * ((index % 3) as f32 / 2.0);
    let value = 0.92 - 0.18 * (((index / 3) % 2) as f32);
    let [r, g, b] = hsv_to_rgb(hue, saturation, value);
    [r, g, b, 1.0]
}

pub(crate) fn unique_edge_colors(edges: &[[u32; 2]]) -> Vec<[f32; 4]> {
    let mut chosen = Vec::with_capacity(edges.len());
    let mut used = vec![usize::MAX; edges.len()];
    for (index, &[a, b]) in edges.iter().enumerate() {
        for (other, &[c, d]) in edges[..index].iter().enumerate() {
            if a == c || a == d || b == c || b == d {
                used[chosen[other]] = index;
            }
        }
        let color = used
            .iter()
            .position(|generation| *generation != index)
            .unwrap_or(index);
        chosen.push(color);
    }
    chosen
        .into_iter()
        .flat_map(|index| [palette_color(index); 2])
        .collect()
}

pub(crate) fn vertex_gradient_colors(topology: &Polytope4Topology) -> Vec<[f32; 4]> {
    topology
        .edges
        .iter()
        .flat_map(|&[a, b]| {
            [
                vertex_color_by_position(topology.vertices[a as usize]),
                vertex_color_by_position(topology.vertices[b as usize]),
            ]
        })
        .collect()
}

pub(crate) fn w_extent(topology: &Polytope4Topology, scale: f32) -> f32 {
    topology
        .vertices
        .iter()
        .map(|vertex| vertex.w.abs())
        .fold(0.0_f32, f32::max)
        .max(1e-6)
        * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_greedy_palette_separates_edges_that_share_a_vertex() {
        let edges: &[[u32; 2]] = &[[0, 1], [0, 2], [0, 3]];
        let colors = unique_edge_colors(edges);
        assert_eq!(colors.len(), 6, "one color per endpoint");
        for edge in 0..3 {
            assert_eq!(
                colors[edge * 2],
                colors[edge * 2 + 1],
                "an edge is one color"
            );
            for other in (edge + 1)..3 {
                assert_ne!(
                    colors[edge * 2],
                    colors[other * 2],
                    "edges {edge} and {other} share vertex 0 and got one color"
                );
            }
        }
    }
}
