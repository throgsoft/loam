//! Mesh colors are linear RGBA; widths and point radii are in screen pixels.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotVisualizable {
    Unbounded,

    WrongDimension,

    /// No drawable geometry.
    Degenerate,
}

pub trait Visualizable<const N: usize> {
    fn to_lines(&self) -> Result<LineMesh<N>, NotVisualizable>;

    fn to_triangles(&self) -> Result<TriangleMesh<N>, NotVisualizable>;

    fn to_points(&self) -> Result<PointMesh<N>, NotVisualizable>;
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(
    serialize = "[f32; N]: Serialize",
    deserialize = "[f32; N]: Deserialize<'de>"
))]
pub struct LineMesh<const N: usize> {
    pub segments: Vec<([f32; N], [f32; N])>,
    /// `(start_color, end_color)`; `colors.len() == segments.len()`.
    pub colors: Vec<([f32; 4], [f32; 4])>,
    /// Pixels. `widths.len() == segments.len()`.
    pub widths: Vec<f32>,
}

/// Vertices carry no surface normals.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(
    serialize = "[f32; N]: Serialize",
    deserialize = "[f32; N]: Deserialize<'de>"
))]
pub struct TriangleMesh<const N: usize> {
    pub vertices: Vec<[f32; N]>,
    /// Counter-clockwise winding, looking down the normal.
    pub indices: Vec<[u32; 3]>,
    /// `colors.len() == vertices.len()`.
    pub colors: Vec<[f32; 4]>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(
    serialize = "[f32; N]: Serialize",
    deserialize = "[f32; N]: Deserialize<'de>"
))]
pub struct PointMesh<const N: usize> {
    pub positions: Vec<[f32; N]>,
    /// `colors.len() == positions.len()`.
    pub colors: Vec<[f32; 4]>,
    /// Screen-space radius in pixels.
    pub sizes: Vec<f32>,
}
