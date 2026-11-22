use std::sync::LazyLock;

use glam::{Vec3, Vec4};

use crate::polytope_geom::{
    cell120_vertices, cell16_vertices, cell24_vertices, cell600_vertices, pentatope_vertices,
    tesseract_vertices,
};

/// Discriminants match `loam_render::raymarch::SHAPE_*`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Polytope4 {
    Pentatope = 0,
    Tesseract = 1,
    Cell16 = 2,
    Cell24 = 3,
    Cell120 = 4,
    Cell600 = 5,
}

#[derive(Debug)]
pub struct Polytope4Topology {
    /// Canonical (unit-circumradius) coordinates.
    pub vertices: &'static [Vec4],
    /// `[lo, hi]` pairs, lexicographically sorted for deterministic iteration.
    pub edges: &'static [[u32; 2]],
    /// Ascending index lists, lexicographically sorted for the same reason.
    pub cells: &'static [&'static [u32]],
}

impl Polytope4 {
    /// In `repr(u32)` discriminant order.
    pub const ALL: [Polytope4; 6] = [
        Polytope4::Pentatope,
        Polytope4::Tesseract,
        Polytope4::Cell16,
        Polytope4::Cell24,
        Polytope4::Cell120,
        Polytope4::Cell600,
    ];

    pub fn topology(self) -> &'static Polytope4Topology {
        match self {
            Polytope4::Pentatope => &PENTATOPE_TOPOLOGY,
            Polytope4::Tesseract => &TESSERACT_TOPOLOGY,
            Polytope4::Cell16 => &CELL16_TOPOLOGY,
            Polytope4::Cell24 => &CELL24_TOPOLOGY,
            Polytope4::Cell120 => &CELL120_TOPOLOGY,
            Polytope4::Cell600 => &CELL600_TOPOLOGY,
        }
    }

    pub fn vertex_count(self) -> usize {
        self.topology().vertices.len()
    }

    /// Undirected edges, each counted once.
    pub fn edge_count(self) -> usize {
        self.topology().edges.len()
    }

    /// Bounding 3-cells (the facets), not the full face lattice.
    pub fn cell_count(self) -> usize {
        self.topology().cells.len()
    }

    /// Length equal to the inradius, along the cell's outward face normal.
    pub fn cell_centers(self) -> Vec<Vec4> {
        let topo = self.topology();
        topo.cells
            .iter()
            .map(|cell| {
                cell.iter()
                    .map(|&i| topo.vertices[i as usize])
                    .sum::<Vec4>()
                    / cell.len() as f32
            })
            .collect()
    }

    /// Outward unit normals and mean facet offset in canonical coordinates.
    pub fn face_planes(self) -> (Vec<Vec4>, f32) {
        let topo = self.topology();
        let mut normals = Vec::with_capacity(topo.cells.len());
        let mut inradius_sum = 0.0;
        for cell in topo.cells {
            let centroid: Vec4 = cell
                .iter()
                .map(|&i| topo.vertices[i as usize])
                .sum::<Vec4>()
                / cell.len() as f32;
            let r = centroid.length();

            normals.push(centroid / r);
            inradius_sum += r;
        }
        let inradius = inradius_sum / normals.len() as f32;
        (normals, inradius)
    }
}

const DEFAULT_LINE_COLOR: [f32; 4] = [0.9, 0.9, 0.9, 1.0];
const DEFAULT_LINE_WIDTH: f32 = 1.5;

impl crate::Visualizable<4> for Polytope4 {
    fn to_lines(&self) -> Result<crate::LineMesh<4>, crate::NotVisualizable> {
        let topo = self.topology();
        let mut mesh = crate::LineMesh::<4>::default();
        mesh.segments.reserve(topo.edges.len());
        mesh.colors.reserve(topo.edges.len());
        mesh.widths.reserve(topo.edges.len());
        for &[i, j] in topo.edges {
            let a = topo.vertices[i as usize].to_array();
            let b = topo.vertices[j as usize].to_array();
            mesh.segments.push((a, b));
            mesh.colors.push((DEFAULT_LINE_COLOR, DEFAULT_LINE_COLOR));
            mesh.widths.push(DEFAULT_LINE_WIDTH);
        }
        Ok(mesh)
    }

    fn to_triangles(&self) -> Result<crate::TriangleMesh<4>, crate::NotVisualizable> {
        Err(crate::NotVisualizable::Degenerate)
    }

    fn to_points(&self) -> Result<crate::PointMesh<4>, crate::NotVisualizable> {
        let topo = self.topology();
        let mut mesh = crate::PointMesh::<4>::default();
        mesh.positions.reserve(topo.vertices.len());
        mesh.colors.reserve(topo.vertices.len());
        mesh.sizes.reserve(topo.vertices.len());
        for v in topo.vertices {
            mesh.positions.push(v.to_array());
            mesh.colors.push(DEFAULT_LINE_COLOR);
            mesh.sizes.push(4.0);
        }
        Ok(mesh)
    }
}

impl Polytope4 {
    /// Indexes `palette` modulo its length.
    pub fn lines_colored_by_cell(self, palette: &[[f32; 4]]) -> crate::LineMesh<4> {
        let topo = self.topology();
        let mut mesh = crate::LineMesh::<4>::default();
        let n_palette = palette.len().max(1);
        for &[i, j] in topo.edges {
            let cell_idx = topo
                .cells
                .iter()
                .position(|cell| cell.contains(&i) && cell.contains(&j))
                .unwrap_or(0);
            let color = palette
                .get(cell_idx % n_palette)
                .copied()
                .unwrap_or(DEFAULT_LINE_COLOR);
            mesh.segments.push((
                topo.vertices[i as usize].to_array(),
                topo.vertices[j as usize].to_array(),
            ));
            mesh.colors.push((color, color));
            mesh.widths.push(DEFAULT_LINE_WIDTH);
        }
        mesh
    }

    pub fn lines_colored_by_position(self) -> crate::LineMesh<4> {
        let topo = self.topology();
        let mut mesh = crate::LineMesh::<4>::default();
        mesh.segments.reserve(topo.edges.len());
        mesh.colors.reserve(topo.edges.len());
        mesh.widths.reserve(topo.edges.len());
        for &[i, j] in topo.edges {
            let va = topo.vertices[i as usize];
            let vb = topo.vertices[j as usize];
            mesh.segments.push((va.to_array(), vb.to_array()));
            mesh.colors
                .push((vertex_color_by_position(va), vertex_color_by_position(vb)));
            mesh.widths.push(DEFAULT_LINE_WIDTH);
        }
        mesh
    }
}

pub fn vertex_color_by_position(v: Vec4) -> [f32; 4] {
    let n = v.try_normalize().unwrap_or(Vec4::ZERO);
    let bias = |c: f32| 0.25 + 0.75 * (0.5 + 0.5 * c);
    let w_mod = 0.7 + 0.3 * (0.5 + 0.5 * n.w);
    [bias(n.x) * w_mod, bias(n.y) * w_mod, bias(n.z) * w_mod, 1.0]
}

#[cfg(test)]
const SECTION_FILL_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 0.55];
const SECTION_EDGE_COLOR: [f32; 4] = [0.30, 0.85, 0.95, 1.0];
const SECTION_EDGE_WIDTH: f32 = 2.0;

#[cfg(test)]
fn polytope_section_overlay_with_vertices(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: loam_math::WPlane,
) -> (crate::TriangleMesh<3>, crate::LineMesh<3>) {
    let mut tri_mesh = crate::TriangleMesh::<3>::default();
    let mut edge_mesh = crate::LineMesh::<3>::default();
    let mut scratch = SectionScratch::default();

    for_each_section_cap(
        edges,
        cells,
        vertices,
        slice,
        &mut scratch,
        |ordered, centroid| {
            push_cap_fan(ordered, centroid, SECTION_FILL_COLOR, &mut tri_mesh);
            push_cap_perimeter(ordered, &mut edge_mesh);
        },
    );

    (tri_mesh, edge_mesh)
}

pub fn polytope_section_perimeter_append(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: loam_math::WPlane,
    scratch: &mut SectionScratch,
    out: &mut crate::LineMesh<3>,
) {
    for_each_section_cap(edges, cells, vertices, slice, scratch, |ordered, _| {
        push_cap_perimeter(ordered, out);
    });
}

fn push_cap_fan(
    ordered: &[Vec3],
    centroid: Vec3,
    color: [f32; 4],
    out: &mut crate::TriangleMesh<3>,
) {
    let cv_base = out.vertices.len() as u32;
    out.vertices.push(centroid.to_array());
    out.colors.push(color);
    for cap_v in ordered {
        out.vertices.push(cap_v.to_array());
        out.colors.push(color);
    }
    let n = ordered.len() as u32;
    for k in 0..n {
        let k_next = (k + 1) % n;
        out.indices
            .push([cv_base, cv_base + 1 + k, cv_base + 1 + k_next]);
    }
}

fn push_cap_perimeter(ordered: &[Vec3], out: &mut crate::LineMesh<3>) {
    for k in 0..ordered.len() {
        let a = ordered[k];
        let b = ordered[(k + 1) % ordered.len()];
        out.segments.push((a.to_array(), b.to_array()));
        out.colors.push((SECTION_EDGE_COLOR, SECTION_EDGE_COLOR));
        out.widths.push(SECTION_EDGE_WIDTH);
    }
}

pub fn polytope_section_faces_append(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: loam_math::WPlane,
    color: [f32; 4],
    scratch: &mut SectionScratch,
    out: &mut crate::TriangleMesh<3>,
) {
    for_each_section_cap(
        edges,
        cells,
        vertices,
        slice,
        scratch,
        |ordered, centroid| push_cap_fan(ordered, centroid, color, out),
    );
}

#[derive(Debug, Default)]
pub struct SectionScratch {
    cap: Vec<Vec3>,
    keys: Vec<(usize, f32)>,
    ordered: Vec<Vec3>,
}

// `emit` runs in topology cell order, so mesh output is deterministic.
fn for_each_section_cap(
    edges: &[[u32; 2]],
    cells: &[&[u32]],
    vertices: &[Vec4],
    slice: loam_math::WPlane,
    scratch: &mut SectionScratch,
    mut emit: impl FnMut(&[Vec3], Vec3),
) {
    let slice = perturb_slice_if_needed(slice, vertices);

    let polytope_center_r3: Vec3 = if vertices.is_empty() {
        Vec3::ZERO
    } else {
        let mean: Vec4 = vertices.iter().copied().sum::<Vec4>() / vertices.len() as f32;
        Vec3::new(mean.x, mean.y, mean.z)
    };

    for cell in cells {
        let (w_min, w_max) = cell_w_range(cell, vertices);
        if w_max < slice.w_slice - loam_math::SLICE_PERTURBATION_EPSILON
            || w_min > slice.w_slice + loam_math::SLICE_PERTURBATION_EPSILON
        {
            continue;
        }

        let cap = &mut scratch.cap;
        cap.clear();
        for &[i, j] in edges {
            if !cell.contains(&i) || !cell.contains(&j) {
                continue;
            }
            if let Some((_, p3)) = slice.intersect_edge(vertices[i as usize], vertices[j as usize])
            {
                cap.push(p3);
            }
        }
        if cap.len() < 3 {
            continue;
        }

        let centroid: Vec3 = cap.iter().copied().sum::<Vec3>() / cap.len() as f32;
        let Some((basis_u, mut basis_v)) = fit_plane_basis(centroid, cap) else {
            continue;
        };

        let outward = centroid - polytope_center_r3;
        let face_normal = basis_u.cross(basis_v);
        if outward.length_squared() > 1e-12 && face_normal.dot(outward) < 0.0 {
            basis_v = -basis_v;
        }

        order_around_centroid(
            &scratch.cap,
            centroid,
            basis_u,
            basis_v,
            &mut scratch.keys,
            &mut scratch.ordered,
        );

        emit(&scratch.ordered, centroid);
    }
}

fn perturb_slice_if_needed(slice: loam_math::WPlane, vertices: &[Vec4]) -> loam_math::WPlane {
    let eps = loam_math::SLICE_PERTURBATION_EPSILON;
    let near = vertices.iter().any(|v| (v.w - slice.w_slice).abs() < eps);
    if near {
        loam_math::WPlane::new(slice.w_slice + eps)
    } else {
        slice
    }
}

fn cell_w_range(cell: &[u32], vertices: &[Vec4]) -> (f32, f32) {
    let mut w_min = f32::INFINITY;
    let mut w_max = f32::NEG_INFINITY;
    for &i in cell {
        let w = vertices[i as usize].w;
        if w < w_min {
            w_min = w;
        }
        if w > w_max {
            w_max = w;
        }
    }
    (w_min, w_max)
}

fn fit_plane_basis(centroid: Vec3, points: &[Vec3]) -> Option<(Vec3, Vec3)> {
    let eps = loam_math::EDGE_PARALLEL_EPSILON;
    let mut basis_u = Vec3::ZERO;
    for p in points {
        let off = *p - centroid;
        if off.length_squared() > eps * eps {
            basis_u = off.normalize();
            break;
        }
    }
    if basis_u == Vec3::ZERO {
        return None;
    }
    for p in points {
        let off = *p - centroid;
        let cross = basis_u.cross(off);
        if cross.length_squared() > eps * eps {
            let normal = cross.normalize();
            let basis_v = normal.cross(basis_u);
            return Some((basis_u, basis_v));
        }
    }
    None
}

fn order_around_centroid(
    points: &[Vec3],
    centroid: Vec3,
    basis_u: Vec3,
    basis_v: Vec3,
    keys: &mut Vec<(usize, f32)>,
    ordered: &mut Vec<Vec3>,
) {
    keys.clear();
    keys.extend(points.iter().enumerate().map(|(i, p)| {
        let off = *p - centroid;
        let angle = off.dot(basis_v).atan2(off.dot(basis_u));
        (i, angle)
    }));

    keys.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    ordered.clear();
    ordered.extend(keys.iter().map(|&(i, _)| points[i]));
}

static PENTATOPE_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(pentatope_vertices(1.0).into_boxed_slice()));
static TESSERACT_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(tesseract_vertices(1.0).into_boxed_slice()));
static CELL16_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(cell16_vertices(1.0).into_boxed_slice()));
static CELL24_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(cell24_vertices(1.0).into_boxed_slice()));
static CELL120_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(cell120_vertices(1.0).into_boxed_slice()));
static CELL600_VERTICES: LazyLock<&'static [Vec4]> =
    LazyLock::new(|| Box::leak(cell600_vertices(1.0).into_boxed_slice()));

// Coxeter, Regular Polytopes, Table I.
fn canonical_edge_length(p: Polytope4) -> f32 {
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let sqrt2 = 2.0_f32.sqrt();
    match p {
        Polytope4::Pentatope => (5.0_f32 / 2.0).sqrt(),
        Polytope4::Tesseract => 1.0,
        Polytope4::Cell16 => sqrt2,
        Polytope4::Cell24 => 1.0,
        Polytope4::Cell120 => 1.0 / (phi * phi * sqrt2),
        Polytope4::Cell600 => 1.0 / phi,
    }
}

const EDGE_TOLERANCE: f32 = 1e-4;

fn derive_edges(vertices: &[Vec4], edge_length: f32) -> Vec<[u32; 2]> {
    let mut edges = Vec::new();
    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            let d = (vertices[i] - vertices[j]).length();
            if (d - edge_length).abs() < EDGE_TOLERANCE {
                edges.push([i as u32, j as u32]);
            }
        }
    }
    edges
}

fn cache_edges(vertices: &'static [Vec4], edge_length: f32) -> &'static [[u32; 2]] {
    Box::leak(derive_edges(vertices, edge_length).into_boxed_slice())
}

static PENTATOPE_EDGES: LazyLock<&'static [[u32; 2]]> = LazyLock::new(|| {
    cache_edges(
        *PENTATOPE_VERTICES,
        canonical_edge_length(Polytope4::Pentatope),
    )
});
static TESSERACT_EDGES: LazyLock<&'static [[u32; 2]]> = LazyLock::new(|| {
    cache_edges(
        *TESSERACT_VERTICES,
        canonical_edge_length(Polytope4::Tesseract),
    )
});
static CELL16_EDGES: LazyLock<&'static [[u32; 2]]> =
    LazyLock::new(|| cache_edges(*CELL16_VERTICES, canonical_edge_length(Polytope4::Cell16)));
static CELL24_EDGES: LazyLock<&'static [[u32; 2]]> =
    LazyLock::new(|| cache_edges(*CELL24_VERTICES, canonical_edge_length(Polytope4::Cell24)));
static CELL120_EDGES: LazyLock<&'static [[u32; 2]]> =
    LazyLock::new(|| cache_edges(*CELL120_VERTICES, canonical_edge_length(Polytope4::Cell120)));
static CELL600_EDGES: LazyLock<&'static [[u32; 2]]> =
    LazyLock::new(|| cache_edges(*CELL600_VERTICES, canonical_edge_length(Polytope4::Cell600)));

const CELL_TOLERANCE: f32 = 1e-4;

const fn cell_vertex_count(p: Polytope4) -> usize {
    match p {
        Polytope4::Pentatope => 4,
        Polytope4::Tesseract => 8,
        Polytope4::Cell16 => 4,
        Polytope4::Cell24 => 6,
        Polytope4::Cell120 => 20,
        Polytope4::Cell600 => 4,
    }
}

fn cross4(a: Vec4, b: Vec4, c: Vec4) -> Vec4 {
    let drop_x = |v: Vec4| Vec3::new(v.y, v.z, v.w);
    let drop_y = |v: Vec4| Vec3::new(v.x, v.z, v.w);
    let drop_z = |v: Vec4| Vec3::new(v.x, v.y, v.w);
    let drop_w = |v: Vec4| Vec3::new(v.x, v.y, v.z);
    Vec4::new(
        drop_x(a).dot(drop_x(b).cross(drop_x(c))),
        -drop_y(a).dot(drop_y(b).cross(drop_y(c))),
        drop_z(a).dot(drop_z(b).cross(drop_z(c))),
        -drop_w(a).dot(drop_w(b).cross(drop_w(c))),
    )
}

fn adjacency(num_vertices: usize, edges: &[[u32; 2]]) -> Vec<Vec<u32>> {
    let mut adj = vec![Vec::new(); num_vertices];
    for &[i, j] in edges {
        adj[i as usize].push(j);
        adj[j as usize].push(i);
    }
    adj
}

const MIN_CROSS4_LENGTH: f32 = 1e-4;

fn derive_cells(vertices: &[Vec4], edges: &[[u32; 2]], cell_size: usize) -> Vec<Vec<u32>> {
    use std::collections::BTreeSet;

    let adj = adjacency(vertices.len(), edges);
    let mut cells_set: BTreeSet<Vec<u32>> = BTreeSet::new();
    for v_idx in 0..vertices.len() {
        let v_0 = vertices[v_idx];
        let neighbors = &adj[v_idx];
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                for k in (j + 1)..neighbors.len() {
                    let n_a = vertices[neighbors[i] as usize];
                    let n_b = vertices[neighbors[j] as usize];
                    let n_c = vertices[neighbors[k] as usize];
                    let normal = cross4(n_a - v_0, n_b - v_0, n_c - v_0);
                    let mag = normal.length();
                    if mag < MIN_CROSS4_LENGTH {
                        continue;
                    }
                    let n = normal / mag;
                    let offset = v_0.dot(n);
                    let on_plane: Vec<u32> = (0..vertices.len() as u32)
                        .filter(|&p| (vertices[p as usize].dot(n) - offset).abs() < CELL_TOLERANCE)
                        .collect();
                    if on_plane.len() == cell_size {
                        cells_set.insert(on_plane);
                    }
                }
            }
        }
    }
    cells_set.into_iter().collect()
}

fn cache_cells(
    vertices: &'static [Vec4],
    edges: &'static [[u32; 2]],
    cell_size: usize,
) -> &'static [&'static [u32]] {
    let cells: Vec<&'static [u32]> = derive_cells(vertices, edges, cell_size)
        .into_iter()
        .map(|c| &*Box::leak(c.into_boxed_slice()))
        .collect();
    Box::leak(cells.into_boxed_slice())
}

static PENTATOPE_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *PENTATOPE_VERTICES,
        *PENTATOPE_EDGES,
        cell_vertex_count(Polytope4::Pentatope),
    )
});
static TESSERACT_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *TESSERACT_VERTICES,
        *TESSERACT_EDGES,
        cell_vertex_count(Polytope4::Tesseract),
    )
});
static CELL16_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *CELL16_VERTICES,
        *CELL16_EDGES,
        cell_vertex_count(Polytope4::Cell16),
    )
});
static CELL24_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *CELL24_VERTICES,
        *CELL24_EDGES,
        cell_vertex_count(Polytope4::Cell24),
    )
});
static CELL120_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *CELL120_VERTICES,
        *CELL120_EDGES,
        cell_vertex_count(Polytope4::Cell120),
    )
});
static CELL600_CELLS: LazyLock<&'static [&'static [u32]]> = LazyLock::new(|| {
    cache_cells(
        *CELL600_VERTICES,
        *CELL600_EDGES,
        cell_vertex_count(Polytope4::Cell600),
    )
});

static PENTATOPE_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *PENTATOPE_VERTICES,
    edges: *PENTATOPE_EDGES,
    cells: *PENTATOPE_CELLS,
});
static TESSERACT_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *TESSERACT_VERTICES,
    edges: *TESSERACT_EDGES,
    cells: *TESSERACT_CELLS,
});
static CELL16_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL16_VERTICES,
    edges: *CELL16_EDGES,
    cells: *CELL16_CELLS,
});
static CELL24_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL24_VERTICES,
    edges: *CELL24_EDGES,
    cells: *CELL24_CELLS,
});
static CELL120_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL120_VERTICES,
    edges: *CELL120_EDGES,
    cells: *CELL120_CELLS,
});
static CELL600_TOPOLOGY: LazyLock<Polytope4Topology> = LazyLock::new(|| Polytope4Topology {
    vertices: *CELL600_VERTICES,
    edges: *CELL600_EDGES,
    cells: *CELL600_CELLS,
});

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn a_rotated_polytope_sections_to_a_closed_surface_inside_and_nothing_outside() {
        let mut state: u32 = 0x517_C0DE;
        let mut rand = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for polytope in Polytope4::ALL {
            let topo = polytope.topology();
            for _ in 0..16 {
                use loam_math::Bivector as _;
                let rotor =
                    loam_math::Bivector4::new(rand(), rand(), rand(), rand(), rand(), rand())
                        .exp()
                        .normalize();
                let rotated: Vec<Vec4> = {
                    use loam_math::Rotor as _;
                    topo.vertices.iter().map(|v| rotor.apply(*v)).collect()
                };
                let section = |w: f32| {
                    polytope_section_overlay_with_vertices(
                        topo.edges,
                        topo.cells,
                        &rotated,
                        loam_math::WPlane::new(w),
                    )
                };

                let (tri, perim) = section(2.0);
                assert!(
                    tri.indices.is_empty() && perim.segments.is_empty(),
                    "{polytope:?} sliced beyond its circumradius yields geometry"
                );

                let slice_w = rand() * 0.8;
                let (tri, perim) = section(slice_w);
                for v in &tri.vertices {
                    for c in v {
                        assert!(c.is_finite(), "{polytope:?} tri vertex non-finite: {v:?}");
                    }
                }
                let n = tri.vertices.len() as u32;
                for &[i0, i1, i2] in &tri.indices {
                    assert!(
                        i0 < n && i1 < n && i2 < n,
                        "{polytope:?} index out of bounds"
                    );
                }
                let (w_min, w_max) = rotated
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
                        (lo.min(v.w), hi.max(v.w))
                    });
                if slice_w > w_min + 0.05 && slice_w < w_max - 0.05 {
                    assert!(
                        !perim.segments.is_empty(),
                        "{polytope:?} slice w={slice_w} inside [{w_min}, {w_max}] but produced empty section"
                    );
                }

                let point = |p: [f32; 3]| p.map(|c| (c * 1.0e4).round() as i32);
                let mut sharing = BTreeMap::new();
                for &(a, b) in &perim.segments {
                    for c in a.iter().chain(b.iter()) {
                        assert!(c.is_finite(), "{polytope:?} perim endpoint non-finite");
                    }
                    let (a, b) = (point(a), point(b));
                    *sharing.entry((a.min(b), a.max(b))).or_insert(0) += 1;
                }
                assert!(
                    sharing.values().all(|&caps| caps == 2),
                    "{polytope:?} slice w={slice_w}: a section edge does not join two caps"
                );
            }
        }
    }
}
