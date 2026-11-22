//! Every grid cell is split into two triangles and clipped against `d <= 0`
//! (Sutherland & Hodgman, 1974). The split sidesteps the saddle ambiguity a
//! marching-squares cell would have.

use glam::{Vec2, Vec3, Vec4};
use loam_shape::{
    Isovolume, LineMesh, NotVisualizable, PointMesh, Shape, TriangleMesh, Visualizable,
};

use super::field::DistanceField2D;
use super::hull::{centroid, convex_hull, double_area, reduce_sides, MAX_HULL_SIDES};
use super::{GlyphParams, BLANK_DISTANCE};

// `Isovolume::extract` requires a true 1-Lipschitz field, which `sample` is not.
const LIPSCHITZ_SCALE: f32 = std::f32::consts::FRAC_1_SQRT_2;

const COVER_PADDING_CELLS: f32 = 2.0;

const RING_MERGE_FRACTION: f32 = 1.0e-4;

const SLIVER_AREA_FRACTION: f32 = 1.0e-6;

const WIREFRAME_WIDTH_PX: f32 = 1.0;

const POINT_MARKER_PX: f32 = 2.0;

// Counter-clockwise in world XY.
#[derive(Clone, Debug)]
pub(super) struct Piece {
    vertices: [Vec2; 4],
    len: usize,
    cut: Option<usize>,
}

impl Piece {
    fn ring(&self) -> &[Vec2] {
        &self.vertices[..self.len]
    }
}

#[derive(Clone, Debug)]
pub struct GlyphSolid {
    ch: char,
    pen_origin: Vec2,
    advance: f32,
    half_depth: f32,
    slab_center: f32,
    slab_half: f32,
    color: [f32; 4],
    collider_cell: f32,
    field: Option<DistanceField2D>,
    pieces: Vec<Piece>,
    cover: Option<Isovolume<2>>,
    hull: Vec<Vec2>,
}

impl GlyphSolid {
    pub(super) fn new(
        ch: char,
        pen_origin: Vec2,
        advance: f32,
        params: &GlyphParams,
        field: Option<DistanceField2D>,
    ) -> Self {
        let collider_cell = params.em_size / params.collider_resolution as f32;
        let pieces = field.as_ref().map(extract_pieces).unwrap_or_default();
        let cover = field
            .as_ref()
            .map(|field| extract_cover(field, collider_cell));
        let hull = cover.as_ref().map(rigid_ring).unwrap_or_default();
        Self {
            ch,
            pen_origin,
            advance,
            half_depth: 0.5 * params.depth,
            slab_center: 0.5 * (params.slab.0 + params.slab.1),
            slab_half: 0.5 * (params.slab.1 - params.slab.0),
            color: params.color,
            collider_cell,
            field,
            pieces,
            cover,
            hull,
        }
    }

    pub fn ch(&self) -> char {
        self.ch
    }

    pub fn pen_origin(&self) -> Vec2 {
        self.pen_origin
    }

    pub fn advance(&self) -> f32 {
        self.advance
    }

    pub fn is_blank(&self) -> bool {
        self.field.is_none()
    }

    pub fn field(&self) -> Option<&DistanceField2D> {
        self.field.as_ref()
    }

    pub fn piece_count(&self) -> usize {
        self.pieces.len()
    }

    /// Use [`Self::collider_margin`] for the enclosure bound of this scaled field.
    pub fn collider_cover(&self) -> Option<&Isovolume<2>> {
        self.cover.as_ref()
    }

    pub fn collider_count(&self) -> usize {
        self.cover.as_ref().map_or(0, Isovolume::piece_count)
    }

    /// Upper bound on how far the collider cover reaches past the letter's baked surface.
    pub fn collider_margin(&self) -> f32 {
        2.0 * self.collider_cell
    }

    /// Negative inside.
    pub fn distance_2d(&self, p: Vec2) -> f32 {
        self.field
            .as_ref()
            .map_or(BLANK_DISTANCE, |field| field.sample(p))
    }

    /// Negative inside, with bilinear `xy` distance and exact `z` and `w` interval distances.
    pub fn distance_4d(&self, p: Vec4) -> f32 {
        let cross_section = self.distance_2d(Vec2::new(p.x, p.y));
        let extruded = extend(cross_section, p.z, 0.0, self.half_depth);
        extend(extruded, p.w, self.slab_center, self.slab_half)
    }

    /// Origin-centred box colliders for static geometry; use [`Self::rigid_hull_4d`] for a single dynamic body.
    pub fn colliders_4d(&self) -> Vec<(Vec4, Shape)> {
        let Some(cover) = &self.cover else {
            return Vec::new();
        };
        (0..cover.piece_count())
            .map(|index| {
                let (lo, hi) = cover.piece_bounds(index);
                let centre = Vec4::new(
                    0.5 * (lo[0] + hi[0]),
                    0.5 * (lo[1] + hi[1]),
                    0.0,
                    self.slab_center,
                );
                let half = Vec4::new(
                    0.5 * (hi[0] - lo[0]),
                    0.5 * (hi[1] - lo[1]),
                    self.half_depth,
                    self.slab_half,
                );
                let mut vertices = Vec::with_capacity(16);
                for sw in [-1.0f32, 1.0] {
                    for sz in [-1.0f32, 1.0] {
                        for sy in [-1.0f32, 1.0] {
                            for sx in [-1.0f32, 1.0] {
                                vertices.push(half * Vec4::new(sx, sy, sz, sw));
                            }
                        }
                    }
                }
                (centre, Shape::ConvexPolytope4D { vertices })
            })
            .collect()
    }

    pub fn rigid_hull_sides(&self) -> usize {
        self.hull.len()
    }

    /// A convex prism with at most 32 vertices, centred at its centre of mass; fills glyph counters.
    pub fn rigid_hull_4d(&self) -> Option<(Vec4, Shape)> {
        if self.hull.is_empty() {
            return None;
        }
        let centre_2d = centroid(&self.hull);
        let centre = Vec4::new(centre_2d.x, centre_2d.y, 0.0, self.slab_center);
        let mut vertices = Vec::with_capacity(4 * self.hull.len());
        for sw in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                for p in &self.hull {
                    let offset = *p - centre_2d;
                    vertices.push(Vec4::new(
                        offset.x,
                        offset.y,
                        sz * self.half_depth,
                        sw * self.slab_half,
                    ));
                }
            }
        }
        Some((centre, Shape::ConvexPolytope4D { vertices }))
    }
}

fn rigid_ring(cover: &Isovolume<2>) -> Vec<Vec2> {
    let mut corners = Vec::with_capacity(4 * cover.piece_count());
    for index in 0..cover.piece_count() {
        let (lo, hi) = cover.piece_bounds(index);
        for y in [lo[1], hi[1]] {
            for x in [lo[0], hi[0]] {
                corners.push(Vec2::new(x, y));
            }
        }
    }
    let mut ring = convex_hull(corners);
    reduce_sides(&mut ring, MAX_HULL_SIDES);
    ring
}

fn extract_cover(field: &DistanceField2D, cell: f32) -> Isovolume<2> {
    let (cells_x, cells_y) = field.cell_counts();
    let padding = Vec2::splat(COVER_PADDING_CELLS * cell);
    let lo = field.corner(0, 0) - padding;
    let span = field.corner(cells_x, cells_y) + padding - lo;
    let counts = (span / cell).ceil().max(Vec2::ONE);
    let resolution = counts.max_element() as usize;
    Isovolume::extract(
        lo.to_array(),
        (lo + counts * cell).to_array(),
        resolution,
        |p| LIPSCHITZ_SCALE * field.sample(Vec2::from_array(p)),
    )
}

/// Appends the triangulated extrusion and returns whether the field enclosed any area.
pub fn append_field_prism(
    field: &DistanceField2D,
    half_depth: f32,
    color: [f32; 4],
    mesh: &mut TriangleMesh<3>,
) -> bool {
    let mut emitted = false;
    for_each_piece(field, |piece| {
        append_prism(std::slice::from_ref(&piece), half_depth, color, mesh);
        emitted = true;
    });
    emitted
}

fn append_prism(pieces: &[Piece], half_depth: f32, color: [f32; 4], mesh: &mut TriangleMesh<3>) {
    for piece in pieces {
        let n = piece.ring().len();
        let back = mesh.vertices.len() as u32;
        let front = back + n as u32;
        for z in [-half_depth, half_depth] {
            for q in piece.ring() {
                mesh.vertices.push([q.x, q.y, z]);
                mesh.colors.push(color);
            }
        }
        for k in 1..n as u32 - 1 {
            mesh.indices.push([front, front + k, front + k + 1]);
            mesh.indices.push([back, back + k + 1, back + k]);
        }
        if let Some(cut) = piece.cut {
            let a = cut as u32;
            let b = ((cut + 1) % n) as u32;
            mesh.indices.push([back + a, back + b, front + b]);
            mesh.indices.push([back + a, front + b, front + a]);
        }
    }
}

impl Visualizable<3> for GlyphSolid {
    fn to_triangles(&self) -> Result<TriangleMesh<3>, NotVisualizable> {
        if self.pieces.is_empty() {
            return Err(NotVisualizable::Degenerate);
        }
        let mut mesh = TriangleMesh::default();
        append_prism(&self.pieces, self.half_depth, self.color, &mut mesh);
        Ok(mesh)
    }

    fn to_lines(&self) -> Result<LineMesh<3>, NotVisualizable> {
        if self.pieces.is_empty() {
            return Err(NotVisualizable::Degenerate);
        }
        let mut mesh = LineMesh::default();
        let mut push = |from: Vec3, to: Vec3| {
            mesh.segments.push((from.to_array(), to.to_array()));
            mesh.colors.push((self.color, self.color));
            mesh.widths.push(WIREFRAME_WIDTH_PX);
        };
        for piece in &self.pieces {
            let Some(cut) = piece.cut else { continue };
            let a = piece.ring()[cut];
            let b = piece.ring()[(cut + 1) % piece.ring().len()];
            let (a_back, a_front) = (at_z(a, -self.half_depth), at_z(a, self.half_depth));
            let (b_back, b_front) = (at_z(b, -self.half_depth), at_z(b, self.half_depth));
            push(a_back, b_back);
            push(a_front, b_front);
            push(a_back, a_front);
            push(b_back, b_front);
        }
        Ok(mesh)
    }

    fn to_points(&self) -> Result<PointMesh<3>, NotVisualizable> {
        if self.pieces.is_empty() {
            return Err(NotVisualizable::Degenerate);
        }
        let mut mesh = PointMesh::default();
        for piece in &self.pieces {
            let Some(cut) = piece.cut else { continue };
            let a = piece.ring()[cut];
            for z in [-self.half_depth, self.half_depth] {
                mesh.positions.push(at_z(a, z).to_array());
                mesh.colors.push(self.color);
                mesh.sizes.push(POINT_MARKER_PX);
            }
        }
        Ok(mesh)
    }
}

fn at_z(p: Vec2, z: f32) -> Vec3 {
    Vec3::new(p.x, p.y, z)
}

// Quilez, "distance functions" (2019), `opExtrusion`.
fn extend(d: f32, x: f32, center: f32, half: f32) -> f32 {
    let axial = (x - center).abs() - half;
    Vec2::new(d, axial).max(Vec2::ZERO).length() + d.max(axial).min(0.0)
}

fn extract_pieces(field: &DistanceField2D) -> Vec<Piece> {
    let mut pieces = Vec::new();
    for_each_piece(field, |piece| pieces.push(piece));
    pieces
}

fn for_each_piece(field: &DistanceField2D, mut emit: impl FnMut(Piece)) {
    let (cells_x, cells_y) = field.cell_counts();
    let cell = field.cell_size();

    for j in 0..cells_y {
        for i in 0..cells_x {
            let indices = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
            let corners: [Corner; 4] = std::array::from_fn(|k| {
                let (ci, cj) = indices[k];
                Corner {
                    position: field.corner(ci, cj),
                    distance: field.at(ci, cj),
                }
            });
            let triangles: [[usize; 3]; 2] = if (i + j) % 2 == 0 {
                [[0, 1, 2], [0, 2, 3]]
            } else {
                [[0, 1, 3], [1, 2, 3]]
            };
            for tri in triangles {
                let corners = [
                    corners[tri[0]].clone(),
                    corners[tri[1]].clone(),
                    corners[tri[2]].clone(),
                ];
                if let Some(piece) = clip_triangle(&corners, cell) {
                    emit(piece);
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Corner {
    position: Vec2,
    distance: f32,
}

fn clip_triangle(tri: &[Corner; 3], cell: f32) -> Option<Piece> {
    let mut ring = [Vec2::ZERO; 4];
    let mut on_isoline = [false; 4];
    let mut len = 0;
    for k in 0..3 {
        let a = &tri[k];
        let b = &tri[(k + 1) % 3];
        if a.distance <= 0.0 {
            ring[len] = a.position;
            on_isoline[len] = a.distance == 0.0;
            len += 1;
        }
        if (a.distance <= 0.0) != (b.distance <= 0.0) {
            let t = a.distance / (a.distance - b.distance);
            ring[len] = a.position + (b.position - a.position) * t;
            on_isoline[len] = true;
            len += 1;
        }
    }
    let merge2 = (cell * RING_MERGE_FRACTION).powi(2);
    let mut kept = 0;
    for index in 0..len {
        if kept > 0 && ring[kept - 1].distance_squared(ring[index]) <= merge2 {
            on_isoline[kept - 1] |= on_isoline[index];
        } else {
            ring[kept] = ring[index];
            on_isoline[kept] = on_isoline[index];
            kept += 1;
        }
    }
    while kept >= 2 && ring[0].distance_squared(ring[kept - 1]) <= merge2 {
        kept -= 1;
        on_isoline[0] |= on_isoline[kept];
    }
    if kept < 3 || double_area(&ring[..kept]) <= SLIVER_AREA_FRACTION * cell * cell {
        return None;
    }
    let cut = (0..kept).find(|&k| on_isoline[k] && on_isoline[(k + 1) % kept]);
    Some(Piece {
        vertices: ring,
        len: kept,
        cut,
    })
}
