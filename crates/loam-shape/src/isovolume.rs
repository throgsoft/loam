//! For a 1-Lipschitz field, the half-diagonal margin encloses its nonpositive set within the sampled domain.
//! Hart, Sphere Tracing, 1996, §2.

use glam::{Vec3, Vec4};

use crate::Shape;

/// Why [`Isovolume::extract`] refuses its sampling domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IsovolumeError {
    NoCells,
    /// The axis whose extent is not positive.
    EmptyAxis(usize),
}

impl std::fmt::Display for IsovolumeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoCells => write!(f, "the resolution must be at least one cell"),
            Self::EmptyAxis(axis) => write!(f, "the domain has no positive extent on axis {axis}"),
        }
    }
}

impl std::error::Error for IsovolumeError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CellBox<const D: usize> {
    lo: [usize; D],
    hi: [usize; D],
}

/// Encloses `{p : f(p) <= 0}` within the sampled domain; [`Self::clipped`] detects contact with its boundary.
#[derive(Clone, Debug)]
pub struct Isovolume<const D: usize> {
    origin: [f32; D],
    cell: f32,
    boxes: Vec<CellBox<D>>,
    occupied_cells: usize,
    clipped: bool,
}

impl<const D: usize> Isovolume<D> {
    /// Requires a 1-Lipschitz signed distance; `resolution` counts cells along the longest axis.
    pub fn extract(
        min: [f32; D],
        max: [f32; D],
        resolution: usize,
        sdf: impl Fn([f32; D]) -> f32,
    ) -> Result<Self, IsovolumeError> {
        if resolution == 0 {
            return Err(IsovolumeError::NoCells);
        }
        let mut extent = [0.0f32; D];
        for (axis, (e, (lo, hi))) in extent
            .iter_mut()
            .zip(min.iter().zip(max.iter()))
            .enumerate()
        {
            *e = hi - lo;
            if e.is_nan() || *e <= 0.0 {
                return Err(IsovolumeError::EmptyAxis(axis));
            }
        }

        let cell = extent.iter().copied().fold(0.0f32, f32::max) / resolution as f32;
        let mut counts = [0usize; D];
        for (n, e) in counts.iter_mut().zip(extent.iter()) {
            *n = ((e / cell).ceil() as usize).max(1);
        }

        let margin = 0.5 * cell * (D as f32).sqrt();
        let total: usize = counts.iter().product();
        let mut occupied = vec![false; total];
        let mut occupied_cells = 0usize;
        let mut clipped = false;
        for (index, marked) in occupied.iter_mut().enumerate() {
            let coords = unflatten(index, &counts);
            if sdf(cell_centre(&min, cell, &coords)) > margin {
                continue;
            }
            *marked = true;
            occupied_cells += 1;
            clipped |= coords
                .iter()
                .zip(counts.iter())
                .any(|(&c, &n)| c == 0 || c + 1 == n);
        }

        let mut covered = vec![false; total];
        let mut boxes = Vec::new();
        let mut seed = 0usize;
        loop {
            while seed < total && (covered[seed] || !occupied[seed]) {
                seed += 1;
            }
            if seed == total {
                break;
            }
            let grown = grow(unflatten(seed, &counts), &occupied, &counts);
            visit_cells(&grown, |coords| {
                covered[flatten(coords, &counts)] = true;
                true
            });
            boxes.push(grown);
        }

        Ok(Self {
            origin: min,
            cell,
            boxes,
            occupied_cells,
            clipped,
        })
    }

    pub fn piece_count(&self) -> usize {
        self.boxes.len()
    }

    pub fn occupied_cells(&self) -> usize {
        self.occupied_cells
    }

    pub fn cell_size(&self) -> f32 {
        self.cell
    }

    /// Upper bound on how far the cover extends past the true surface.
    pub fn enclosure_margin(&self) -> f32 {
        self.cell * (D as f32).sqrt()
    }

    /// Exact volume of the union of the pieces: that union is the marked cell set.
    pub fn volume(&self) -> f32 {
        self.occupied_cells as f32 * self.cell.powi(D as i32)
    }

    /// True when a marked cell touched the sampled domain's boundary.
    pub fn clipped(&self) -> bool {
        self.clipped
    }

    /// Corner-to-corner, in the grid frame.
    pub fn piece_bounds(&self, index: usize) -> ([f32; D], [f32; D]) {
        let b = self.boxes[index];
        let mut lo = [0.0f32; D];
        let mut hi = [0.0f32; D];
        for k in 0..D {
            lo[k] = self.origin[k] + b.lo[k] as f32 * self.cell;
            hi[k] = self.origin[k] + (b.hi[k] + 1) as f32 * self.cell;
        }
        (lo, hi)
    }

    /// Linear in piece count: a verification query, not a broadphase.
    pub fn contains(&self, p: [f32; D]) -> bool {
        (0..self.boxes.len()).any(|i| {
            let (lo, hi) = self.piece_bounds(i);
            (0..D).all(|k| p[k] >= lo[k] && p[k] <= hi[k])
        })
    }
}

impl Isovolume<3> {
    /// Pose is extrinsic per the [`Shape`] contract, so the hull is origin-centred.
    pub fn colliders(&self) -> Vec<(Vec3, Shape)> {
        (0..self.boxes.len())
            .map(|i| {
                let (lo, hi) = self.piece_bounds(i);
                let centre = Vec3::from_array(lo).lerp(Vec3::from_array(hi), 0.5);
                let h = 0.5 * (Vec3::from_array(hi) - Vec3::from_array(lo));
                let mut vertices = Vec::with_capacity(8);
                for sz in [-1.0f32, 1.0] {
                    for sy in [-1.0f32, 1.0] {
                        for sx in [-1.0f32, 1.0] {
                            vertices.push(Vec3::new(sx * h.x, sy * h.y, sz * h.z));
                        }
                    }
                }
                (centre, Shape::ConvexPolytope3D { vertices })
            })
            .collect()
    }
}

impl Isovolume<4> {
    pub fn colliders(&self) -> Vec<(Vec4, Shape)> {
        (0..self.boxes.len())
            .map(|i| {
                let (lo, hi) = self.piece_bounds(i);
                let centre = Vec4::from_array(lo).lerp(Vec4::from_array(hi), 0.5);
                let h = 0.5 * (Vec4::from_array(hi) - Vec4::from_array(lo));
                let mut vertices = Vec::with_capacity(16);
                for sw in [-1.0f32, 1.0] {
                    for sz in [-1.0f32, 1.0] {
                        for sy in [-1.0f32, 1.0] {
                            for sx in [-1.0f32, 1.0] {
                                vertices.push(Vec4::new(sx * h.x, sy * h.y, sz * h.z, sw * h.w));
                            }
                        }
                    }
                }
                (centre, Shape::ConvexPolytope4D { vertices })
            })
            .collect()
    }
}

fn cell_centre<const D: usize>(origin: &[f32; D], cell: f32, coords: &[usize; D]) -> [f32; D] {
    let mut p = [0.0f32; D];
    for k in 0..D {
        p[k] = origin[k] + (coords[k] as f32 + 0.5) * cell;
    }
    p
}

fn flatten<const D: usize>(coords: &[usize; D], counts: &[usize; D]) -> usize {
    let mut index = 0;
    for k in (0..D).rev() {
        index = index * counts[k] + coords[k];
    }
    index
}

fn unflatten<const D: usize>(mut index: usize, counts: &[usize; D]) -> [usize; D] {
    let mut coords = [0usize; D];
    for k in 0..D {
        coords[k] = index % counts[k];
        index /= counts[k];
    }
    coords
}

fn visit_cells<const D: usize>(b: &CellBox<D>, mut f: impl FnMut(&[usize; D]) -> bool) {
    let mut coords = b.lo;
    loop {
        if !f(&coords) {
            return;
        }
        let mut axis = 0;
        loop {
            if axis == D {
                return;
            }
            if coords[axis] < b.hi[axis] {
                coords[axis] += 1;
                break;
            }
            coords[axis] = b.lo[axis];
            axis += 1;
        }
    }
}

fn all_occupied<const D: usize>(b: &CellBox<D>, occupied: &[bool], counts: &[usize; D]) -> bool {
    let mut ok = true;
    visit_cells(b, |coords| {
        ok = occupied[flatten(coords, counts)];
        ok
    });
    ok
}

fn grow<const D: usize>(seed: [usize; D], occupied: &[bool], counts: &[usize; D]) -> CellBox<D> {
    let mut b = CellBox { lo: seed, hi: seed };
    loop {
        let mut grew = false;
        for axis in 0..D {
            if b.lo[axis] > 0 {
                let mut layer = b;
                layer.lo[axis] = b.lo[axis] - 1;
                layer.hi[axis] = b.lo[axis] - 1;
                if all_occupied(&layer, occupied, counts) {
                    b.lo[axis] -= 1;
                    grew = true;
                }
            }
            if b.hi[axis] + 1 < counts[axis] {
                let mut layer = b;
                layer.lo[axis] = b.hi[axis] + 1;
                layer.hi[axis] = b.hi[axis] + 1;
                if all_occupied(&layer, occupied, counts) {
                    b.hi[axis] += 1;
                    grew = true;
                }
            }
        }
        if !grew {
            return b;
        }
    }
}
