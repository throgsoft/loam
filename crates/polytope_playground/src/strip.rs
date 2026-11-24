use loam::math::{Bivector, Bivector4, Rotor4};
use loam::render::raymarch::{shape_id, BodyUniform};
use loam::render::Viewport;

use crate::catalog::SHAPE_CATALOG;
use crate::consts::{BODY_SIZE, BODY_Y};

pub(crate) const MIN_CELLS: usize = 3;

pub(crate) const MAX_CELLS: usize = 21;

pub(crate) const MIN_T_EXTENT: f32 = 0.1;

pub(crate) const MAX_T_EXTENT: f32 = 10.0;

const DEFAULT_T_EXTENT: f32 = 3.0;

const DEFAULT_SUBJECT: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Strip {
    pub(crate) on: bool,
    pub(crate) w: bool,
    pub(crate) t: bool,
    pub(crate) swap_axes: bool,
    pub(crate) count_w: usize,
    pub(crate) count_t: usize,
    pub(crate) t_extent: f32,
    pub(crate) subject: usize,
}

impl Default for Strip {
    fn default() -> Self {
        Self {
            on: false,
            w: true,
            t: false,
            swap_axes: false,
            count_w: 11,
            count_t: 5,
            t_extent: DEFAULT_T_EXTENT,
            subject: DEFAULT_SUBJECT,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cell {
    pub(crate) viewport: Viewport,
    pub(crate) w: f32,
    pub(crate) t: f32,
}

impl Strip {
    pub(crate) fn subject(&self) -> usize {
        self.subject.min(SHAPE_CATALOG.len() - 1)
    }

    pub(crate) fn grid(&self) -> (usize, usize, bool) {
        match (self.w, self.t) {
            (true, true) if self.swap_axes => (self.count_t, self.count_w, false),
            (true, true) => (self.count_w, self.count_t, true),
            (true, false) => (self.count_w, 1, true),
            (false, true) => (1, self.count_t, true),
            (false, false) => (1, 1, true),
        }
    }

    pub(crate) fn cells(&self, frame: [u32; 2], slice: f32, w_extent: f32, out: &mut Vec<Cell>) {
        out.clear();
        let (cols, rows, w_on_cols) = self.grid();
        if cols == 0 || rows == 0 {
            return;
        }
        for (col, column) in Viewport::full(frame)
            .split_horizontal(cols as u32)
            .enumerate()
        {
            for (row, viewport) in column.split_vertical(rows as u32).enumerate() {
                let (w_index, w_count, t_index, t_count) = if w_on_cols {
                    (col, cols, row, rows)
                } else {
                    (row, rows, col, cols)
                };
                let across = if !self.w || w_count <= 1 {
                    0.5
                } else {
                    w_index as f32 / (w_count - 1) as f32
                };
                let along = if !self.t || t_count <= 1 {
                    0.0
                } else {
                    t_index as f32 / (t_count - 1) as f32 * self.t_extent
                };
                out.push(Cell {
                    viewport,
                    w: slice + (across * 2.0 - 1.0) * w_extent,
                    t: along,
                });
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct Scratch {
    pub(crate) cells: Vec<Cell>,
    pub(crate) strip: Vec<(Viewport, f32, BodyUniform)>,
}

pub(crate) fn fill_strip(
    strip: &Strip,
    omega: Bivector4,
    subject: Rotor4,
    frame: (u32, u32),
    slice: f32,
    scratch: &mut Scratch,
) {
    scratch.strip.clear();
    if !strip.on {
        scratch.cells.clear();
        return;
    }
    strip.cells([frame.0, frame.1], slice, BODY_SIZE, &mut scratch.cells);
    let entry = SHAPE_CATALOG[strip.subject()];
    for cell in &scratch.cells {
        let rotor = ((omega * cell.t).exp() * subject).normalize();
        scratch.strip.push((
            cell.viewport,
            cell.w,
            BodyUniform::polytope_with_rotor(
                [0.0, BODY_Y, 0.0, 0.0],
                shape_id(entry.shape),
                BODY_SIZE,
                rotor,
                entry.body_color,
            ),
        ));
    }
}
