use loam::render::Viewport;

use crate::catalog::SHAPE_CATALOG;

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
