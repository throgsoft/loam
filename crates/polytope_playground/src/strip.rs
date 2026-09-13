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

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: [u32; 2] = [1280, 720];

    #[test]
    fn the_grid_tiles_the_frame_once_per_cell_and_spans_the_w_extent() {
        const EXTENT: f32 = 0.35;
        const SLICE: f32 = 0.1;
        let strip = Strip {
            on: true,
            w: true,
            t: true,
            count_w: 5,
            count_t: 3,
            t_extent: 2.0,
            ..Strip::default()
        };
        let mut cells = Vec::new();
        strip.cells(FRAME, SLICE, EXTENT, &mut cells);
        assert_eq!(cells.len(), 15, "a 5 by 3 grid has fifteen cells");

        let covered: u64 = cells
            .iter()
            .map(|cell| u64::from(cell.viewport.width) * u64::from(cell.viewport.height))
            .sum();
        assert_eq!(
            covered,
            u64::from(FRAME[0]) * u64::from(FRAME[1]),
            "the cells leave a gap or overlap"
        );

        assert!((cells[0].w - (SLICE - EXTENT)).abs() < 1e-6);
        assert!((cells[14].w - (SLICE + EXTENT)).abs() < 1e-6);
        assert!(
            (cells[0].t - 0.0).abs() < 1e-6 && (cells[2].t - 2.0).abs() < 1e-6,
            "the t fan spans the extent down each column"
        );
    }

    #[test]
    fn one_axis_off_leaves_a_single_row_or_column_at_the_slider() {
        let mut cells = Vec::new();
        let across = Strip {
            on: true,
            w: true,
            t: false,
            count_w: 4,
            ..Strip::default()
        };
        across.cells(FRAME, 0.0, 1.0, &mut cells);
        assert_eq!(cells.len(), 4);
        assert!(cells.iter().all(|cell| cell.t == 0.0));

        let forward = Strip {
            on: true,
            w: false,
            t: true,
            count_t: 4,
            ..Strip::default()
        };
        forward.cells(FRAME, 0.25, 1.0, &mut cells);
        assert_eq!(cells.len(), 4);
        assert!(
            cells.iter().all(|cell| (cell.w - 0.25).abs() < 1e-6),
            "with w off every cell cuts at the slider's own w"
        );
    }
}
