/// Pixel rectangle, origin top-left, +y down.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Viewport {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Viewport {
    pub fn full(framebuffer: [u32; 2]) -> Self {
        Self {
            x: 0,
            y: 0,
            width: framebuffer[0],
            height: framebuffer[1],
        }
    }

    pub fn apply(&self, rp: &mut wgpu::RenderPass<'_>) {
        rp.set_viewport(
            self.x as f32,
            self.y as f32,
            self.width as f32,
            self.height as f32,
            0.0,
            1.0,
        );
    }

    pub fn resolution_f32(&self) -> [f32; 2] {
        [self.width as f32, self.height as f32]
    }

    /// The last cell absorbs the rounding remainder; empty when `n == 0`.
    pub fn split_horizontal(&self, n: u32) -> impl Iterator<Item = Viewport> {
        let viewport = *self;
        (0..n).map(move |i| {
            let width = viewport.width / n;
            Viewport {
                x: viewport.x + i * width,
                width: if i + 1 == n {
                    viewport.width - i * width
                } else {
                    width
                },
                ..viewport
            }
        })
    }

    pub fn split_vertical(&self, n: u32) -> impl Iterator<Item = Viewport> {
        let viewport = *self;
        (0..n).map(move |i| {
            let height = viewport.height / n;
            Viewport {
                y: viewport.y + i * height,
                height: if i + 1 == n {
                    viewport.height - i * height
                } else {
                    height
                },
                ..viewport
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uneven_split() {
        let v = Viewport {
            x: 100,
            y: 50,
            width: 17,
            height: 30,
        };
        let cells: Vec<_> = v.split_horizontal(5).collect();
        assert_eq!(cells.len(), 5);
        assert_eq!(cells.first().unwrap().x, 100);
        assert_eq!(
            cells.last().unwrap().x + cells.last().unwrap().width,
            100 + 17
        );
        for win in cells.windows(2) {
            assert_eq!(win[0].x + win[0].width, win[1].x);
        }
        assert_eq!(cells[0].width, 3);
        assert_eq!(cells[4].width, 5);
    }
}
