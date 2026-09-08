// Matches the depth cue in the LineRasterStaticR4 shader.
const W_DEPTH_BACK: [f32; 3] = [0.30, 0.42, 0.58];
const W_DEPTH_FRONT: [f32; 3] = [1.00, 0.78, 0.45];

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h6 = h.fract() * 6.0;
    let c = v * s;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h6.floor() as i32 % 6 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m]
}

fn unique_edge_palette_color(idx: usize) -> [f32; 4] {
    const PHI_INV: f32 = 0.618_034;
    let h = ((idx as f32) * PHI_INV).fract();
    let s = 0.78 + 0.18 * ((idx % 3) as f32 / 2.0);
    let v = 0.92 - 0.18 * (((idx / 3) % 2) as f32);
    let [r, g, b] = hsv_to_rgb(h, s, v);
    [r, g, b, 1.0]
}

// Greedy first-fit coloring of the edge line-graph.
pub(crate) fn unique_edge_palette(edges: &[[u32; 2]]) -> Vec<[f32; 4]> {
    let mut color_idx = Vec::with_capacity(edges.len());
    let mut used = vec![usize::MAX; edges.len()];
    for (i, &[a, b]) in edges.iter().enumerate() {
        for (j, &[c, d]) in edges[..i].iter().enumerate() {
            if a == c || a == d || b == c || b == d {
                used[color_idx[j]] = i;
            }
        }
        let color = used
            .iter()
            .position(|&generation| generation != i)
            .unwrap_or(i);
        color_idx.push(color);
    }
    color_idx
        .into_iter()
        .map(unique_edge_palette_color)
        .collect()
}

pub(crate) fn w_depth_color(w: f32, w_extent_local: f32) -> [f32; 4] {
    let denom = w_extent_local.max(1e-6);
    let t = ((w / denom) * 0.5 + 0.5).clamp(0.0, 1.0);
    [
        W_DEPTH_BACK[0] + (W_DEPTH_FRONT[0] - W_DEPTH_BACK[0]) * t,
        W_DEPTH_BACK[1] + (W_DEPTH_FRONT[1] - W_DEPTH_BACK[1]) * t,
        W_DEPTH_BACK[2] + (W_DEPTH_FRONT[2] - W_DEPTH_BACK[2]) * t,
        1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsv_to_rgb_primaries() {
        let red = hsv_to_rgb(0.0, 1.0, 1.0);
        assert!((red[0] - 1.0).abs() < 1e-5);
        assert!(red[1].abs() < 1e-5);
        assert!(red[2].abs() < 1e-5);

        let green = hsv_to_rgb(1.0 / 3.0, 1.0, 1.0);
        assert!(green[0].abs() < 1e-5);
        assert!((green[1] - 1.0).abs() < 1e-5);
        assert!(green[2].abs() < 1e-5);

        let blue = hsv_to_rgb(2.0 / 3.0, 1.0, 1.0);
        assert!(blue[0].abs() < 1e-5);
        assert!(blue[1].abs() < 1e-5);
        assert!((blue[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn unique_edge_palette_separates_adjacent_edges() {
        let edges: &[[u32; 2]] = &[[0, 1], [0, 2], [0, 3]];
        let palette = unique_edge_palette(edges);
        assert_eq!(palette.len(), 3, "one color per edge");
        for i in 0..palette.len() {
            for j in (i + 1)..palette.len() {
                assert_ne!(
                    palette[i], palette[j],
                    "edges {i} and {j} share vertex 0; palette must differ"
                );
            }
        }
    }

    #[test]
    fn w_depth_color_clamps_past_extent() {
        let c_far = w_depth_color(5.0, 1.0);
        let c_at_extent = w_depth_color(1.0, 1.0);
        assert_eq!(c_far, c_at_extent, "+w past extent should clamp to front");

        let c_back = w_depth_color(-5.0, 1.0);
        let c_at_neg_extent = w_depth_color(-1.0, 1.0);
        assert_eq!(
            c_back, c_at_neg_extent,
            "-w past extent should clamp to back"
        );
    }
}
