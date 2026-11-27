//! Simplification only ever grows the ring, so a hull of the collider cover
//! stays a hull of the letter at every side count.

use glam::Vec2;

pub(super) const MAX_HULL_SIDES: usize = 8;

// Andrew's monotone chain (Andrew 1979, Inf. Process. Lett. 9(5)).
pub(super) fn convex_hull(mut points: Vec<Vec2>) -> Vec<Vec2> {
    points.sort_unstable_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    points.dedup();
    if points.len() < 3 {
        return points;
    }

    let mut hull: Vec<Vec2> = Vec::with_capacity(points.len() + 1);
    for &p in &points {
        pop_non_left_turns(&mut hull, p, 2);
        hull.push(p);
    }
    let lower = hull.len() + 1;
    for &p in points.iter().rev().skip(1) {
        pop_non_left_turns(&mut hull, p, lower);
        hull.push(p);
    }
    hull.pop();
    hull
}

fn pop_non_left_turns(hull: &mut Vec<Vec2>, p: Vec2, floor: usize) {
    while hull.len() >= floor {
        let b = hull[hull.len() - 1];
        let a = hull[hull.len() - 2];
        if (b - a).perp_dot(p - a) > 0.0 {
            break;
        }
        hull.pop();
    }
}

pub(super) fn reduce_sides(ring: &mut Vec<Vec2>, sides: usize) {
    debug_assert!(sides >= 4, "a convex ring cannot be reduced below a quad");
    while ring.len() > sides {
        let n = ring.len();
        let mut best: Option<(f32, usize, Vec2)> = None;
        for i in 0..n {
            let a = ring[(i + n - 1) % n];
            let b = ring[i];
            let c = ring[(i + 1) % n];
            let d = ring[(i + 2) % n];
            let Some(x) = extend_to_meet(a, b, c, d) else {
                continue;
            };
            let added = 0.5 * (c - b).perp_dot(x - b).abs();
            if best.is_none_or(|(least, _, _)| added < least) {
                best = Some((added, i, x));
            }
        }
        let Some((_, i, x)) = best else { break };
        ring[i] = x;
        ring.remove((i + 1) % n);
    }
}

fn extend_to_meet(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> Option<Vec2> {
    let ab = b - a;
    let cd = d - c;
    let denom = ab.perp_dot(cd);
    if denom == 0.0 {
        return None;
    }
    let t = (c - a).perp_dot(cd) / denom;
    let s = (c - a).perp_dot(ab) / denom;
    if !(t >= 1.0 && s <= 0.0) {
        return None;
    }
    Some(a + ab * t)
}

pub(super) fn double_area(ring: &[Vec2]) -> f32 {
    let n = ring.len();
    (0..n)
        .map(|i| {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            a.x * b.y - b.x * a.y
        })
        .sum()
}

pub(super) fn centroid(ring: &[Vec2]) -> Vec2 {
    let n = ring.len();
    let mut moment = Vec2::ZERO;
    for i in 0..n {
        let a = ring[i];
        let b = ring[(i + 1) % n];
        moment += (a + b) * (a.x * b.y - b.x * a.y);
    }
    moment / (3.0 * double_area(ring))
}
