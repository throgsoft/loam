use bytemuck::{Pod, Zeroable};

use crate::domain::{DomainError, DomainSpace, Field, FieldKind, FieldProgram, Pose};
use crate::entity::Entity;
use crate::store::{Change, Cursor, Store};

pub const OP_SPHERE: u32 = 1;
pub const OP_BOX: u32 = 2;
pub const OP_HALFSPACE: u32 = 3;
pub const OP_HYPERSPHERE: u32 = 4;
pub const OP_HALFSPACE4: u32 = 5;
pub const OP_UNION: u32 = 6;
pub const OP_INTERSECTION: u32 = 7;
pub const OP_SUBTRACTION: u32 = 8;
pub const OP_SMOOTH_UNION: u32 = 9;
pub const OP_PUSH_POSE: u32 = 10;
pub const OP_POP_POSE: u32 = 11;

pub const MAX_STACK: usize = 32;
pub const MAX_POSE_DEPTH: usize = 8;
pub const MAX_PROGRAM_WORDS: usize = 1 << 20;

pub const FIELD_FAR: f32 = 1e9;

/// The minimum error a compiled program states when read through `DistanceField`.
pub const FIELD_PROGRAM_ERROR: f32 = 1e-4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldOp {
    Sphere { radius: f32 },
    Box { half_extents: [f32; 3] },
    HalfSpace { normal: [f32; 3], offset: f32 },
    HyperSphere { radius: f32 },
    HalfSpace4 { normal: [f32; 4], offset: f32 },
    Union,
    Intersection,
    Subtraction,
    SmoothUnion { radius: f32 },
    Transform,
}

impl FieldOp {
    pub fn operands(self) -> usize {
        match self {
            FieldOp::Sphere { .. }
            | FieldOp::Box { .. }
            | FieldOp::HalfSpace { .. }
            | FieldOp::HyperSphere { .. }
            | FieldOp::HalfSpace4 { .. } => 0,
            FieldOp::Transform => 1,
            FieldOp::Union
            | FieldOp::Intersection
            | FieldOp::Subtraction
            | FieldOp::SmoothUnion { .. } => 2,
        }
    }

    pub fn opcode(self) -> u32 {
        match self {
            FieldOp::Sphere { .. } => OP_SPHERE,
            FieldOp::Box { .. } => OP_BOX,
            FieldOp::HalfSpace { .. } => OP_HALFSPACE,
            FieldOp::HyperSphere { .. } => OP_HYPERSPHERE,
            FieldOp::HalfSpace4 { .. } => OP_HALFSPACE4,
            FieldOp::Union => OP_UNION,
            FieldOp::Intersection => OP_INTERSECTION,
            FieldOp::Subtraction => OP_SUBTRACTION,
            FieldOp::SmoothUnion { .. } => OP_SMOOTH_UNION,
            FieldOp::Transform => OP_PUSH_POSE,
        }
    }

    pub fn result_kind(self, operands: &[FieldKind]) -> FieldKind {
        let weakest = operands
            .iter()
            .copied()
            .fold(FieldKind::ExactDistance, FieldKind::weaker);
        match self {
            FieldOp::Union
            | FieldOp::Intersection
            | FieldOp::Subtraction
            | FieldOp::SmoothUnion { .. } => weakest.weaker(FieldKind::ConservativeBound),
            _ => weakest,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct FieldPrimitive {
    pub frame: [[f32; 4]; 4],
    pub translation: [f32; 4],
    pub params: [f32; 4],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldCounts {
    pub primitive_evals: u64,
    pub instructions: u64,
    pub node_visits: u64,
    pub node_skips: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldError {
    Opcode(u32),
    Truncated,
    StackOverflow,
    StackUnderflow,
    PoseDepth,
    Primitive(u32),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldCost {
    pub changed_inputs: u32,
    pub affected_dependencies: u32,
    pub program_layout: u32,
    pub index_maintenance: u32,
    pub full_rebuild: bool,
}

impl FieldCost {
    pub fn is_idle(self) -> bool {
        self == FieldCost::default()
    }
}

// Moore, Kearfott, and Cloud 2009, Introduction to Interval Analysis, §2.1.
#[derive(Clone, Copy)]
struct FieldInterval {
    lo: f64,
    hi: f64,
}

impl FieldInterval {
    const WHOLE: Self = Self {
        lo: f64::NEG_INFINITY,
        hi: f64::INFINITY,
    };

    fn exact(value: f32) -> Self {
        if value.is_finite() {
            Self {
                lo: value as f64,
                hi: value as f64,
            }
        } else {
            Self::WHOLE
        }
    }

    fn exact_f64(value: f64) -> Self {
        if value.is_finite() {
            Self {
                lo: value,
                hi: value,
            }
        } else {
            Self::WHOLE
        }
    }

    fn rounded(lo: f64, hi: f64) -> Self {
        if lo.is_nan() || hi.is_nan() || lo > hi || !lo.is_finite() || !hi.is_finite() {
            Self::WHOLE
        } else {
            Self {
                lo: lo.next_down(),
                hi: hi.next_up(),
            }
        }
    }

    fn add(self, other: Self) -> Self {
        Self::rounded(self.lo + other.lo, self.hi + other.hi)
    }

    fn sub(self, other: Self) -> Self {
        Self::rounded(self.lo - other.hi, self.hi - other.lo)
    }

    fn mul(self, other: Self) -> Self {
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        if products.iter().any(|value| value.is_nan()) {
            return Self::WHOLE;
        }
        let lo = products.into_iter().fold(f64::INFINITY, f64::min);
        let hi = products.into_iter().fold(f64::NEG_INFINITY, f64::max);
        Self::rounded(lo, hi)
    }

    fn div(self, other: Self) -> Self {
        if other.lo <= 0.0 && other.hi >= 0.0 {
            return Self::WHOLE;
        }
        self.mul(Self::rounded(1.0 / other.hi, 1.0 / other.lo))
    }

    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }

    fn abs(self) -> Self {
        if self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            self.neg()
        } else {
            Self {
                lo: 0.0,
                hi: (-self.lo).max(self.hi),
            }
        }
    }

    fn min(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.min(other.hi),
        }
    }

    fn max(self, other: Self) -> Self {
        Self {
            lo: self.lo.max(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    fn clamp(self, lo: f64, hi: f64) -> Self {
        Self {
            lo: self.lo.clamp(lo, hi),
            hi: self.hi.clamp(lo, hi),
        }
    }

    fn sqrt(self) -> Self {
        if self.hi < 0.0 {
            return Self::WHOLE;
        }
        Self::rounded(self.lo.max(0.0).sqrt(), self.hi.sqrt())
    }
}

#[derive(Clone, Copy)]
struct FrameBounds {
    deviation: f64,
    singular_min: f64,
    singular_max: f64,
    inverse_max: f64,
}

#[derive(Clone, Copy)]
struct MetricBounds {
    min: f64,
    max: f64,
}

impl MetricBounds {
    const IDENTITY: Self = Self { min: 1.0, max: 1.0 };

    fn through(self, frame: [[f32; 4]; 4]) -> Option<Self> {
        let frame = frame_bounds(frame)?;
        let min = FieldInterval::exact_f64(self.min)
            .mul(FieldInterval::exact_f64(frame.singular_min))
            .lo;
        let max = FieldInterval::exact_f64(self.max)
            .mul(FieldInterval::exact_f64(frame.singular_max))
            .hi;
        (min > 0.0 && min.is_finite() && max.is_finite()).then_some(Self { min, max })
    }

    fn scaled(self, scale: FieldInterval) -> Option<Self> {
        let min = FieldInterval::exact_f64(self.min).mul(scale).lo;
        let max = FieldInterval::exact_f64(self.max).mul(scale).hi;
        (min > 0.0 && min.is_finite() && max.is_finite()).then_some(Self { min, max })
    }
}

fn upper_f32(value: f64) -> f32 {
    if value.is_nan() || value > f32::MAX as f64 {
        return f32::INFINITY;
    }
    if value < -(f32::MAX as f64) {
        return -f32::MAX;
    }
    let rounded = value as f32;
    if rounded as f64 >= value {
        rounded
    } else {
        rounded.next_up()
    }
}

fn lower_f32(value: f64) -> f32 {
    if value.is_nan() || value < -(f32::MAX as f64) {
        return f32::NEG_INFINITY;
    }
    if value > f32::MAX as f64 {
        return f32::MAX;
    }
    let rounded = value as f32;
    if rounded as f64 <= value {
        rounded
    } else {
        rounded.next_down()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct FieldNode {
    pub center: [f32; 4],
    pub radius: f32,
    pub scale: f32,
    pub start: u32,
    pub end: u32,
    pub escape: u32,
}

impl FieldNode {
    pub fn is_bounded(self) -> bool {
        self.radius >= 0.0 && self.scale > 0.0
    }

    pub fn is_leaf(self) -> bool {
        self.end != 0
    }

    pub fn lower_bound(self, point: [f32; 4]) -> f32 {
        let mut sum = FieldInterval::exact(0.0);
        for (at, center) in point.iter().zip(self.center) {
            let d = FieldInterval::exact(*at).sub(FieldInterval::exact(center));
            sum = sum.add(d.mul(d));
        }
        let gap = sum.sqrt().sub(FieldInterval::exact(self.radius));
        if gap.lo <= 0.0 {
            f32::NEG_INFINITY
        } else {
            lower_f32(FieldInterval::exact(self.scale).mul(gap).lo)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Ball {
    center: [f32; 4],
    radius: f32,
    scale: f32,
}

const UNBOUNDED: Ball = Ball {
    center: [0.0; 4],
    radius: -1.0,
    scale: 0.0,
};

impl Ball {
    fn bounded(self) -> bool {
        self.radius >= 0.0 && self.scale > 0.0
    }

    fn expanded(self, by: f32) -> Self {
        if !self.bounded() {
            return UNBOUNDED;
        }
        let expansion = FieldInterval::exact(by.abs()).div(FieldInterval::exact(self.scale));
        let radius = upper_f32(FieldInterval::exact(self.radius).add(expansion).hi);
        if !radius.is_finite() {
            return UNBOUNDED;
        }
        Ball {
            center: self.center,
            radius,
            scale: self.scale,
        }
    }

    fn smaller(self, other: Self) -> Self {
        match (self.bounded(), other.bounded()) {
            (true, true) if other.radius < self.radius => other,
            (true, _) => self,
            (false, true) => other,
            _ => UNBOUNDED,
        }
    }

    fn enclosing(self, other: Self) -> Self {
        if !self.bounded() || !other.bounded() {
            return UNBOUNDED;
        }
        let mut center = [0.0; 4];
        for (axis, component) in center.iter_mut().enumerate() {
            *component = ((self.center[axis] as f64 + other.center[axis] as f64) * 0.5) as f32;
        }
        let radius_for = |ball: Self| {
            ball.center
                .into_iter()
                .zip(center)
                .fold(FieldInterval::exact(0.0), |sum, (from, to)| {
                    let delta = FieldInterval::exact(from).sub(FieldInterval::exact(to));
                    sum.add(delta.mul(delta))
                })
                .sqrt()
                .add(FieldInterval::exact(ball.radius))
        };
        let radius = upper_f32(radius_for(self).max(radius_for(other)).hi);
        if !radius.is_finite() || center.iter().any(|component| !component.is_finite()) {
            UNBOUNDED
        } else {
            Ball {
                center,
                radius,
                scale: self.scale.min(other.scale),
            }
        }
    }

    fn through(self, prim: &FieldPrimitive) -> Self {
        let Some(frame) = frame_bounds(prim.frame) else {
            return UNBOUNDED;
        };
        if !self.bounded() {
            return UNBOUNDED;
        }
        let mut center = [0.0; 4];
        let mut center_error_sq = FieldInterval::exact(0.0);
        for (axis, component) in center.iter_mut().enumerate() {
            let mut exact = FieldInterval::exact(prim.translation[axis]);
            for (component, row) in self.center.into_iter().zip(prim.frame) {
                exact =
                    exact.add(FieldInterval::exact(row[axis]).mul(FieldInterval::exact(component)));
            }
            *component = ((exact.lo + exact.hi) * 0.5) as f32;
            let error = FieldInterval::exact(*component).sub(exact).abs();
            center_error_sq = center_error_sq.add(error.mul(error));
        }
        let center_norm = self
            .center
            .into_iter()
            .fold(FieldInterval::exact(0.0), |sum, component| {
                let component = FieldInterval::exact(component);
                sum.add(component.mul(component))
            })
            .sqrt();
        let inverse_norm = FieldInterval::exact_f64(frame.inverse_max);
        let delta = FieldInterval::exact_f64(frame.deviation);
        let radius = inverse_norm
            .mul(FieldInterval::exact(self.radius))
            .add(inverse_norm.mul(delta).mul(center_norm))
            .add(center_error_sq.sqrt());
        let radius = upper_f32(radius.hi);
        let scale = lower_f32(
            FieldInterval::exact(self.scale)
                .mul(FieldInterval::exact_f64(frame.singular_min))
                .lo,
        );
        if !radius.is_finite()
            || !scale.is_finite()
            || scale <= 0.0
            || center.iter().any(|component| !component.is_finite())
        {
            UNBOUNDED
        } else {
            Ball {
                center,
                radius,
                scale,
            }
        }
    }
}

// Horn and Johnson 2013, Matrix Analysis, §§5.6, 7.3.
fn frame_bounds(frame: [[f32; 4]; 4]) -> Option<FrameBounds> {
    let mut delta = 0.0f64;
    for (left_index, left_column) in frame.iter().enumerate() {
        let mut row_sum = FieldInterval::exact(0.0);
        for (right_index, right_column) in frame.iter().enumerate() {
            let mut gram = FieldInterval::exact(0.0);
            for (&left, &right) in left_column.iter().zip(right_column) {
                gram = gram.add(FieldInterval::exact(left).mul(FieldInterval::exact(right)));
            }
            row_sum = row_sum.add(
                FieldInterval::exact(if left_index == right_index { 1.0 } else { 0.0 })
                    .sub(gram)
                    .abs(),
            );
        }
        delta = delta.max(row_sum.hi);
    }
    if !delta.is_finite() || delta >= 1.0 {
        return None;
    }
    let remaining = FieldInterval::exact_f64(1.0)
        .sub(FieldInterval::exact_f64(delta))
        .sqrt();
    let inverse_norm = FieldInterval::exact_f64(1.0).div(remaining).hi;
    let singular_min = remaining.lo;
    let singular_max = FieldInterval::exact_f64(1.0)
        .add(FieldInterval::exact_f64(delta))
        .sqrt()
        .hi;
    (inverse_norm.is_finite() && singular_min > 0.0 && singular_max.is_finite()).then_some(
        FrameBounds {
            deviation: delta,
            singular_min,
            singular_max,
            inverse_max: inverse_norm,
        },
    )
}

fn primitive_ball(op: u32, prim: &FieldPrimitive, extruded_w: bool) -> Ball {
    let r = prim.params;
    let local = match op {
        OP_HYPERSPHERE => Ball {
            center: [0.0; 4],
            radius: r[0].abs(),
            scale: 1.0,
        },
        OP_SPHERE if !extruded_w => Ball {
            center: [0.0; 4],
            radius: r[0].abs(),
            scale: 1.0,
        },
        OP_BOX if !extruded_w => Ball {
            center: [0.0; 4],
            radius: upper_f32(
                r[..3]
                    .iter()
                    .fold(FieldInterval::exact(0.0), |sum, &component| {
                        let component = FieldInterval::exact(component);
                        sum.add(component.mul(component))
                    })
                    .sqrt()
                    .hi,
            ),
            scale: 1.0,
        },
        _ => UNBOUNDED,
    };
    local.through(prim)
}

fn subtree_ball(primitives: &[FieldPrimitive], range: &[u32], extruded_w: bool) -> Ball {
    if !range.len().is_multiple_of(2) {
        return UNBOUNDED;
    }
    let mut stack = [UNBOUNDED; MAX_STACK];
    let mut poses = [0u32; MAX_POSE_DEPTH];
    let mut sp = 0usize;
    let mut pp = 0usize;
    for word in range.chunks_exact(2) {
        let (op, arg) = (word[0], word[1]);
        match op {
            OP_SPHERE | OP_BOX | OP_HALFSPACE | OP_HYPERSPHERE | OP_HALFSPACE4 => {
                let Some(prim) = primitives.get(arg as usize) else {
                    return UNBOUNDED;
                };
                if sp == MAX_STACK {
                    return UNBOUNDED;
                }
                stack[sp] = primitive_ball(op, prim, extruded_w);
                sp += 1;
            }
            OP_UNION | OP_INTERSECTION | OP_SUBTRACTION | OP_SMOOTH_UNION => {
                if sp < 2 {
                    return UNBOUNDED;
                }
                let b = stack[sp - 1];
                let a = stack[sp - 2];
                sp -= 1;
                stack[sp - 1] = match op {
                    OP_UNION => a.enclosing(b),
                    OP_INTERSECTION => a.smaller(b),
                    OP_SUBTRACTION => a,
                    _ => a.enclosing(b).expanded(f32::from_bits(arg)),
                };
            }
            OP_PUSH_POSE => {
                if pp == MAX_POSE_DEPTH {
                    return UNBOUNDED;
                }
                poses[pp] = arg;
                pp += 1;
            }
            OP_POP_POSE => {
                if pp == 0 || sp == 0 {
                    return UNBOUNDED;
                }
                pp -= 1;
                let Some(prim) = primitives.get(poses[pp] as usize) else {
                    return UNBOUNDED;
                };
                stack[sp - 1] = stack[sp - 1].through(prim);
            }
            _ => return UNBOUNDED,
        }
    }
    if sp == 1 {
        stack[0]
    } else {
        UNBOUNDED
    }
}

#[derive(Clone, Copy, Debug)]
struct Cut {
    node: u32,
    start: u32,
    end: u32,
    implicit: bool,
    ball: Ball,
}

fn widest_axis(leaves: &[Cut]) -> usize {
    let mut widest = 0;
    let mut spread = -1.0f32;
    for axis in 0..4 {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for leaf in leaves {
            lo = lo.min(leaf.ball.center[axis]);
            hi = hi.max(leaf.ball.center[axis]);
        }
        if hi - lo > spread {
            spread = hi - lo;
            widest = axis;
        }
    }
    widest
}

fn build_tree(leaves: &mut [Cut], out: &mut Vec<FieldNode>) {
    if leaves.len() == 1 {
        let leaf = leaves[0];
        out.push(FieldNode {
            center: leaf.ball.center,
            radius: leaf.ball.radius,
            scale: leaf.ball.scale,
            start: leaf.start,
            end: leaf.end,
            escape: 0,
        });
        let at = out.len() - 1;
        out[at].escape = out.len() as u32;
        return;
    }
    let at = out.len();
    out.push(FieldNode::default());
    let axis = widest_axis(leaves);
    let mid = leaves.len() / 2;
    leaves.select_nth_unstable_by(mid, |a, b| {
        a.ball.center[axis]
            .total_cmp(&b.ball.center[axis])
            .then(a.start.cmp(&b.start))
    });
    let (left, right) = leaves.split_at_mut(mid);
    build_tree(left, out);
    let right_at = out[at + 1].escape as usize;
    build_tree(right, out);
    let ball = ball_of(out[at + 1]).enclosing(ball_of(out[right_at]));
    out[at] = FieldNode {
        center: ball.center,
        radius: ball.radius,
        scale: ball.scale,
        start: 0,
        end: 0,
        escape: out.len() as u32,
    };
}

fn ball_of(node: FieldNode) -> Ball {
    Ball {
        center: node.center,
        radius: node.radius,
        scale: node.scale,
    }
}

fn local(prim: &FieldPrimitive, p: [f32; 4]) -> [f32; 4] {
    let d = [
        p[0] - prim.translation[0],
        p[1] - prim.translation[1],
        p[2] - prim.translation[2],
        p[3] - prim.translation[3],
    ];
    let mut out = [0.0; 4];
    for (slot, column) in out.iter_mut().zip(prim.frame) {
        *slot = column[0] * d[0] + column[1] * d[1] + column[2] * d[2] + column[3] * d[3];
    }
    out
}

fn local_interval(prim: &FieldPrimitive, p: [FieldInterval; 4]) -> [FieldInterval; 4] {
    let d = [
        p[0].sub(FieldInterval::exact(prim.translation[0])),
        p[1].sub(FieldInterval::exact(prim.translation[1])),
        p[2].sub(FieldInterval::exact(prim.translation[2])),
        p[3].sub(FieldInterval::exact(prim.translation[3])),
    ];
    let mut out = [FieldInterval::exact(0.0); 4];
    for (slot, column) in out.iter_mut().zip(prim.frame) {
        *slot = FieldInterval::exact(column[0])
            .mul(d[0])
            .add(FieldInterval::exact(column[1]).mul(d[1]))
            .add(FieldInterval::exact(column[2]).mul(d[2]))
            .add(FieldInterval::exact(column[3]).mul(d[3]));
    }
    out
}

fn primitive_distance(op: u32, prim: &FieldPrimitive, p: [f32; 4]) -> f32 {
    let q = local(prim, p);
    let r = prim.params;
    match op {
        OP_SPHERE => (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt() - r[0],
        OP_BOX => {
            let d = [q[0].abs() - r[0], q[1].abs() - r[1], q[2].abs() - r[2]];
            let outside =
                (d[0].max(0.0).powi(2) + d[1].max(0.0).powi(2) + d[2].max(0.0).powi(2)).sqrt();
            outside + d[0].max(d[1]).max(d[2]).min(0.0)
        }
        OP_HALFSPACE => q[0] * r[0] + q[1] * r[1] + q[2] * r[2],
        OP_HYPERSPHERE => (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt() - r[0],
        OP_HALFSPACE4 => q[0] * r[0] + q[1] * r[1] + q[2] * r[2] + q[3] * r[3],
        _ => FIELD_FAR,
    }
}

fn primitive_interval(op: u32, prim: &FieldPrimitive, p: [FieldInterval; 4]) -> FieldInterval {
    let q = local_interval(prim, p);
    let r = prim.params.map(FieldInterval::exact);
    match op {
        OP_SPHERE => q[..3]
            .iter()
            .fold(FieldInterval::exact(0.0), |sum, &component| {
                sum.add(component.mul(component))
            })
            .sqrt()
            .sub(r[0]),
        OP_BOX => {
            let d = [
                q[0].abs().sub(r[0]),
                q[1].abs().sub(r[1]),
                q[2].abs().sub(r[2]),
            ];
            let zero = FieldInterval::exact(0.0);
            let outside = d
                .iter()
                .fold(zero, |sum, &component| {
                    let positive = component.max(zero);
                    sum.add(positive.mul(positive))
                })
                .sqrt();
            outside.add(d[0].max(d[1]).max(d[2]).min(zero))
        }
        OP_HALFSPACE => q[0].mul(r[0]).add(q[1].mul(r[1])).add(q[2].mul(r[2])),
        OP_HYPERSPHERE => q
            .iter()
            .fold(FieldInterval::exact(0.0), |sum, &component| {
                sum.add(component.mul(component))
            })
            .sqrt()
            .sub(r[0]),
        OP_HALFSPACE4 => q[0]
            .mul(r[0])
            .add(q[1].mul(r[1]))
            .add(q[2].mul(r[2]))
            .add(q[3].mul(r[3])),
        _ => FieldInterval::exact(FIELD_FAR),
    }
}

fn primitive_metric_interval(
    op: u32,
    prim: &FieldPrimitive,
    p: [FieldInterval; 4],
    metric: MetricBounds,
) -> FieldInterval {
    let value = primitive_interval(op, prim, p);
    let Some(mut metric) = metric.through(prim.frame) else {
        return FieldInterval::WHOLE;
    };
    if matches!(op, OP_HALFSPACE | OP_HALFSPACE4) {
        let dimensions = if op == OP_HALFSPACE { 3 } else { 4 };
        let normal = prim.params[..dimensions]
            .iter()
            .fold(FieldInterval::exact(0.0), |sum, &component| {
                let component = FieldInterval::exact(component);
                sum.add(component.mul(component))
            })
            .sqrt();
        let Some(scaled) = metric.scaled(normal) else {
            return FieldInterval::WHOLE;
        };
        metric = scaled;
    }
    value.div(FieldInterval {
        lo: metric.min,
        hi: metric.max,
    })
}

// Quilez, "Smooth minimum", iquilezles.org/articles/smin, polynomial form.
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    (b * (1.0 - h) + a * h) - k * h * (1.0 - h)
}

fn smooth_min_interval(a: FieldInterval, b: FieldInterval, k: FieldInterval) -> FieldInterval {
    let one = FieldInterval::exact(1.0);
    let half = FieldInterval::exact(0.5);
    let h = half.add(half.mul(b.sub(a)).div(k)).clamp(0.0, 1.0);
    b.mul(one.sub(h))
        .add(a.mul(h))
        .sub(k.mul(h).mul(one.sub(h)))
}

fn run<const COUNT: bool>(
    program: &[u32],
    primitives: &[FieldPrimitive],
    point: [f32; 4],
    counts: &mut FieldCounts,
) -> Result<f32, FieldError> {
    if program.is_empty() {
        return Ok(FIELD_FAR);
    }
    if !program.len().is_multiple_of(2) {
        return Err(FieldError::Truncated);
    }
    let mut stack = [0.0f32; MAX_STACK];
    let mut points = [[0.0f32; 4]; MAX_POSE_DEPTH];
    let mut sp = 0usize;
    let mut pp = 0usize;
    let mut p = point;
    for word in program.chunks_exact(2) {
        let (op, arg) = (word[0], word[1]);
        if COUNT {
            counts.instructions += 1;
        }
        match op {
            OP_SPHERE | OP_BOX | OP_HALFSPACE | OP_HYPERSPHERE | OP_HALFSPACE4 => {
                let prim = primitives
                    .get(arg as usize)
                    .ok_or(FieldError::Primitive(arg))?;
                if COUNT {
                    counts.primitive_evals += 1;
                }
                if sp == MAX_STACK {
                    return Err(FieldError::StackOverflow);
                }
                stack[sp] = primitive_distance(op, prim, p);
                sp += 1;
            }
            OP_UNION | OP_INTERSECTION | OP_SUBTRACTION | OP_SMOOTH_UNION => {
                if sp < 2 {
                    return Err(FieldError::StackUnderflow);
                }
                let b = stack[sp - 1];
                let a = stack[sp - 2];
                sp -= 1;
                stack[sp - 1] = match op {
                    OP_UNION => a.min(b),
                    OP_INTERSECTION => a.max(b),
                    OP_SUBTRACTION => a.max(-b),
                    _ => smooth_min(a, b, f32::from_bits(arg)),
                };
            }
            OP_PUSH_POSE => {
                let prim = primitives
                    .get(arg as usize)
                    .ok_or(FieldError::Primitive(arg))?;
                if pp == MAX_POSE_DEPTH {
                    return Err(FieldError::PoseDepth);
                }
                points[pp] = p;
                pp += 1;
                p = local(prim, p);
            }
            OP_POP_POSE => {
                if pp == 0 {
                    return Err(FieldError::PoseDepth);
                }
                pp -= 1;
                p = points[pp];
            }
            other => return Err(FieldError::Opcode(other)),
        }
    }
    if sp != 1 {
        return Err(FieldError::StackUnderflow);
    }
    Ok(stack[0])
}

fn run_interval(
    program: &[u32],
    primitives: &[FieldPrimitive],
    point: [f32; 4],
    exact_distance: bool,
) -> Result<FieldInterval, FieldError> {
    if program.is_empty() {
        return Ok(FieldInterval::exact(FIELD_FAR));
    }
    if !program.len().is_multiple_of(2) {
        return Err(FieldError::Truncated);
    }
    let mut stack = [FieldInterval::exact(0.0); MAX_STACK];
    let mut points = [[FieldInterval::exact(0.0); 4]; MAX_POSE_DEPTH];
    let mut metrics = [Some(MetricBounds::IDENTITY); MAX_POSE_DEPTH];
    let mut sp = 0usize;
    let mut pp = 0usize;
    let mut p = point.map(FieldInterval::exact);
    let mut metric = Some(MetricBounds::IDENTITY);
    for word in program.chunks_exact(2) {
        let (op, arg) = (word[0], word[1]);
        match op {
            OP_SPHERE | OP_BOX | OP_HALFSPACE | OP_HYPERSPHERE | OP_HALFSPACE4 => {
                let prim = primitives
                    .get(arg as usize)
                    .ok_or(FieldError::Primitive(arg))?;
                if sp == MAX_STACK {
                    return Err(FieldError::StackOverflow);
                }
                stack[sp] = if exact_distance {
                    match metric {
                        Some(metric) => primitive_metric_interval(op, prim, p, metric),
                        None => FieldInterval::WHOLE,
                    }
                } else {
                    primitive_interval(op, prim, p)
                };
                sp += 1;
            }
            OP_UNION | OP_INTERSECTION | OP_SUBTRACTION | OP_SMOOTH_UNION => {
                if sp < 2 {
                    return Err(FieldError::StackUnderflow);
                }
                let b = stack[sp - 1];
                let a = stack[sp - 2];
                sp -= 1;
                stack[sp - 1] = match (exact_distance, op) {
                    (true, _) => FieldInterval::WHOLE,
                    (false, OP_UNION) => a.min(b),
                    (false, OP_INTERSECTION) => a.max(b),
                    (false, OP_SUBTRACTION) => a.max(b.neg()),
                    (false, _) => {
                        smooth_min_interval(a, b, FieldInterval::exact(f32::from_bits(arg)))
                    }
                };
            }
            OP_PUSH_POSE => {
                let prim = primitives
                    .get(arg as usize)
                    .ok_or(FieldError::Primitive(arg))?;
                if pp == MAX_POSE_DEPTH {
                    return Err(FieldError::PoseDepth);
                }
                points[pp] = p;
                metrics[pp] = metric;
                pp += 1;
                p = local_interval(prim, p);
                metric = metric.and_then(|metric| metric.through(prim.frame));
            }
            OP_POP_POSE => {
                if pp == 0 {
                    return Err(FieldError::PoseDepth);
                }
                pp -= 1;
                p = points[pp];
                metric = metrics[pp];
            }
            other => return Err(FieldError::Opcode(other)),
        }
    }
    if sp != 1 {
        return Err(FieldError::StackUnderflow);
    }
    Ok(stack[0])
}

fn forward_error(
    program: &[u32],
    primitives: &[FieldPrimitive],
    point: [f32; 4],
    value: f32,
    kind: FieldKind,
) -> Result<f32, FieldError> {
    let interval = run_interval(program, primitives, point, kind == FieldKind::ExactDistance)?;
    if !value.is_finite() || !interval.lo.is_finite() || !interval.hi.is_finite() {
        return Ok(f32::INFINITY);
    }
    let error = FieldInterval::exact(value).sub(interval).abs().hi;
    Ok(upper_f32(error).max(FIELD_PROGRAM_ERROR))
}

fn traverse<const COUNT: bool>(
    program: &[u32],
    primitives: &[FieldPrimitive],
    nodes: &[FieldNode],
    point: [f32; 4],
    tolerance: f32,
    counts: &mut FieldCounts,
) -> Result<f32, FieldError> {
    if nodes.is_empty() {
        return run::<COUNT>(program, primitives, point, counts);
    }
    let mut best = f32::INFINITY;
    let mut index = 0usize;
    while index < nodes.len() {
        let node = nodes[index];
        if COUNT {
            counts.node_visits += 1;
        }
        if node.is_bounded() && node.lower_bound(point) > best + tolerance {
            if COUNT {
                counts.node_skips += 1;
            }
            index = node.escape as usize;
            continue;
        }
        if node.is_leaf() {
            let range = program
                .get(node.start as usize..node.end as usize)
                .ok_or(FieldError::Truncated)?;
            best = best.min(run::<COUNT>(range, primitives, point, counts)?);
        }
        index += 1;
    }
    Ok(best)
}

pub fn evaluate(
    program: &[u32],
    primitives: &[FieldPrimitive],
    point: [f32; 4],
) -> Result<f32, FieldError> {
    run::<false>(program, primitives, point, &mut FieldCounts::default())
}

pub fn evaluate_counted(
    program: &[u32],
    primitives: &[FieldPrimitive],
    point: [f32; 4],
    counts: &mut FieldCounts,
) -> Result<f32, FieldError> {
    run::<true>(program, primitives, point, counts)
}

/// Skips a subtree whose ball's lower bound exceeds the best value so far by more than `tolerance`; an empty `nodes` slice evaluates the whole program.
pub fn evaluate_bounded(
    program: &[u32],
    primitives: &[FieldPrimitive],
    nodes: &[FieldNode],
    point: [f32; 4],
    tolerance: f32,
) -> Result<f32, FieldError> {
    traverse::<false>(
        program,
        primitives,
        nodes,
        point,
        tolerance,
        &mut FieldCounts::default(),
    )
}

pub fn evaluate_bounded_counted(
    program: &[u32],
    primitives: &[FieldPrimitive],
    nodes: &[FieldNode],
    point: [f32; 4],
    tolerance: f32,
    counts: &mut FieldCounts,
) -> Result<f32, FieldError> {
    traverse::<true>(program, primitives, nodes, point, tolerance, counts)
}

pub(crate) fn evaluate_error(
    program: &[u32],
    primitives: &[FieldPrimitive],
    nodes: &[FieldNode],
    point: [f32; 4],
    kind: FieldKind,
) -> Result<f32, FieldError> {
    let value = traverse::<false>(
        program,
        primitives,
        nodes,
        point,
        0.0,
        &mut FieldCounts::default(),
    )?;
    forward_error(program, primitives, point, value, kind)
}

const NONE: u32 = u32::MAX;
const MAX_OPERANDS: usize = 2;

#[derive(Clone, Copy)]
struct Node {
    entity: Entity,
    op: FieldOp,
    declared: FieldKind,
    first_parent: u32,
    edge_start: u32,
    edge_len: u32,
    record: u32,
    marked: u32,
    pending: u32,
}

#[derive(Clone, Copy)]
struct Frame {
    node: u32,
    cursor: u32,
    start: u32,
    cullable: bool,
}

#[derive(Clone, Copy)]
struct Edge {
    operand: u32,
    operator: u32,
    next: u32,
    live: bool,
}

const EMPTY_EDGE: Edge = Edge {
    operand: NONE,
    operator: NONE,
    next: NONE,
    live: false,
};

pub(crate) struct FieldCompiler {
    program: FieldProgram,
    staged: FieldProgram,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    slots: Vec<u32>,
    roots: Vec<u32>,
    dfs: Vec<Frame>,
    cuts: Vec<Cut>,
    ordered: Vec<Cut>,
    kinds: Vec<FieldKind>,
    work: Vec<u32>,
    touched: Vec<u32>,
    patches: Vec<(usize, FieldPrimitive)>,
    fields_cursor: Cursor,
    poses_cursor: Cursor,
    dirty_stamp: u32,
    compiled: bool,
}

impl Default for FieldCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldCompiler {
    pub(crate) fn new() -> Self {
        Self {
            program: FieldProgram::default(),
            staged: FieldProgram::default(),
            nodes: Vec::new(),
            edges: Vec::new(),
            slots: Vec::new(),
            roots: Vec::new(),
            dfs: Vec::new(),
            cuts: Vec::new(),
            ordered: Vec::new(),
            kinds: Vec::new(),
            work: Vec::new(),
            touched: Vec::new(),
            patches: Vec::new(),
            fields_cursor: Cursor::default(),
            poses_cursor: Cursor::default(),
            dirty_stamp: 0,
            compiled: false,
        }
    }

    pub(crate) fn program(&self) -> &FieldProgram {
        &self.program
    }

    pub(crate) fn invalidate(&mut self) {
        self.compiled = false;
        self.fields_cursor = Cursor::default();
        self.poses_cursor = Cursor::default();
    }

    fn node_of(&self, entity: Entity) -> Option<u32> {
        let node = *self.slots.get(entity.key().slot() as usize)?;
        (node != NONE && self.nodes[node as usize].entity == entity).then_some(node)
    }

    fn set_slot(&mut self, entity: Entity, node: u32) {
        let slot = entity.key().slot() as usize;
        if slot >= self.slots.len() {
            self.slots.resize(slot + 1, NONE);
        }
        self.slots[slot] = node;
    }

    fn has_live_parent(&self, node: u32) -> bool {
        self.nodes[node as usize].first_parent != NONE
    }

    fn link(&mut self, operator: u32, operands: &[Entity]) -> Result<(), DomainError> {
        let entity = self.nodes[operator as usize].entity;
        if operands.len() != self.nodes[operator as usize].op.operands() {
            return Err(DomainError::FieldArity(entity));
        }
        let edge_start = operator as usize * MAX_OPERANDS;
        self.nodes[operator as usize].edge_start = edge_start as u32;
        self.nodes[operator as usize].edge_len = operands.len() as u32;
        for (offset, &operand) in operands.iter().enumerate() {
            let target = self.node_of(operand).ok_or(DomainError::Stale(operand))?;
            let edge = (edge_start + offset) as u32;
            self.edges[edge as usize] = Edge {
                operand: target,
                operator,
                next: self.nodes[target as usize].first_parent,
                live: true,
            };
            self.nodes[target as usize].first_parent = edge;
        }
        Ok(())
    }

    fn detach_parent(&mut self, operand: u32, remove: u32) {
        let mut previous = NONE;
        let mut edge = self.nodes[operand as usize].first_parent;
        while edge != NONE {
            let next = self.edges[edge as usize].next;
            if edge == remove {
                if previous == NONE {
                    self.nodes[operand as usize].first_parent = next;
                } else {
                    self.edges[previous as usize].next = next;
                }
                return;
            }
            previous = edge;
            edge = next;
        }
    }

    fn unlink(&mut self, operator: u32) {
        let node = self.nodes[operator as usize];
        for edge in node.edge_start..node.edge_start + node.edge_len {
            let operand = self.edges[edge as usize].operand;
            self.detach_parent(operand, edge);
            self.edges[edge as usize] = EMPTY_EDGE;
        }
        self.nodes[operator as usize].edge_len = 0;
    }

    fn operands_match(&self, operator: u32, operands: &[Entity]) -> bool {
        let node = self.nodes[operator as usize];
        if node.edge_len as usize != operands.len() {
            return false;
        }
        (0..node.edge_len).all(|i| {
            let edge = self.edges[(node.edge_start + i) as usize];
            edge.live && self.nodes[edge.operand as usize].entity == operands[i as usize]
        })
    }

    fn rebuild_index(&mut self, fields: &Store<Field>) -> Result<u32, DomainError> {
        self.nodes.clear();
        self.edges.clear();
        self.slots.clear();
        for (entity, field) in fields.iter() {
            validate_op(field.op)?;
            let node = self.nodes.len() as u32;
            self.nodes.push(Node {
                entity,
                op: field.op,
                declared: field.kind,
                first_parent: NONE,
                edge_start: NONE,
                edge_len: 0,
                record: NONE,
                marked: 0,
                pending: 0,
            });
            self.set_slot(entity, node);
        }
        self.edges
            .resize(self.nodes.len() * MAX_OPERANDS, EMPTY_EDGE);
        let mut work = 0;
        for index in 0..self.nodes.len() as u32 {
            let entity = self.nodes[index as usize].entity;
            let field = fields.get(entity).ok_or(DomainError::Stale(entity))?;
            work += field.operands.len() as u32;
            self.link(index, &field.operands)?;
        }
        Ok(work)
    }

    fn mark_ancestors(&mut self) -> u32 {
        self.dirty_stamp = match self.dirty_stamp.checked_add(1) {
            Some(stamp) => stamp,
            None => {
                for node in &mut self.nodes {
                    node.marked = 0;
                }
                1
            }
        };
        let stamp = self.dirty_stamp;
        self.work.clear();
        for i in 0..self.touched.len() {
            let node = self.touched[i];
            if self.nodes[node as usize].marked != stamp {
                self.nodes[node as usize].marked = stamp;
                self.work.push(node);
            }
        }
        let mut affected = 0;
        let mut read = 0;
        while read < self.work.len() {
            let node = self.work[read];
            read += 1;
            let mut edge = self.nodes[node as usize].first_parent;
            while edge != NONE {
                let e = self.edges[edge as usize];
                if e.live && self.nodes[e.operator as usize].marked != stamp {
                    self.nodes[e.operator as usize].marked = stamp;
                    affected += 1;
                    self.work.push(e.operator);
                }
                edge = e.next;
            }
        }
        affected
    }

    fn record_for<S: DomainSpace>(
        &mut self,
        node: u32,
        space: &S,
        poses: &Store<Pose<S>>,
    ) -> Result<u32, DomainError> {
        if self.nodes[node as usize].record != NONE {
            return Ok(self.nodes[node as usize].record);
        }
        let (entity, op) = {
            let node = self.nodes[node as usize];
            (node.entity, node.op)
        };
        let record = self.program.primitives.len() as u32;
        self.program
            .primitives
            .push(primitive_record(space, poses, entity, op)?);
        self.nodes[node as usize].record = record;
        Ok(record)
    }

    fn patch<S: DomainSpace>(
        &mut self,
        space: &S,
        poses: &Store<Pose<S>>,
    ) -> Result<u32, DomainError> {
        self.patches.clear();
        for i in 0..self.touched.len() {
            let node = self.nodes[self.touched[i] as usize];
            if node.record == NONE {
                continue;
            }
            self.patches.push((
                node.record as usize,
                primitive_record(space, poses, node.entity, node.op)?,
            ));
        }
        for &(record, primitive) in &self.patches {
            self.program.primitives[record] = primitive;
        }
        Ok(self.patches.len() as u32)
    }

    fn emit<S: DomainSpace>(
        &mut self,
        space: &S,
        poses: &Store<Pose<S>>,
    ) -> Result<u32, DomainError> {
        std::mem::swap(&mut self.program, &mut self.staged);
        let result = self.emit_staged(space, poses);
        if result.is_err() {
            std::mem::swap(&mut self.program, &mut self.staged);
        }
        result
    }

    fn emit_staged<S: DomainSpace>(
        &mut self,
        space: &S,
        poses: &Store<Pose<S>>,
    ) -> Result<u32, DomainError> {
        self.program.program.clear();
        self.program.primitives.clear();
        for node in &mut self.nodes {
            node.record = NONE;
            node.pending = node.edge_len;
        }
        self.check_acyclic()?;
        let mut roots = std::mem::take(&mut self.roots);
        roots.clear();
        roots.extend((0..self.nodes.len() as u32).filter(|&node| !self.has_live_parent(node)));
        roots.sort_unstable_by_key(|&node| self.nodes[node as usize].entity);
        self.roots = roots;

        self.kinds.clear();
        self.dfs.clear();
        self.cuts.clear();
        let mut height = 0u32;
        let mut peak = 0u32;
        let mut pose_depth = 0usize;
        for index in 0..self.roots.len() {
            self.dfs.push(Frame {
                node: self.roots[index],
                cursor: 0,
                start: self.program.program.len() as u32,
                cullable: true,
            });
            while !self.dfs.is_empty() {
                let top = self.dfs.len() - 1;
                let frame = self.dfs[top];
                let (node, cursor) = (frame.node, frame.cursor);
                let op = self.nodes[node as usize].op;
                if cursor == 0 && matches!(op, FieldOp::Transform) {
                    let record = self.record_for(node, space, poses)?;
                    self.push_instruction(OP_PUSH_POSE, record)?;
                    pose_depth += 1;
                    if pose_depth > MAX_POSE_DEPTH {
                        return Err(DomainError::Unsupported("field pose depth"));
                    }
                }
                if (cursor as usize) < op.operands() {
                    self.dfs[top].cursor = cursor + 1;
                    let start = self.nodes[node as usize].edge_start;
                    let child = self.edges[(start + cursor) as usize].operand;
                    self.dfs.push(Frame {
                        node: child,
                        cursor: 0,
                        start: self.program.program.len() as u32,
                        cullable: frame.cullable && matches!(op, FieldOp::Union),
                    });
                    continue;
                }
                match op {
                    FieldOp::Transform => {
                        self.push_instruction(OP_POP_POSE, 0)?;
                        pose_depth -= 1;
                        let operand = self.kinds.pop().unwrap_or(FieldKind::ExactDistance);
                        self.kinds.push(op.result_kind(&[operand]));
                    }
                    FieldOp::Union
                    | FieldOp::Intersection
                    | FieldOp::Subtraction
                    | FieldOp::SmoothUnion { .. } => {
                        let argument = match op {
                            FieldOp::SmoothUnion { radius } => radius.to_bits(),
                            _ => 0,
                        };
                        self.push_instruction(op.opcode(), argument)?;
                        let right = self.kinds.pop().unwrap_or(FieldKind::ExactDistance);
                        let left = self.kinds.pop().unwrap_or(FieldKind::ExactDistance);
                        self.kinds.push(op.result_kind(&[left, right]));
                        height -= 1;
                    }
                    _ => {
                        let record = self.record_for(node, space, poses)?;
                        self.push_instruction(op.opcode(), record)?;
                        self.kinds.push(self.nodes[node as usize].declared);
                        height += 1;
                        peak = peak.max(height);
                    }
                }
                if frame.cullable && !matches!(op, FieldOp::Union) {
                    self.cuts.push(Cut {
                        node,
                        start: frame.start,
                        end: self.program.program.len() as u32,
                        implicit: self.kinds.last().copied() == Some(FieldKind::Implicit),
                        ball: UNBOUNDED,
                    });
                }
                self.dfs.pop();
            }
            if index > 0 {
                self.push_instruction(OP_UNION, 0)?;
                let right = self.kinds.pop().unwrap_or(FieldKind::ExactDistance);
                let left = self.kinds.pop().unwrap_or(FieldKind::ExactDistance);
                self.kinds.push(FieldOp::Union.result_kind(&[left, right]));
                height -= 1;
            }
        }
        if peak as usize > MAX_STACK {
            return Err(DomainError::Unsupported("field stack depth"));
        }
        self.program.stack = peak;
        self.program.kind = self.kinds.pop().unwrap_or(FieldKind::ConservativeBound);
        Ok((self.program.program.len() / 2) as u32 + self.program.primitives.len() as u32)
    }

    fn rebuild_bounds(&mut self, extruded_w: bool, changed_only: bool) -> u32 {
        for cut in &mut self.cuts {
            if changed_only && self.nodes[cut.node as usize].marked != self.dirty_stamp {
                continue;
            }
            cut.ball = if cut.implicit {
                UNBOUNDED
            } else {
                match self
                    .program
                    .program
                    .get(cut.start as usize..cut.end as usize)
                {
                    Some(range) => subtree_ball(&self.program.primitives, range, extruded_w),
                    None => UNBOUNDED,
                }
            };
        }
        let mut ordered = std::mem::take(&mut self.ordered);
        ordered.clear();
        ordered.extend(self.cuts.iter().copied().filter(|cut| cut.ball.bounded()));
        let bounded = ordered.len();
        ordered.extend(self.cuts.iter().copied().filter(|cut| !cut.ball.bounded()));
        self.program.nodes.clear();
        if bounded > 0 {
            build_tree(&mut ordered[..bounded], &mut self.program.nodes);
            for cut in &ordered[bounded..] {
                let escape = self.program.nodes.len() as u32 + 1;
                self.program.nodes.push(FieldNode {
                    center: [0.0; 4],
                    radius: UNBOUNDED.radius,
                    scale: UNBOUNDED.scale,
                    start: cut.start,
                    end: cut.end,
                    escape,
                });
            }
        }
        self.ordered = ordered;
        self.program.nodes.len() as u32
    }

    // Kahn 1962, "Topological sorting of large networks", CACM 5(11).
    fn check_acyclic(&mut self) -> Result<(), DomainError> {
        self.work.clear();
        for index in 0..self.nodes.len() as u32 {
            if self.nodes[index as usize].pending == 0 {
                self.work.push(index);
            }
        }
        let mut read = 0;
        while read < self.work.len() {
            let node = self.work[read];
            read += 1;
            let mut edge = self.nodes[node as usize].first_parent;
            while edge != NONE {
                let e = self.edges[edge as usize];
                if e.live {
                    self.nodes[e.operator as usize].pending -= 1;
                    if self.nodes[e.operator as usize].pending == 0 {
                        self.work.push(e.operator);
                    }
                }
                edge = e.next;
            }
        }
        match self.nodes.iter().find(|node| node.pending != 0) {
            Some(node) => Err(DomainError::FieldCycle(node.entity)),
            None => Ok(()),
        }
    }

    fn push_instruction(&mut self, op: u32, argument: u32) -> Result<(), DomainError> {
        if self.program.program.len() + 2 > MAX_PROGRAM_WORDS {
            return Err(DomainError::Unsupported("field program size"));
        }
        self.program.program.push(op);
        self.program.program.push(argument);
        Ok(())
    }

    pub(crate) fn compile<S: DomainSpace>(
        &mut self,
        space: &S,
        fields: &Store<Field>,
        poses: &Store<Pose<S>>,
    ) -> Result<FieldCost, DomainError> {
        if !space.is_chart_flat() {
            self.compiled = false;
            return Err(DomainError::Unsupported("curved field chart"));
        }
        let dimension = space.chart_dimension();
        let structural = fields.changed_since(&mut self.fields_cursor);
        let placed = poses.changed_since(&mut self.poses_cursor);
        if self.compiled && !structural && !placed {
            return Ok(FieldCost::default());
        }
        if dimension < 4
            && (!self.compiled || structural)
            && fields.iter().any(|(_, field)| {
                matches!(
                    field.op,
                    FieldOp::HyperSphere { .. } | FieldOp::HalfSpace4 { .. }
                )
            })
        {
            return Err(DomainError::Unsupported("field primitive dimension"));
        }

        let mut cost = FieldCost::default();
        let mut rebuild = !self.compiled;
        self.compiled = false;
        self.touched.clear();
        if structural {
            let changes = fields.changes(&mut self.fields_cursor);
            rebuild |= changes.is_resync();
            for change in changes {
                match change {
                    Change::Row(entity, _) => match self.node_of(entity) {
                        Some(node) => {
                            self.touched.push(node);
                            cost.changed_inputs += 1;
                        }
                        None => {
                            cost.changed_inputs += 1;
                            rebuild = true;
                        }
                    },
                    Change::Removed(_) => {
                        cost.changed_inputs += 1;
                        rebuild = true;
                    }
                }
            }
        }
        let edited = self.touched.len();
        if placed {
            let changes = poses.changes(&mut self.poses_cursor);
            for change in changes {
                match change {
                    Change::Row(entity, _) => {
                        if let Some(node) = self.node_of(entity) {
                            self.touched.push(node);
                            cost.changed_inputs += 1;
                        }
                    }
                    Change::Removed(removal) => {
                        if let Some(node) = self.node_of(removal.entity) {
                            self.touched.push(node);
                            cost.changed_inputs += 1;
                        }
                    }
                }
            }
        }

        if rebuild {
            cost.index_maintenance = self.rebuild_index(fields)?;
            cost.full_rebuild = true;
            self.touched.clear();
        } else {
            for i in 0..edited {
                let node = self.touched[i];
                let entity = self.nodes[node as usize].entity;
                let field = fields.get(entity).ok_or(DomainError::Stale(entity))?;
                validate_op(field.op)?;
                self.nodes[node as usize].op = field.op;
                self.nodes[node as usize].declared = field.kind;
                if !self.operands_match(node, &field.operands) {
                    cost.index_maintenance += self.nodes[node as usize].edge_len;
                    cost.index_maintenance += field.operands.len() as u32;
                    self.unlink(node);
                    self.link(node, &field.operands)?;
                }
            }
        }

        cost.affected_dependencies = self.mark_ancestors();
        cost.program_layout = if rebuild || edited > 0 {
            self.emit(space, poses)?
        } else {
            self.patch(space, poses)?
        };
        cost.index_maintenance += self.rebuild_bounds(dimension > 3, !rebuild && edited == 0);
        self.program.dimension = dimension;
        self.compiled = true;
        Ok(cost)
    }
}

fn primitive_record<S: DomainSpace>(
    space: &S,
    poses: &Store<Pose<S>>,
    entity: Entity,
    op: FieldOp,
) -> Result<FieldPrimitive, DomainError> {
    validate_op(op)?;
    let pose = poses.get(entity).ok_or(DomainError::Stale(entity))?;
    let chart = space.chart_pose(pose);
    let mut record = FieldPrimitive {
        frame: chart.frame,
        translation: chart.coordinates,
        params: [0.0; 4],
    };
    match op {
        FieldOp::Sphere { radius } | FieldOp::HyperSphere { radius } => record.params[0] = radius,
        FieldOp::Box { half_extents } => {
            record.params[..3].copy_from_slice(&half_extents);
        }
        FieldOp::HalfSpace { normal, offset } => {
            let plane = unit_plane([normal[0], normal[1], normal[2], 0.0], offset)?;
            record.params = plane.0;
            shift_along(&mut record, plane.0, plane.1);
        }
        FieldOp::HalfSpace4 { normal, offset } => {
            let plane = unit_plane(normal, offset)?;
            record.params = plane.0;
            shift_along(&mut record, plane.0, plane.1);
        }
        FieldOp::Transform => {}
        _ => return Err(DomainError::FieldArity(entity)),
    }
    if record
        .frame
        .iter()
        .flatten()
        .chain(&record.translation)
        .chain(&record.params)
        .any(|component| !component.is_finite())
    {
        return Err(DomainError::InvalidCoordinate("field primitive"));
    }
    Ok(record)
}

fn validate_op(op: FieldOp) -> Result<(), DomainError> {
    match op {
        FieldOp::Sphere { radius } | FieldOp::HyperSphere { radius }
            if !radius.is_finite() || radius <= 0.0 =>
        {
            Err(DomainError::InvalidCoordinate("field radius"))
        }
        FieldOp::Box { half_extents }
            if half_extents
                .into_iter()
                .any(|extent| !extent.is_finite() || extent <= 0.0) =>
        {
            Err(DomainError::InvalidCoordinate("field half extent"))
        }
        FieldOp::HalfSpace { offset, .. } | FieldOp::HalfSpace4 { offset, .. }
            if !offset.is_finite() =>
        {
            Err(DomainError::InvalidCoordinate("field offset"))
        }
        FieldOp::SmoothUnion { radius } if !radius.is_finite() || radius <= 0.0 => {
            Err(DomainError::InvalidCoordinate("field smoothing radius"))
        }
        _ => Ok(()),
    }
}

fn unit_plane(normal: [f32; 4], offset: f32) -> Result<([f32; 4], f32), DomainError> {
    let length = normal
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt();
    if !length.is_finite() || length < 1e-6 || !offset.is_finite() {
        return Err(DomainError::InvalidCoordinate("field normal"));
    }
    Ok((normal.map(|component| component / length), offset / length))
}

fn shift_along(record: &mut FieldPrimitive, normal: [f32; 4], offset: f32) {
    for axis in 0..4 {
        let mut sum = 0.0;
        for (component, column) in normal.iter().zip(record.frame) {
            sum += column[axis] * component;
        }
        record.translation[axis] += sum * offset;
    }
}

#[cfg(test)]
mod tests {
    use loam_math::EuclideanR3;
    use loam_time::alloc::bytes_allocated_by;

    use super::*;
    use crate::domain::{Domain, DomainBuilder, DomainId, TypedDomain};
    use crate::entity::{Entities, Entity, Epoch, RuntimeId, SceneId};
    use crate::store::DEFAULT_LOG_CAPACITY;
    use crate::view::Vec3;

    fn test_entities() -> Entities {
        Entities::new(SceneId {
            runtime: RuntimeId::allocate(),
            epoch: Epoch::default(),
        })
    }

    struct Fixture {
        entities: Entities,
        domain: TypedDomain<EuclideanR3>,
    }

    impl Fixture {
        fn new() -> Self {
            let entities = test_entities();
            let domain = DomainBuilder::new("fields", EuclideanR3)
                .tracked(DEFAULT_LOG_CAPACITY)
                .fields()
                .build(DomainId::new(0), entities.scene());
            Self { entities, domain }
        }

        fn spawn(&mut self, at: Vec3, kind: FieldKind, op: FieldOp, operands: &[Entity]) -> Entity {
            let entity = self.entities.spawn();
            self.domain
                .attach_pose(entity, Pose::at(at))
                .expect("pose row");
            self.domain
                .attach_field(
                    entity,
                    Field {
                        kind,
                        op,
                        operands: operands.to_vec(),
                    },
                )
                .expect("field row");
            entity
        }

        fn compile(&mut self) -> Result<FieldCost, DomainError> {
            self.domain.compile_fields()
        }

        fn compile_with(&self, compiler: &mut FieldCompiler) -> Result<FieldCost, DomainError> {
            compiler.compile(
                &EuclideanR3,
                self.domain.fields().expect("field store"),
                self.domain.poses(),
            )
        }

        fn move_to(&mut self, entity: Entity, to: Vec3) {
            self.domain.set_point(entity, to).expect("pose row");
        }
    }

    fn sphere(radius: f32) -> FieldOp {
        FieldOp::Sphere { radius }
    }

    #[test]
    fn a_missing_operand_is_refused_and_names_the_operand() {
        let mut fixture = Fixture::new();
        let ghost = fixture.entities.spawn();
        fixture.spawn(Vec3::ZERO, FieldKind::ExactDistance, sphere(1.0), &[]);
        let root = fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Transform,
            &[ghost],
        );
        assert_eq!(fixture.compile(), Err(DomainError::Stale(ghost)));
        assert_ne!(root, ghost);
    }

    #[test]
    fn a_failed_compile_repeats_then_recovers_without_replacing_the_valid_program() {
        let mut fixture = Fixture::new();
        let leaf = fixture.spawn(Vec3::ZERO, FieldKind::ExactDistance, sphere(1.0), &[]);
        let up = fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Union,
            &[leaf, leaf],
        );
        let down = fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Union,
            &[up, leaf],
        );
        fixture.compile().expect("valid compile");
        let point = [3.0, 0.0, 0.0, 0.0];
        let valid = fixture.domain.field_program().evaluate(point).unwrap();

        fixture.domain.field_mut(up).unwrap().operands = vec![down, leaf];
        assert_eq!(fixture.compile(), Err(DomainError::FieldCycle(up)));
        assert_eq!(fixture.compile(), Err(DomainError::FieldCycle(up)));
        assert_eq!(
            fixture.domain.field_program().evaluate(point).unwrap(),
            valid
        );

        fixture.domain.field_mut(up).unwrap().operands = vec![leaf, leaf];
        assert!(fixture.compile().expect("repaired compile").full_rebuild);
        assert_eq!(
            fixture.domain.field_program().evaluate(point).unwrap(),
            valid
        );

        let mut fields = Store::tracked(DEFAULT_LOG_CAPACITY);
        let mut poses = Store::tracked(DEFAULT_LOG_CAPACITY);
        fields.bind(fixture.entities.scene());
        poses.bind(fixture.entities.scene());
        fields
            .insert(
                &fixture.entities,
                leaf,
                Field {
                    kind: FieldKind::ExactDistance,
                    op: sphere(1.0),
                    operands: Vec::new(),
                },
            )
            .unwrap();
        poses
            .insert(&fixture.entities, leaf, Pose::at(Vec3::ZERO))
            .unwrap();
        let mut compiler = FieldCompiler::new();
        compiler.compile(&EuclideanR3, &fields, &poses).unwrap();
        let pose_valid = compiler.program().evaluate(point).unwrap();
        poses.remove(leaf).unwrap();
        assert_eq!(
            compiler.compile(&EuclideanR3, &fields, &poses),
            Err(DomainError::Stale(leaf))
        );
        assert_eq!(
            compiler.compile(&EuclideanR3, &fields, &poses),
            Err(DomainError::Stale(leaf))
        );
        assert_eq!(compiler.program().evaluate(point).unwrap(), pose_valid);
    }

    #[test]
    fn moving_one_primitive_marks_only_the_operators_above_it() {
        let mut fixture = Fixture::new();
        let left = fixture.spawn(-Vec3::X, FieldKind::ExactDistance, sphere(0.5), &[]);
        let right = fixture.spawn(Vec3::X, FieldKind::ExactDistance, sphere(0.5), &[]);
        let far = fixture.spawn(Vec3::Y * 4.0, FieldKind::ExactDistance, sphere(0.5), &[]);
        let inner = fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Union,
            &[left, right],
        );
        fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Union,
            &[inner, far],
        );
        assert!(fixture.compile().expect("first compile").full_rebuild);
        assert!(fixture.compile().expect("settle").is_idle());

        fixture.move_to(left, Vec3::new(-2.0, 0.0, 0.0));
        let cost = fixture.compile().expect("incremental compile");
        assert_eq!(
            cost,
            FieldCost {
                changed_inputs: 1,
                affected_dependencies: 2,
                program_layout: 1,
                index_maintenance: 5,
                full_rebuild: false,
            }
        );

        fixture.move_to(far, Vec3::new(0.0, 5.0, 0.0));
        let cost = fixture.compile().expect("incremental compile");
        assert_eq!(cost.affected_dependencies, 1);
    }

    #[test]
    fn an_unchanged_store_compiles_with_no_work_and_no_rebuild() {
        let mut fixture = Fixture::new();
        let leaf = fixture.spawn(Vec3::ZERO, FieldKind::ExactDistance, sphere(1.0), &[]);
        fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Transform,
            &[leaf],
        );
        assert!(fixture.compile().expect("first compile").full_rebuild);
        let words = fixture.domain.field_program().program.len();
        for _ in 0..4 {
            assert!(fixture.compile().expect("idle compile").is_idle());
        }
        assert_eq!(fixture.domain.field_program().program.len(), words);
    }

    #[test]
    fn repeated_operand_edits_keep_index_storage_and_work_bounded() {
        let mut fixture = Fixture::new();
        let mut compiler = FieldCompiler::new();
        let left = fixture.spawn(-Vec3::X, FieldKind::ExactDistance, sphere(0.5), &[]);
        let right = fixture.spawn(Vec3::X, FieldKind::ExactDistance, sphere(0.5), &[]);
        let root = fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Union,
            &[left, right],
        );
        fixture.compile_with(&mut compiler).expect("first compile");
        fixture.compile_with(&mut compiler).expect("settle");

        let edge_slots = compiler.edges.len();
        for step in 0..128 {
            fixture.domain.field_mut(root).expect("root row").operands = if step % 2 == 0 {
                vec![right, left]
            } else {
                vec![left, right]
            };
            let cost = fixture.compile_with(&mut compiler).expect("operand edit");
            assert!(!cost.full_rebuild);
            assert_eq!(cost.index_maintenance, 7);
            assert!(cost.program_layout > 0);
            assert_eq!(compiler.edges.len(), edge_slots);
        }

        fixture.move_to(left, -Vec3::Y);
        assert_eq!(
            fixture
                .compile_with(&mut compiler)
                .unwrap()
                .affected_dependencies,
            1
        );
    }

    #[test]
    fn a_smooth_union_of_exact_leaves_reports_a_conservative_bound() {
        let mut fixture = Fixture::new();
        let left = fixture.spawn(-Vec3::X, FieldKind::ExactDistance, sphere(0.5), &[]);
        let right = fixture.spawn(Vec3::X, FieldKind::ExactDistance, sphere(0.5), &[]);
        fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::SmoothUnion { radius: 0.4 },
            &[left, right],
        );
        fixture.compile().expect("compile");
        let program = fixture.domain.field_program();
        assert_eq!(program.kind, FieldKind::ConservativeBound);
        assert_eq!(program.stack, 2);

        let closed_form = |p: Vec3| {
            let a = (p - Vec3::new(-1.0, 0.0, 0.0)).length() - 0.5;
            let b = (p - Vec3::new(1.0, 0.0, 0.0)).length() - 0.5;
            smooth_min(a, b, 0.4)
        };
        for probe in [
            Vec3::ZERO,
            Vec3::new(0.3, 0.2, 0.0),
            Vec3::new(-1.4, 0.0, 0.1),
        ] {
            let (value, kind) = program
                .evaluate([probe.x, probe.y, probe.z, 0.0])
                .expect("well-formed program");
            assert_eq!(kind, FieldKind::ConservativeBound);
            assert!(
                (value - closed_form(probe)).abs() < 1e-6,
                "{probe:?}: interpreter {value} against closed form {}",
                closed_form(probe)
            );
        }
    }

    #[test]
    fn invalid_field_dimensions_and_zero_smoothing_are_refused() {
        for op in [
            sphere(0.0),
            sphere(f32::INFINITY),
            FieldOp::Box {
                half_extents: [1.0, -1.0, 1.0],
            },
            FieldOp::HalfSpace {
                normal: Vec3::Y.to_array(),
                offset: f32::NAN,
            },
        ] {
            let mut fixture = Fixture::new();
            fixture.spawn(Vec3::ZERO, FieldKind::ExactDistance, op, &[]);
            assert!(matches!(
                fixture.compile(),
                Err(DomainError::InvalidCoordinate(_))
            ));
        }

        let mut fixture = Fixture::new();
        let left = fixture.spawn(-Vec3::X, FieldKind::ExactDistance, sphere(1.0), &[]);
        let right = fixture.spawn(Vec3::X, FieldKind::ExactDistance, sphere(1.0), &[]);
        fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::SmoothUnion { radius: 0.0 },
            &[left, right],
        );
        assert_eq!(
            fixture.compile(),
            Err(DomainError::InvalidCoordinate("field smoothing radius"))
        );
    }

    #[test]
    fn a_pose_transform_places_its_whole_subtree() {
        let mut fixture = Fixture::new();
        let leaf = fixture.spawn(Vec3::ZERO, FieldKind::ExactDistance, sphere(0.5), &[]);
        fixture.spawn(
            Vec3::new(3.0, 0.0, 0.0),
            FieldKind::ExactDistance,
            FieldOp::Transform,
            &[leaf],
        );
        fixture.compile().expect("compile");
        let program = fixture.domain.field_program();
        assert!((program.evaluate([3.0, 0.0, 0.0, 0.0]).expect("eval").0 + 0.5).abs() < 1e-6);
        assert!((program.evaluate([0.0, 0.0, 0.0, 0.0]).expect("eval").0 - 2.5).abs() < 1e-6);
    }

    #[test]
    fn a_posed_half_space_matches_the_domain_isometry() {
        use loam_math::{EuclideanR4, Iso4Flat, IsometryGroup, Rotor4};

        let half = std::f32::consts::FRAC_1_SQRT_2;
        let pose = Iso4Flat {
            rotation: Rotor4 {
                s: half,
                xy: half,
                ..Rotor4::IDENTITY
            },
            translation: crate::view::Vec4::new(0.3, -0.2, 1.5, 0.4),
        };
        let normal = [0.0, 2.0, 0.0, 0.0];
        let offset = 0.5;

        let mut entities = test_entities();
        let mut domain = DomainBuilder::new("r4", EuclideanR4)
            .tracked(DEFAULT_LOG_CAPACITY)
            .fields()
            .build(DomainId::new(0), entities.scene());
        let entity = entities.spawn();
        domain
            .attach_pose(entity, Pose::from(pose))
            .expect("pose row");
        domain
            .attach_field(
                entity,
                Field {
                    kind: FieldKind::ExactDistance,
                    op: FieldOp::HalfSpace4 { normal, offset },
                    operands: Vec::new(),
                },
            )
            .expect("field row");
        domain.compile_fields().expect("compile");

        let space = EuclideanR4;
        let inverse = space.iso_inverse(pose);
        let unit = crate::view::Vec4::from_array(normal).normalize();
        for point in [
            crate::view::Vec4::ZERO,
            crate::view::Vec4::new(1.0, -0.5, 0.25, 0.75),
            crate::view::Vec4::new(-2.0, 3.0, -1.0, 0.0),
        ] {
            let expected = space.iso_apply(inverse, point).dot(unit) - offset / 2.0;
            let value = domain
                .field_program()
                .evaluate(point.to_array())
                .expect("eval")
                .0;
            assert!(
                (value - expected).abs() < 1e-5,
                "{point:?}: record transform gave {value}, the isometry gives {expected}"
            );
        }
    }

    #[test]
    fn a_warmed_incremental_compile_allocates_nothing() {
        let mut fixture = Fixture::new();
        let leaves: Vec<Entity> = (0..32)
            .map(|i| {
                fixture.spawn(
                    Vec3::new(i as f32, 0.0, 0.0),
                    FieldKind::ExactDistance,
                    sphere(0.4),
                    &[],
                )
            })
            .collect();
        let mut level = leaves.clone();
        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|pair| {
                    fixture.spawn(Vec3::ZERO, FieldKind::ExactDistance, FieldOp::Union, pair)
                })
                .collect();
        }
        for _ in 0..4 {
            fixture.move_to(leaves[0], Vec3::new(0.5, 0.0, 0.0));
            fixture.compile().expect("warm up");
        }
        let bytes = bytes_allocated_by(|| {
            for step in 0..32 {
                fixture.move_to(
                    leaves[step % leaves.len()],
                    Vec3::new(step as f32, 1.0, 0.0),
                );
                let cost = fixture.compile().expect("incremental compile");
                assert!(!cost.full_rebuild);
                assert_eq!(cost.program_layout, 1);
            }
        });
        assert_eq!(
            bytes, 0,
            "32 warmed incremental compiles allocated {bytes} bytes"
        );
    }

    #[test]
    fn a_smooth_union_bound_that_drops_its_radius_culls_the_blend_surface() {
        let mut fixture = Fixture::new();
        let left = fixture.spawn(-Vec3::X, FieldKind::ExactDistance, sphere(0.5), &[]);
        let right = fixture.spawn(Vec3::X, FieldKind::ExactDistance, sphere(0.5), &[]);
        let blend = fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::SmoothUnion { radius: 5.0 },
            &[left, right],
        );
        let decoy = fixture.spawn(
            Vec3::new(-9.0, 4.0, 0.0),
            FieldKind::ExactDistance,
            sphere(6.55),
            &[],
        );
        fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Union,
            &[blend, decoy],
        );
        fixture.compile().expect("compile");

        let program = fixture.domain.field_program();
        let probe = [0.0, 4.0, 0.0, 0.0];
        let unculled = program.evaluate(probe).expect("eval").0;
        let culled = program.evaluate_bounded(probe, 0.0).expect("eval").0;
        assert!(
            (unculled - (17.0f32.sqrt() - 1.75)).abs() < 1e-5,
            "the blend value moved: {unculled}"
        );
        assert_eq!(
            culled, unculled,
            "the blend bulge reaches {unculled}, outside the unexpanded enclosing ball"
        );
    }

    #[test]
    fn a_posed_subtrees_bound_moves_with_its_transform() {
        let mut fixture = Fixture::new();
        let leaf = fixture.spawn(Vec3::ZERO, FieldKind::ExactDistance, sphere(0.5), &[]);
        let placed = fixture.spawn(
            Vec3::new(10.0, 0.0, 0.0),
            FieldKind::ExactDistance,
            FieldOp::Transform,
            &[leaf],
        );
        let decoy = fixture.spawn(
            Vec3::new(-20.0, 0.0, 0.0),
            FieldKind::ExactDistance,
            sphere(28.0),
            &[],
        );
        fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Union,
            &[decoy, placed],
        );
        fixture.compile().expect("compile");

        let program = fixture.domain.field_program();
        let probe = [10.0, 0.0, 0.0, 0.0];
        let culled = program.evaluate_bounded(probe, 0.0).expect("eval").0;
        assert!(
            (culled + 0.5).abs() < 1e-6,
            "the posed sphere reads {culled}, not -0.5, so its bound stayed at the origin"
        );
    }

    #[test]
    fn moving_a_shared_leaf_after_stamp_rollover_refreshes_each_dependent_cut_bound() {
        let mut fixture = Fixture::new();
        let mut compiler = FieldCompiler::new();
        let leaf = fixture.spawn(Vec3::ZERO, FieldKind::ExactDistance, sphere(0.5), &[]);
        let left = fixture.spawn(
            Vec3::new(-100.0, 0.0, 0.0),
            FieldKind::ExactDistance,
            FieldOp::Transform,
            &[leaf],
        );
        let right = fixture.spawn(
            Vec3::new(100.0, 0.0, 0.0),
            FieldKind::ExactDistance,
            FieldOp::Transform,
            &[leaf],
        );
        let left_decoy = fixture.spawn(
            Vec3::new(-120.0, 0.0, 0.0),
            FieldKind::ExactDistance,
            sphere(28.0),
            &[],
        );
        let right_decoy = fixture.spawn(
            Vec3::new(80.0, 0.0, 0.0),
            FieldKind::ExactDistance,
            sphere(28.0),
            &[],
        );
        let west = fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Union,
            &[left_decoy, left],
        );
        let east = fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Union,
            &[right_decoy, right],
        );
        fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Union,
            &[west, east],
        );
        fixture
            .compile_with(&mut compiler)
            .expect("initial compile");
        fixture.move_to(left, Vec3::new(-99.0, 0.0, 0.0));
        fixture
            .compile_with(&mut compiler)
            .expect("prime parent stamp");
        compiler.dirty_stamp = u32::MAX;
        fixture.move_to(leaf, Vec3::X * 10.0);
        fixture
            .compile_with(&mut compiler)
            .expect("move shared leaf");

        let program = compiler.program();
        for point in [[-89.0, 0.0, 0.0, 0.0], [110.0, 0.0, 0.0, 0.0]] {
            let value = program.evaluate_bounded(point, 0.0).expect("eval").0;
            assert!((value + 0.5).abs() < 1e-6, "{point:?} reads {value}");
        }
    }

    #[test]
    fn an_unbounded_subtree_stays_in_the_traversal_and_still_contributes() {
        let mut fixture = Fixture::new();
        let ball = fixture.spawn(Vec3::ZERO, FieldKind::ExactDistance, sphere(0.5), &[]);
        let ground = fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::HalfSpace {
                normal: [0.0, 1.0, 0.0],
                offset: 0.0,
            },
            &[],
        );
        fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Union,
            &[ball, ground],
        );
        fixture.compile().expect("compile");

        let program = fixture.domain.field_program();
        let probe = [0.0, -3.0, 0.0, 0.0];
        let culled = program.evaluate_bounded(probe, 0.0).expect("eval").0;
        assert!(
            (culled + 3.0).abs() < 1e-6,
            "the half-space reads {culled}, not -3.0, so the unbounded subtree was dropped"
        );
    }

    #[test]
    fn a_skipped_subtree_never_changes_the_value_the_unculled_program_returns() {
        let mut fixture = Fixture::new();
        let mut roots = Vec::new();
        for i in 0..64 {
            let t = i as f32;
            roots.push(fixture.spawn(
                Vec3::new((t * 0.37).sin() * 40.0, (t * 0.71).cos() * 40.0, t * 0.9),
                FieldKind::ExactDistance,
                sphere(0.4),
                &[],
            ));
        }
        let minuend = fixture.spawn(
            Vec3::new(2.0, 0.0, 0.0),
            FieldKind::ExactDistance,
            sphere(1.0),
            &[],
        );
        let subtrahend = fixture.spawn(
            Vec3::new(0.0, 30.0, 0.0),
            FieldKind::ExactDistance,
            sphere(1.0),
            &[],
        );
        roots.push(fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Subtraction,
            &[minuend, subtrahend],
        ));
        let inner = fixture.spawn(
            Vec3::new(5.0, 0.0, 0.0),
            FieldKind::ExactDistance,
            sphere(1.0),
            &[],
        );
        let outer = fixture.spawn(
            Vec3::new(5.0, 0.0, 0.0),
            FieldKind::ExactDistance,
            sphere(3.0),
            &[],
        );
        roots.push(fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Intersection,
            &[inner, outer],
        ));
        let mut level = roots;
        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|pair| {
                    if pair.len() == 1 {
                        pair[0]
                    } else {
                        fixture.spawn(Vec3::ZERO, FieldKind::ExactDistance, FieldOp::Union, pair)
                    }
                })
                .collect();
        }
        fixture.compile().expect("compile");

        let program = fixture.domain.field_program();
        let mut counts = FieldCounts::default();
        for i in 0..64 {
            let t = i as f32;
            let probe = [
                (t * 0.29).cos() * 12.0,
                (t * 0.53).sin() * 12.0,
                t * 0.5 - 8.0,
                0.0,
            ];
            let unculled = program.evaluate(probe).expect("eval").0;
            let culled = evaluate_bounded_counted(
                &program.program,
                &program.primitives,
                &program.nodes,
                probe,
                0.0,
                &mut counts,
            )
            .expect("eval");
            assert_eq!(culled, unculled, "probe {probe:?} changed under culling");
        }
        assert!(
            counts.node_skips > 0,
            "no subtree was skipped, so the traversal was never exercised"
        );
    }

    #[test]
    fn distance_past_the_old_sentinel_matches_a_f64_oracle_with_reported_error() {
        use loam_shape::field::DistanceField;

        let mut fixture = Fixture::new();
        fixture.compile().expect("empty compile");
        let empty = fixture.domain.field_program();
        assert_eq!(empty.field_kind(), FieldKind::ConservativeBound);
        assert_eq!(empty.evaluate([0.0; 4]).unwrap().0, FIELD_FAR);

        fixture.spawn(Vec3::ZERO, FieldKind::ExactDistance, sphere(1.0), &[]);
        fixture.compile().expect("compile");
        let program = fixture.domain.field_program();
        let point = [2.0e9, 0.0, 0.0, 0.0];
        let unculled = program.evaluate(point).expect("eval").0;
        let culled = program.evaluate_bounded(point, 0.0).expect("eval").0;
        let oracle = 2_000_000_000.0f64 - 1.0;

        assert!(culled > FIELD_FAR);
        assert_eq!(culled, unculled);
        assert!((culled as f64 - oracle).abs() <= program.error_at(point) as f64);
    }

    #[test]
    fn an_implicit_union_of_exact_spheres_refuses_an_interior_witness() {
        use loam_shape::field::DistanceField;

        let mut fixture = Fixture::new();
        fixture.spawn(-Vec3::X * 0.5, FieldKind::ExactDistance, sphere(1.0), &[]);
        fixture.spawn(Vec3::X * 0.5, FieldKind::ExactDistance, sphere(1.0), &[]);
        fixture.compile().expect("compile");
        let program = fixture.domain.field_program();
        assert_eq!(program.field_kind(), FieldKind::ConservativeBound);

        let probe = Vec3::new(0.0, 0.2, 0.0);
        let bound = program.distance(probe.extend(0.0).to_array());
        let distance = 0.2 - 0.75f32.sqrt();
        assert!((bound + 0.461_483_5).abs() < 1e-6);
        assert!((bound - distance).abs() > FIELD_PROGRAM_ERROR);
        let gradient = program.gradient(probe.extend(0.0).to_array());
        let normal = Vec3::from_slice(&gradient[..3]).normalize();
        let witness = probe - normal * bound;
        assert!(
            program.distance(witness.extend(0.0).to_array()) < -FIELD_PROGRAM_ERROR,
            "the projected witness {witness:?} reached the union boundary"
        );

        #[cfg(feature = "physics")]
        {
            use loam_physics::euclidean_r3::sphere_body_r3;
            use loam_physics::field_contact::sphere_against_field;
            use loam_physics::{FieldRefusal, World};

            let mut world = World::new(EuclideanR3);
            let body = world.push_body(sphere_body_r3(probe, Vec3::ZERO, 0.1, 1.0).expect("body"));
            assert_eq!(
                sphere_against_field(
                    &world.bodies()[body],
                    world.geometry(),
                    program,
                    &EuclideanR3
                )
                .err(),
                Some(FieldRefusal::Kind(FieldKind::ConservativeBound))
            );
        }
    }

    #[test]
    fn r4_fields_keep_off_slice_culling_and_refuse_r3_use() {
        use crate::view::Vec4;
        use loam_math::EuclideanR4;
        use loam_shape::field::DistanceField;

        for op in [
            FieldOp::HyperSphere { radius: 1.0 },
            FieldOp::HalfSpace4 {
                normal: [1.0, 0.0, 0.0, 1.0],
                offset: 0.0,
            },
        ] {
            let mut r3 = Fixture::new();
            r3.spawn(Vec3::ZERO, FieldKind::ExactDistance, op, &[]);
            assert_eq!(
                r3.compile(),
                Err(DomainError::Unsupported("field primitive dimension"))
            );
        }

        let mut entities = test_entities();
        let mut domain = DomainBuilder::new("r4", EuclideanR4)
            .tracked(DEFAULT_LOG_CAPACITY)
            .fields()
            .build(DomainId::new(0), entities.scene());
        let mut place = |at: Vec4, op: FieldOp, operands: &[Entity]| {
            let entity = entities.spawn();
            domain.attach_pose(entity, Pose::at(at)).expect("pose row");
            domain
                .attach_field(
                    entity,
                    Field {
                        kind: FieldKind::ExactDistance,
                        op,
                        operands: operands.to_vec(),
                    },
                )
                .expect("field row");
            entity
        };
        let shell = place(Vec4::new(0.0, 0.0, 0.0, 10.0), sphere(1.0), &[]);
        let near = place(
            Vec4::new(5.0, 0.0, 0.0, 0.0),
            FieldOp::HyperSphere { radius: 2.0 },
            &[],
        );
        let root = place(Vec4::ZERO, FieldOp::Union, &[near, shell]);
        domain.compile_fields().expect("compile");

        let program = domain.field_program();
        let probe = [1.0, 0.0, 0.0, 0.0];
        let unculled = program.evaluate(probe).expect("eval").0;
        let culled = program.evaluate_bounded(probe, 0.0).expect("eval").0;
        assert!(
            unculled.abs() < 1e-6,
            "the unit sphere translated to w = 10 reads {unculled} on its own surface"
        );
        assert_eq!(
            culled, unculled,
            "the hierarchy culled a surface the program evaluates"
        );

        domain.remove_field(root).expect("root field");
        domain.remove_field(shell).expect("shell field");
        domain
            .set_point(near, Vec4::new(0.0, 0.0, 0.0, 1.0))
            .expect("hypersphere pose");
        domain.compile_fields().expect("compile hypersphere");

        let program = domain.field_program();
        let point = [1.0, 0.0, 0.0, 0.0];
        let r4_distance = program.distance(point);
        let r3_slice_distance = 1.0 - 3.0f32.sqrt();
        assert_eq!(program.dimension(), 4);
        assert_eq!(program.field_kind(), FieldKind::ExactDistance);
        assert!((r4_distance - r3_slice_distance).abs() > program.error_at(point));

        #[cfg(feature = "physics")]
        {
            use loam_physics::euclidean_r3::sphere_body_r3;
            use loam_physics::field_contact::sphere_against_field;
            use loam_physics::{FieldRefusal, World};

            let mut world = World::new(EuclideanR3);
            let body = world
                .push_body(sphere_body_r3(Vec3::X, Vec3::ZERO, 0.1, 1.0).expect("sphere body"));
            assert_eq!(
                sphere_against_field(
                    &world.bodies()[body],
                    world.geometry(),
                    program,
                    &EuclideanR3
                )
                .err(),
                Some(FieldRefusal::Dimension(4))
            );
        }
    }

    #[test]
    fn an_intersection_of_exact_half_spaces_reports_a_bound_not_a_distance() {
        let mut fixture = Fixture::new();
        let west = fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::HalfSpace {
                normal: [-1.0, 0.0, 0.0],
                offset: 0.0,
            },
            &[],
        );
        let south = fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::HalfSpace {
                normal: [0.0, -1.0, 0.0],
                offset: 0.0,
            },
            &[],
        );
        fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Intersection,
            &[west, south],
        );
        fixture.compile().expect("compile");

        let program = fixture.domain.field_program();
        let corner = [-1.0, -1.0, 0.0, 0.0];
        let (value, kind) = program.evaluate(corner).expect("eval");
        assert_eq!(kind, FieldKind::ConservativeBound);
        assert!(
            (value - 1.0).abs() < 1e-6,
            "the intersection reads {value}, not the max of the two half-spaces"
        );
        assert!(
            value < 2.0f32.sqrt() - 1e-3,
            "{value} is not below the true distance {} to the quadrant",
            2.0f32.sqrt()
        );

        #[cfg(feature = "physics")]
        {
            use loam_physics::euclidean_r3::sphere_body_r3;
            use loam_physics::field_contact::sphere_against_field;
            use loam_physics::{FieldRefusal, World};

            let mut world = World::new(EuclideanR3);
            let body = world.push_body(
                sphere_body_r3(Vec3::new(-1.0, -1.0, 0.0), Vec3::ZERO, 1.2, 1.0).expect("body"),
            );
            assert_eq!(
                sphere_against_field(
                    &world.bodies()[body],
                    world.geometry(),
                    program,
                    &EuclideanR3
                )
                .err(),
                Some(FieldRefusal::Kind(FieldKind::ConservativeBound)),
                "the contact query read a bound as separation"
            );
        }
    }

    #[test]
    fn a_curved_chart_refuses_to_compile_a_field() {
        use loam_math::HyperbolicH3;
        let entities = test_entities();
        let mut domain = DomainBuilder::new("h3", HyperbolicH3)
            .tracked(DEFAULT_LOG_CAPACITY)
            .fields()
            .build(DomainId::new(0), entities.scene());
        assert_eq!(
            domain.compile_fields(),
            Err(DomainError::Unsupported("curved field chart"))
        );
    }
}
