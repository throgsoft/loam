use std::fmt;

use bytemuck::{Pod, Zeroable};

use crate::domain::{DomainError, DomainSpace, Field, FieldKind, FieldProgram, Pose};
use crate::entity::Entity;
use crate::store::{Change, Cursor, Store};

pub(crate) const OP_SPHERE: u32 = 1;
pub(crate) const OP_BOX: u32 = 2;
pub(crate) const OP_HALFSPACE: u32 = 3;
pub(crate) const OP_HYPERSPHERE: u32 = 4;
pub(crate) const OP_HALFSPACE4: u32 = 5;
pub(crate) const OP_UNION: u32 = 6;
pub(crate) const OP_INTERSECTION: u32 = 7;
pub(crate) const OP_SUBTRACTION: u32 = 8;
pub(crate) const OP_SMOOTH_UNION: u32 = 9;
pub(crate) const OP_PUSH_POSE: u32 = 10;
pub(crate) const OP_POP_POSE: u32 = 11;

pub(crate) const MAX_STACK: usize = 32;
pub(crate) const MAX_POSE_DEPTH: usize = 8;
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
pub(crate) struct FieldPrimitive {
    pub frame: [[f32; 4]; 4],
    pub translation: [f32; 4],
    pub params: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldError {
    Opcode(u32),
    Truncated,
    StackOverflow,
    StackUnderflow,
    PoseDepth,
    Primitive(u32),
    ProgramSize,
    CurvedChart,
    Dimension,
}

impl fmt::Display for FieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Opcode(op) => write!(f, "the field program has an unknown opcode {op}"),
            Self::Truncated => f.write_str("the field program ends inside an instruction"),
            Self::StackOverflow => f.write_str("the field program overflows its stack"),
            Self::StackUnderflow => f.write_str("the field program underflows its stack"),
            Self::PoseDepth => f.write_str("the field transforms nest too deep"),
            Self::Primitive(index) => write!(f, "the field program has no primitive {index}"),
            Self::ProgramSize => f.write_str("the field program is too large"),
            Self::CurvedChart => f.write_str("fields need a flat chart"),
            Self::Dimension => f.write_str("a field primitive has the wrong dimension"),
        }
    }
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

// Moore, Kearfott, and Cloud 2009, Introduction to Interval Analysis, sec. 2.1.
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
pub(crate) struct FieldNode {
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

// Horn and Johnson 2013, Matrix Analysis, sec. 5.6 and 7.3.
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

pub(crate) fn evaluate(
    program: &[u32],
    primitives: &[FieldPrimitive],
    point: [f32; 4],
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
        match op {
            OP_SPHERE | OP_BOX | OP_HALFSPACE | OP_HYPERSPHERE | OP_HALFSPACE4 => {
                let prim = primitives
                    .get(arg as usize)
                    .ok_or(FieldError::Primitive(arg))?;
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

// Skips a subtree whose ball's lower bound exceeds the best value so far by more than `tolerance`; an empty `nodes` slice evaluates the whole program.
pub(crate) fn evaluate_bounded(
    program: &[u32],
    primitives: &[FieldPrimitive],
    nodes: &[FieldNode],
    point: [f32; 4],
    tolerance: f32,
) -> Result<f32, FieldError> {
    if nodes.is_empty() {
        return evaluate(program, primitives, point);
    }
    let mut best = f32::INFINITY;
    let mut index = 0usize;
    while index < nodes.len() {
        let node = nodes[index];
        if node.is_bounded() && node.lower_bound(point) > best + tolerance {
            index = node.escape as usize;
            continue;
        }
        if node.is_leaf() {
            let range = program
                .get(node.start as usize..node.end as usize)
                .ok_or(FieldError::Truncated)?;
            best = best.min(evaluate(range, primitives, point)?);
        }
        index += 1;
    }
    Ok(best)
}

pub(crate) fn evaluate_error(
    program: &[u32],
    primitives: &[FieldPrimitive],
    nodes: &[FieldNode],
    point: [f32; 4],
    kind: FieldKind,
) -> Result<f32, FieldError> {
    let value = evaluate_bounded(program, primitives, nodes, point, 0.0)?;
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
                        return Err(DomainError::Field(FieldError::PoseDepth));
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
            return Err(DomainError::Field(FieldError::StackOverflow));
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
            return Err(DomainError::Field(FieldError::ProgramSize));
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
            return Err(DomainError::Field(FieldError::CurvedChart));
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
            return Err(DomainError::Field(FieldError::Dimension));
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
