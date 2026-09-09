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

    /// The weakest operand kind, dropped to `ConservativeBound` by subtraction and smooth union, which are not distances.
    pub fn result_kind(self, operands: &[FieldKind]) -> FieldKind {
        let weakest = operands
            .iter()
            .copied()
            .fold(FieldKind::ExactDistance, FieldKind::weaker);
        match self {
            FieldOp::Union | FieldOp::Intersection | FieldOp::Transform => weakest,
            FieldOp::Subtraction | FieldOp::SmoothUnion { .. } => {
                weakest.weaker(FieldKind::ConservativeBound)
            }
            _ => weakest,
        }
    }
}

/// Matches `LoamFieldPrimitive` in field_march.wgsl; a half-space's offset is folded into `translation` at compile time.
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

// Quilez, "Smooth minimum", iquilezles.org/articles/smin, polynomial form.
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    (b * (1.0 - h) + a * h) - k * h * (1.0 - h)
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

const NONE: u32 = u32::MAX;

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
struct Edge {
    operand: u32,
    operator: u32,
    next: u32,
    live: bool,
}

pub struct FieldCompiler {
    program: FieldProgram,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    slots: Vec<u32>,
    roots: Vec<u32>,
    dfs: Vec<(u32, u32)>,
    kinds: Vec<FieldKind>,
    work: Vec<u32>,
    touched: Vec<u32>,
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
    pub fn new() -> Self {
        Self {
            program: FieldProgram::default(),
            nodes: Vec::new(),
            edges: Vec::new(),
            slots: Vec::new(),
            roots: Vec::new(),
            dfs: Vec::new(),
            kinds: Vec::new(),
            work: Vec::new(),
            touched: Vec::new(),
            fields_cursor: Cursor::default(),
            poses_cursor: Cursor::default(),
            dirty_stamp: 0,
            compiled: false,
        }
    }

    pub fn program(&self) -> &FieldProgram {
        &self.program
    }

    pub fn invalidate(&mut self) {
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
        let mut edge = self.nodes[node as usize].first_parent;
        while edge != NONE {
            let e = self.edges[edge as usize];
            if e.live {
                return true;
            }
            edge = e.next;
        }
        false
    }

    fn link(&mut self, operator: u32, operands: &[Entity]) -> Result<(), DomainError> {
        let entity = self.nodes[operator as usize].entity;
        if operands.len() != self.nodes[operator as usize].op.operands() {
            return Err(DomainError::FieldArity(entity));
        }
        self.nodes[operator as usize].edge_start = self.edges.len() as u32;
        self.nodes[operator as usize].edge_len = operands.len() as u32;
        for &operand in operands {
            let target = self.node_of(operand).ok_or(DomainError::Stale(operand))?;
            let edge = self.edges.len() as u32;
            self.edges.push(Edge {
                operand: target,
                operator,
                next: self.nodes[target as usize].first_parent,
                live: true,
            });
            self.nodes[target as usize].first_parent = edge;
        }
        Ok(())
    }

    fn unlink(&mut self, operator: u32) {
        let node = self.nodes[operator as usize];
        for edge in node.edge_start..node.edge_start + node.edge_len {
            self.edges[edge as usize].live = false;
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
        self.dirty_stamp = self.dirty_stamp.wrapping_add(1);
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
        let mut work = 0;
        for i in 0..self.touched.len() {
            let node = self.nodes[self.touched[i] as usize];
            if node.record == NONE {
                continue;
            }
            self.program.primitives[node.record as usize] =
                primitive_record(space, poses, node.entity, node.op)?;
            work += 1;
        }
        Ok(work)
    }

    fn emit<S: DomainSpace>(
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
        let mut height = 0u32;
        let mut peak = 0u32;
        let mut pose_depth = 0usize;
        for index in 0..self.roots.len() {
            self.dfs.push((self.roots[index], 0));
            while !self.dfs.is_empty() {
                let top = self.dfs.len() - 1;
                let (node, cursor) = self.dfs[top];
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
                    self.dfs[top].1 = cursor + 1;
                    let start = self.nodes[node as usize].edge_start;
                    let child = self.edges[(start + cursor) as usize].operand;
                    self.dfs.push((child, 0));
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
        self.program.kind = self.kinds.pop().unwrap_or(FieldKind::ExactDistance);
        Ok((self.program.program.len() / 2) as u32 + self.program.primitives.len() as u32)
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

    /// Refuses a curved chart; then a pose change patches records and marks ancestors, an operand edit repairs that operator's edges, and only an insert, a removal, a resync, or the first compile rebuilds everything.
    pub fn compile<S: DomainSpace>(
        &mut self,
        space: &S,
        fields: &Store<Field>,
        poses: &Store<Pose<S>>,
    ) -> Result<FieldCost, DomainError> {
        if !space.is_chart_flat() {
            return Err(DomainError::Unsupported("curved field chart"));
        }
        let structural = fields.changed_since(&mut self.fields_cursor);
        let placed = poses.changed_since(&mut self.poses_cursor);
        if self.compiled && !structural && !placed {
            return Ok(FieldCost::default());
        }

        let mut cost = FieldCost::default();
        let mut rebuild = !self.compiled;
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
                if let Change::Row(entity, _) = change {
                    if let Some(node) = self.node_of(entity) {
                        self.touched.push(node);
                        cost.changed_inputs += 1;
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
    let pose = poses.get(entity).ok_or(DomainError::Stale(entity))?;
    let chart = space.chart_pose(&pose.0);
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
    Ok(record)
}

fn unit_plane(normal: [f32; 4], offset: f32) -> Result<([f32; 4], f32), DomainError> {
    let length = normal
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt();
    if !length.is_finite() || length < 1e-6 {
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
    use loam_math::{EuclideanR3, Iso3};

    use super::*;
    use crate::domain::{Domain, DomainBuilder, DomainId, TypedDomain};
    use crate::entity::{Entities, Entity, Epoch, RuntimeId, SceneId};
    use crate::store::tests::alloc_probe;
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
                .poses
                .insert(entity, Pose(Iso3::from_translation(at)))
                .expect("pose row");
            self.domain
                .fields_mut()
                .expect("field store")
                .insert(
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

        fn move_to(&mut self, entity: Entity, to: Vec3) {
            self.domain
                .poses
                .get_mut(entity)
                .expect("pose row")
                .0
                .translation = to;
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
    fn a_cycle_is_refused_and_names_an_entity_on_it() {
        let mut fixture = Fixture::new();
        let leaf = fixture.spawn(Vec3::ZERO, FieldKind::ExactDistance, sphere(1.0), &[]);
        let up = fixture.entities.spawn();
        let down = fixture.entities.spawn();
        for (entity, operand) in [(up, down), (down, up)] {
            fixture
                .domain
                .poses
                .insert(entity, Pose(Iso3::IDENTITY))
                .expect("pose row");
            fixture
                .domain
                .fields_mut()
                .expect("field store")
                .insert(
                    entity,
                    Field {
                        kind: FieldKind::ExactDistance,
                        op: FieldOp::Union,
                        operands: vec![operand, leaf],
                    },
                )
                .expect("field row");
        }
        assert_eq!(fixture.compile(), Err(DomainError::FieldCycle(up)));
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
                index_maintenance: 0,
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
    fn an_operand_list_edit_repairs_the_index_without_a_full_rebuild() {
        let mut fixture = Fixture::new();
        let left = fixture.spawn(-Vec3::X, FieldKind::ExactDistance, sphere(0.5), &[]);
        let right = fixture.spawn(Vec3::X, FieldKind::ExactDistance, sphere(0.5), &[]);
        let root = fixture.spawn(
            Vec3::ZERO,
            FieldKind::ExactDistance,
            FieldOp::Union,
            &[left, right],
        );
        fixture.compile().expect("first compile");
        fixture.compile().expect("settle");

        fixture
            .domain
            .fields_mut()
            .expect("field store")
            .get_mut(root)
            .expect("root row")
            .operands = vec![right, left];
        let cost = fixture.compile().expect("operand edit");
        assert!(!cost.full_rebuild);
        assert_eq!(cost.index_maintenance, 4);
        assert!(cost.program_layout > 0);
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
        domain.poses.insert(entity, Pose(pose)).expect("pose row");
        domain
            .fields_mut()
            .expect("field store")
            .insert(
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
        let bytes = alloc_probe::bytes_allocated_by(|| {
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
