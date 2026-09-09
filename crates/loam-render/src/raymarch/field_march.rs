use bytemuck::{Pod, Zeroable};
use loam_runtime::domain::FieldKind;
use loam_runtime::field::{
    FieldNode, FieldPrimitive, MAX_POSE_DEPTH, MAX_STACK, OP_BOX, OP_HALFSPACE, OP_HALFSPACE4,
    OP_HYPERSPHERE, OP_INTERSECTION, OP_POP_POSE, OP_PUSH_POSE, OP_SMOOTH_UNION, OP_SPHERE,
    OP_SUBTRACTION, OP_UNION,
};
use loam_runtime::FieldProgram;
use wgpu::*;

/// Exact and conservative kinds step the full distance; `FixedStep` advances `implicit_step` because an implicit value bounds nothing.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MarchMode {
    ExactDistance,
    ConservativeBound,
    FixedStep,
}

impl MarchMode {
    fn of(kind: FieldKind) -> Self {
        match kind {
            FieldKind::ExactDistance => MarchMode::ExactDistance,
            FieldKind::ConservativeBound => MarchMode::ConservativeBound,
            FieldKind::Implicit => MarchMode::FixedStep,
        }
    }

    fn code(self) -> u32 {
        match self {
            MarchMode::ExactDistance => 0,
            MarchMode::ConservativeBound => 1,
            MarchMode::FixedStep => 2,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FieldMarchUniforms {
    pub camera_pos: [f32; 3],
    pub _pad0: f32,
    pub camera_forward: [f32; 3],
    pub _pad1: f32,
    pub camera_right: [f32; 3],
    pub _pad2: f32,
    pub camera_up: [f32; 3],
    pub fov_y_tan: f32,
    pub resolution: [f32; 2],
    pub viewport_origin: [f32; 2],
    pub params: [f32; 4],
    pub near: f32,
    pub w_slice: f32,
    pub implicit_step: f32,
    pub max_t: f32,
    pub program_len: u32,
    pub kind: u32,
    pub node_len: u32,
    /// Slack added to the best value before a subtree's ball skips it.
    pub cull_tolerance: f32,
}

impl Default for FieldMarchUniforms {
    fn default() -> Self {
        Self {
            camera_pos: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            camera_forward: [0.0, 0.0, -1.0],
            _pad1: 0.0,
            camera_right: [1.0, 0.0, 0.0],
            _pad2: 0.0,
            camera_up: [0.0, 1.0, 0.0],
            fov_y_tan: (60.0_f32.to_radians() * 0.5).tan(),
            resolution: [1.0, 1.0],
            viewport_origin: [0.0, 0.0],
            params: [0.65, 0.65, 0.72, 0.0],
            near: 0.05,
            w_slice: 0.0,
            implicit_step: 0.02,
            max_t: 60.0,
            program_len: 0,
            kind: 0,
            node_len: 0,
            cull_tolerance: 0.001,
        }
    }
}

fn prelude(counting: bool) -> String {
    format!(
        "{depth}\
         const LOAM_OP_SPHERE: u32 = {OP_SPHERE}u;\n\
         const LOAM_OP_BOX: u32 = {OP_BOX}u;\n\
         const LOAM_OP_HALFSPACE: u32 = {OP_HALFSPACE}u;\n\
         const LOAM_OP_HYPERSPHERE: u32 = {OP_HYPERSPHERE}u;\n\
         const LOAM_OP_HALFSPACE4: u32 = {OP_HALFSPACE4}u;\n\
         const LOAM_OP_UNION: u32 = {OP_UNION}u;\n\
         const LOAM_OP_INTERSECTION: u32 = {OP_INTERSECTION}u;\n\
         const LOAM_OP_SUBTRACTION: u32 = {OP_SUBTRACTION}u;\n\
         const LOAM_OP_SMOOTH_UNION: u32 = {OP_SMOOTH_UNION}u;\n\
         const LOAM_OP_PUSH_POSE: u32 = {OP_PUSH_POSE}u;\n\
         const LOAM_OP_POP_POSE: u32 = {OP_POP_POSE}u;\n\
         const LOAM_MAX_STACK: u32 = {MAX_STACK}u;\n\
         const LOAM_MAX_POSE_DEPTH: u32 = {MAX_POSE_DEPTH}u;\n\
         const LOAM_FIELD_IMPLICIT: u32 = 2u;\n\
         const LOAM_FIELD_FAR: f32 = 1.0e9;\n\
         const LOAM_FIELD_COUNTING: bool = {counting};\n\
         {common}",
        depth = include_str!("../shader/projective_depth.wgsl"),
        common = include_str!("field_common.wgsl"),
    )
}

const COUNTS_ENTRY: &str = "\n@fragment\nfn fs_counts(@builtin(position) frag_pos: vec4<f32>) \
                            -> @location(0) vec4<f32> {\n    let shaded = shade(frag_pos);\n    \
                            return vec4<f32>(f32(loam_field_visits), f32(loam_field_skips), \
                            f32(loam_field_evals), shaded.depth);\n}\n";

pub fn field_march_wgsl() -> String {
    interpreted_module(false)
}

pub fn field_march_counting_wgsl() -> String {
    interpreted_module(true)
}

fn interpreted_module(counting: bool) -> String {
    format!(
        "{}{}{INTERPRETED_HOOKS}{}{}{}",
        prelude(counting),
        include_str!("field_march.wgsl"),
        include_str!("field_walk.wgsl"),
        include_str!("field_shade.wgsl"),
        if counting { COUNTS_ENTRY } else { "" },
    )
}

pub fn field_specialized_wgsl(program: &FieldProgram) -> String {
    specialized_module(&program.program, false)
}

fn specialized_module(program: &[u32], counting: bool) -> String {
    format!(
        "{}{}{}{WHOLE_PROGRAM_HOOK}{}{}{}",
        prelude(counting),
        include_str!("field_march.wgsl"),
        specialized_cuts(program),
        include_str!("field_walk.wgsl"),
        include_str!("field_shade.wgsl"),
        if counting { COUNTS_ENTRY } else { "" },
    )
}

const INTERPRETED_HOOKS: &str = "\nfn loam_field_leaf(node: LoamFieldNode, p: vec4<f32>) -> f32 {\n    return loam_field_range(node.start, node.end, p);\n}\n\nfn loam_field_all(p: vec4<f32>) -> f32 {\n    return loam_field_range(0u, u.program_len, p);\n}\n";

const WHOLE_PROGRAM_HOOK: &str = "\nfn loam_field_all(p: vec4<f32>) -> f32 {\n    return loam_field_range(0u, u.program_len, p);\n}\n";

const FAR_CUTS: &str = "fn loam_field_leaf(node: LoamFieldNode, p: vec4<f32>) -> f32 {\n    \
                        return LOAM_FIELD_FAR;\n}\n";

#[derive(Clone, Copy)]
struct Produced {
    start: usize,
    end: usize,
    op: u32,
    left: usize,
    right: usize,
}

fn union_cuts(program: &[u32]) -> Option<Vec<(usize, usize)>> {
    if program.is_empty() || !program.len().is_multiple_of(2) {
        return None;
    }
    let mut arena: Vec<Produced> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut poses: Vec<usize> = Vec::new();
    for (offset, word) in program.chunks_exact(2).enumerate() {
        let index = offset * 2;
        match word[0] {
            OP_SPHERE | OP_BOX | OP_HALFSPACE | OP_HYPERSPHERE | OP_HALFSPACE4 => {
                stack.push(arena.len());
                arena.push(Produced {
                    start: index,
                    end: index + 2,
                    op: word[0],
                    left: usize::MAX,
                    right: usize::MAX,
                });
            }
            OP_UNION | OP_INTERSECTION | OP_SUBTRACTION | OP_SMOOTH_UNION => {
                let (right, left) = (stack.pop()?, stack.pop()?);
                stack.push(arena.len());
                arena.push(Produced {
                    start: arena[left].start,
                    end: index + 2,
                    op: word[0],
                    left,
                    right,
                });
            }
            OP_PUSH_POSE => poses.push(index),
            OP_POP_POSE => {
                let opened = poses.pop()?;
                let top = *stack.last()?;
                arena[top].start = opened;
                arena[top].end = index + 2;
                arena[top].op = OP_POP_POSE;
            }
            _ => return None,
        }
    }
    if stack.len() != 1 || !poses.is_empty() {
        return None;
    }
    let mut cuts = Vec::new();
    let mut pending = vec![stack[0]];
    while let Some(node) = pending.pop() {
        if arena[node].op == OP_UNION {
            pending.push(arena[node].right);
            pending.push(arena[node].left);
        } else {
            cuts.push((arena[node].start, arena[node].end));
        }
    }
    cuts.sort_unstable();
    Some(cuts)
}

fn straight_line(program: &[u32], cut: (usize, usize), name: &str) -> Option<String> {
    let range = program.get(cut.0..cut.1)?;
    if range.is_empty() || !range.len().is_multiple_of(2) {
        return None;
    }
    let mut body = format!("fn {name}(p0: vec4<f32>) -> f32 {{\n");
    let mut values: Vec<u32> = Vec::new();
    let mut poses: Vec<u32> = vec![0];
    let mut next_value = 0u32;
    let mut next_pose = 1u32;
    for (offset, word) in range.chunks_exact(2).enumerate() {
        let (op, arg) = (word[0], word[1]);
        let point = *poses.last()?;
        match op {
            OP_SPHERE | OP_BOX | OP_HALFSPACE | OP_HYPERSPHERE | OP_HALFSPACE4 => {
                body.push_str(&format!(
                    "    let v{next_value} = loam_field_primitive({op}u, {arg}u, p{point});\n"
                ));
                values.push(next_value);
                next_value += 1;
            }
            OP_UNION | OP_INTERSECTION | OP_SUBTRACTION | OP_SMOOTH_UNION => {
                let (b, a) = (values.pop()?, values.pop()?);
                let expression = match op {
                    OP_UNION => format!("min(v{a}, v{b})"),
                    OP_INTERSECTION => format!("max(v{a}, v{b})"),
                    OP_SUBTRACTION => format!("max(v{a}, -v{b})"),
                    _ => format!(
                        "loam_field_smooth_min(v{a}, v{b}, bitcast<f32>(loam_field_prog[{}u]))",
                        cut.0 + offset * 2 + 1
                    ),
                };
                body.push_str(&format!("    let v{next_value} = {expression};\n"));
                values.push(next_value);
                next_value += 1;
            }
            OP_PUSH_POSE => {
                if poses.len() > MAX_POSE_DEPTH {
                    return None;
                }
                body.push_str(&format!(
                    "    let p{next_pose} = loam_field_local(loam_field_prims[{arg}u], p{point});\n"
                ));
                poses.push(next_pose);
                next_pose += 1;
            }
            OP_POP_POSE => {
                poses.pop();
                if poses.is_empty() {
                    return None;
                }
            }
            _ => return None,
        }
    }
    match values.as_slice() {
        [root] => {
            body.push_str(&format!("    return v{root};\n}}\n"));
            Some(body)
        }
        _ => None,
    }
}

fn dispatch(starts: &[usize], lo: usize, hi: usize, depth: usize, out: &mut String) {
    let pad = "    ".repeat(depth + 1);
    if hi - lo == 1 {
        out.push_str(&format!("{pad}return loam_field_cut_{lo}(p0);\n"));
        return;
    }
    let mid = lo + (hi - lo) / 2;
    out.push_str(&format!("{pad}if (start < {}u) {{\n", starts[mid]));
    dispatch(starts, lo, mid, depth + 1, out);
    out.push_str(&format!("{pad}}} else {{\n"));
    dispatch(starts, mid, hi, depth + 1, out);
    out.push_str(&format!("{pad}}}\n"));
}

fn specialized_cuts(program: &[u32]) -> String {
    let Some(cuts) = union_cuts(program) else {
        return FAR_CUTS.to_string();
    };
    if cuts.is_empty() {
        return FAR_CUTS.to_string();
    }
    let mut body = String::new();
    for (index, &cut) in cuts.iter().enumerate() {
        match straight_line(program, cut, &format!("loam_field_cut_{index}")) {
            Some(text) => body.push_str(&text),
            None => return FAR_CUTS.to_string(),
        }
    }
    let starts: Vec<usize> = cuts.iter().map(|cut| cut.0).collect();
    body.push_str("fn loam_field_cut(start: u32, p0: vec4<f32>) -> f32 {\n");
    dispatch(&starts, 0, starts.len(), 0, &mut body);
    body.push_str("    return LOAM_FIELD_FAR;\n}\n\n");
    body.push_str(
        "fn loam_field_leaf(node: LoamFieldNode, p: vec4<f32>) -> f32 {\n    \
         return loam_field_cut(node.start, p);\n}\n",
    );
    body
}

fn same_structure(previous: &[u32], next: &[u32]) -> bool {
    previous.len() == next.len()
        && previous
            .chunks_exact(2)
            .zip(next.chunks_exact(2))
            .all(|(was, now)| was[0] == now[0] && (was[0] == OP_SMOOTH_UNION || was[1] == now[1]))
}

/// Compiles a specialized module off the frame path; `take` hands it over once and `discard` drops a build the node no longer wants.
pub trait SpecializationBuilder: Send {
    fn submit(&mut self, revision: u64, request: SpecializationRequest);

    fn take(&mut self) -> Option<(u64, RenderPipeline)>;

    fn discard(&mut self);
}

pub struct SpecializationRequest {
    pub device: Device,
    pub layout: PipelineLayout,
    pub wgsl: String,
    pub surface_format: TextureFormat,
    pub depth: crate::DepthMode,
    pub sample_count: u32,
    pub entry: &'static str,
}

impl SpecializationRequest {
    pub fn build(&self) -> RenderPipeline {
        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("field march specialized"),
            source: ShaderSource::Wgsl(self.wgsl.as_str().into()),
        });
        pipeline_for(
            &self.device,
            &self.layout,
            &module,
            self.surface_format,
            self.depth,
            self.sample_count,
            self.entry,
        )
    }
}

/// Compiles inside `submit`, on the calling thread.
#[derive(Default)]
pub struct InlineBuilder {
    ready: Option<(u64, RenderPipeline)>,
}

impl SpecializationBuilder for InlineBuilder {
    fn submit(&mut self, revision: u64, request: SpecializationRequest) {
        self.ready = Some((revision, request.build()));
    }

    fn take(&mut self) -> Option<(u64, RenderPipeline)> {
        self.ready.take()
    }

    fn discard(&mut self) {
        self.ready = None;
    }
}

/// Boundaries a program's structure must survive unchanged before the node specializes it.
pub const DEFAULT_SPECIALIZE_AFTER: u32 = 8;

const PRIMITIVE_SIZE: u64 = std::mem::size_of::<FieldPrimitive>() as u64;
const NODE_SIZE: u64 = std::mem::size_of::<FieldNode>() as u64;
const INITIAL_PRIMITIVES: usize = 64;
const INITIAL_PROGRAM_WORDS: usize = 256;
const INITIAL_NODES: usize = 64;

fn storage(device: &Device, label: &'static str, bytes: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn grown(capacity: usize, needed: usize) -> usize {
    let mut capacity = capacity.max(1);
    while capacity < needed {
        capacity *= 2;
    }
    capacity
}

pub struct FieldMarchNode {
    device: Device,
    pipeline: RenderPipeline,
    specialized: Option<RenderPipeline>,
    pipeline_builds: u32,
    layout: BindGroupLayout,
    pipeline_layout: PipelineLayout,
    surface_format: TextureFormat,
    depth: crate::DepthMode,
    sample_count: u32,
    uniforms: FieldMarchUniforms,
    uniform_buf: Buffer,
    primitive_buf: Buffer,
    primitive_capacity: usize,
    program_buf: Buffer,
    program_capacity: usize,
    node_buf: Buffer,
    node_capacity: usize,
    bind_group: BindGroup,
    mode: MarchMode,
    clear_color: Color,
    has_depth: bool,
    counting: bool,
    words: Vec<u32>,
    seen: bool,
    structure_revision: u64,
    stable: u32,
    specialize_after: u32,
    pending: Option<u64>,
    builder: Box<dyn SpecializationBuilder>,
}

impl FieldMarchNode {
    pub fn new(
        device: &Device,
        surface_format: TextureFormat,
        depth: crate::DepthMode,
        sample_count: u32,
    ) -> Self {
        Self::build(device, surface_format, depth, sample_count, false)
    }

    /// A kernel that writes visit, skip, and evaluation counts to an `Rgba32Float` target instead of color; the shipped kernel carries no counters.
    pub fn counting(device: &Device, sample_count: u32) -> Self {
        Self::build(
            device,
            TextureFormat::Rgba32Float,
            crate::DepthMode::Off,
            sample_count,
            true,
        )
    }

    fn build(
        device: &Device,
        surface_format: TextureFormat,
        depth: crate::DepthMode,
        sample_count: u32,
        counting: bool,
    ) -> Self {
        let source = if counting {
            field_march_counting_wgsl()
        } else {
            field_march_wgsl()
        };
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("field march kernel"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("field march uniforms"),
            size: std::mem::size_of::<FieldMarchUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let storage_entry = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("field march bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1),
                storage_entry(2),
                storage_entry(3),
            ],
        });
        let primitive_buf = storage(
            device,
            "field march primitives",
            PRIMITIVE_SIZE * INITIAL_PRIMITIVES as u64,
        );
        let program_buf = storage(
            device,
            "field march program",
            4 * INITIAL_PROGRAM_WORDS as u64,
        );
        let node_buf = storage(
            device,
            "field march nodes",
            NODE_SIZE * INITIAL_NODES as u64,
        );
        let bind_group = bind(
            device,
            &layout,
            &uniform_buf,
            &primitive_buf,
            &program_buf,
            &node_buf,
        );
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("field march pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = pipeline_for(
            device,
            &pipeline_layout,
            &module,
            surface_format,
            depth,
            sample_count,
            entry_point(depth, counting),
        );
        Self {
            device: device.clone(),
            pipeline,
            specialized: None,
            pipeline_builds: 1,
            layout,
            pipeline_layout,
            surface_format,
            depth,
            sample_count,
            uniforms: FieldMarchUniforms::default(),
            uniform_buf,
            primitive_buf,
            primitive_capacity: INITIAL_PRIMITIVES,
            program_buf,
            program_capacity: INITIAL_PROGRAM_WORDS,
            node_buf,
            node_capacity: INITIAL_NODES,
            bind_group,
            mode: MarchMode::ExactDistance,
            clear_color: Color::BLACK,
            has_depth: depth.is_active(),
            counting,
            words: Vec::new(),
            seen: false,
            structure_revision: 0,
            stable: 0,
            specialize_after: DEFAULT_SPECIALIZE_AFTER,
            pending: None,
            builder: Box::new(InlineBuilder::default()),
        }
    }

    pub fn uniforms(&self) -> &FieldMarchUniforms {
        &self.uniforms
    }

    pub fn uniforms_mut(&mut self) -> &mut FieldMarchUniforms {
        &mut self.uniforms
    }

    pub fn set_clear_color(&mut self, color: Color) {
        self.clear_color = color;
    }

    pub fn march_mode(&self) -> MarchMode {
        self.mode
    }

    pub fn pipeline_builds(&self) -> u32 {
        self.pipeline_builds
    }

    pub fn program_capacity(&self) -> usize {
        self.program_capacity
    }

    pub fn node_capacity(&self) -> usize {
        self.node_capacity
    }

    pub fn is_specialized(&self) -> bool {
        self.specialized.is_some()
    }

    pub fn stable_boundaries(&self) -> u32 {
        self.stable
    }

    pub fn specialize_after(&mut self, boundaries: u32) {
        self.specialize_after = boundaries.max(1);
    }

    /// Drops any pending build with the old builder.
    pub fn set_specialization_builder(&mut self, builder: Box<dyn SpecializationBuilder>) {
        self.builder.discard();
        self.pending = None;
        self.builder = builder;
    }

    /// Grows the three storage buffers by doubling and rebinds without rebuilding the interpreter pipeline; a change to opcodes or operands, but not to a smooth-union radius, restarts specialization.
    pub fn set_program(&mut self, queue: &Queue, program: &FieldProgram) {
        let mut rebind = false;
        if program.primitives.len() > self.primitive_capacity {
            self.primitive_capacity = grown(self.primitive_capacity, program.primitives.len());
            self.primitive_buf = storage(
                &self.device,
                "field march primitives",
                PRIMITIVE_SIZE * self.primitive_capacity as u64,
            );
            rebind = true;
        }
        if program.program.len() > self.program_capacity {
            self.program_capacity = grown(self.program_capacity, program.program.len());
            self.program_buf = storage(
                &self.device,
                "field march program",
                4 * self.program_capacity as u64,
            );
            rebind = true;
        }
        if program.nodes.len() > self.node_capacity {
            self.node_capacity = grown(self.node_capacity, program.nodes.len());
            self.node_buf = storage(
                &self.device,
                "field march nodes",
                NODE_SIZE * self.node_capacity as u64,
            );
            rebind = true;
        }
        if rebind {
            self.bind_group = bind(
                &self.device,
                &self.layout,
                &self.uniform_buf,
                &self.primitive_buf,
                &self.program_buf,
                &self.node_buf,
            );
        }
        if !program.primitives.is_empty() {
            queue.write_buffer(
                &self.primitive_buf,
                0,
                bytemuck::cast_slice(&program.primitives),
            );
        }
        if !program.program.is_empty() {
            queue.write_buffer(&self.program_buf, 0, bytemuck::cast_slice(&program.program));
        }
        if !program.nodes.is_empty() {
            queue.write_buffer(&self.node_buf, 0, bytemuck::cast_slice(&program.nodes));
        }

        if !self.seen || self.words != program.program {
            if !self.seen || !same_structure(&self.words, &program.program) {
                self.structure_revision += 1;
                self.stable = 0;
                self.specialized = None;
                self.pending = None;
                self.builder.discard();
            }
            self.seen = true;
            self.words.clear();
            self.words.extend_from_slice(&program.program);
        }

        self.mode = MarchMode::of(program.kind);
        self.uniforms.program_len = program.program.len() as u32;
        self.uniforms.node_len = program.nodes.len() as u32;
        self.uniforms.kind = self.mode.code();
        self.flush_uniforms(queue);
    }

    /// Call once per frame; at `specialize_after` stable boundaries it submits a build and swaps the result in only while the structure still matches.
    pub fn boundary(&mut self) {
        if self.stable < self.specialize_after {
            self.stable += 1;
            if self.stable == self.specialize_after
                && self.specialized.is_none()
                && self.pending.is_none()
            {
                let request = SpecializationRequest {
                    device: self.device.clone(),
                    layout: self.pipeline_layout.clone(),
                    wgsl: specialized_module(&self.words, self.counting),
                    surface_format: self.surface_format,
                    depth: self.depth,
                    sample_count: self.sample_count,
                    entry: entry_point(self.depth, self.counting),
                };
                self.pending = Some(self.structure_revision);
                self.builder.submit(self.structure_revision, request);
            }
        }
        let Some((revision, pipeline)) = self.builder.take() else {
            return;
        };
        self.pending = None;
        if revision != self.structure_revision {
            return;
        }
        self.specialized = Some(pipeline);
    }

    pub fn flush_uniforms(&self, queue: &Queue) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));
    }

    pub fn record(
        &self,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        depth_view: Option<&TextureView>,
        viewport: crate::Viewport,
    ) {
        match (self.has_depth, depth_view.is_some()) {
            (true, false) => panic!(
                "FieldMarchNode::record: the pipeline has a depth format but no depth view was given"
            ),
            (false, true) => panic!(
                "FieldMarchNode::record: the pipeline has no depth format but a depth view was given"
            ),
            _ => {}
        }
        let depth_stencil_attachment = depth_view.map(|dv| RenderPassDepthStencilAttachment {
            view: dv,
            depth_ops: Some(Operations {
                load: LoadOp::Clear(crate::view::DEPTH_CLEAR),
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        });
        let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("field march pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(self.clear_color),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        viewport.apply(&mut rp);
        rp.set_pipeline(self.specialized.as_ref().unwrap_or(&self.pipeline));
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.draw(0..3, 0..1);
    }
}

fn entry_point(depth: crate::DepthMode, counting: bool) -> &'static str {
    if counting {
        "fs_counts"
    } else if depth.is_active() {
        "fs_depth"
    } else {
        "fs_main"
    }
}

fn pipeline_for(
    device: &Device,
    layout: &PipelineLayout,
    module: &ShaderModule,
    surface_format: TextureFormat,
    depth: crate::DepthMode,
    sample_count: u32,
    entry: &str,
) -> RenderPipeline {
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("field march pipeline"),
        layout: Some(layout),
        vertex: VertexState {
            module,
            entry_point: Some("vs_fullscreen"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(FragmentState {
            module,
            entry_point: Some(entry),
            targets: &[Some(ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: depth.format().map(|format| DepthStencilState {
            format,
            depth_write_enabled: depth.writes(),
            depth_compare: crate::view::DEPTH_COMPARE,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        multiview: None,
        cache: None,
    })
}

fn bind(
    device: &Device,
    layout: &BindGroupLayout,
    uniforms: &Buffer,
    primitives: &Buffer,
    program: &Buffer,
    nodes: &Buffer,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("field march bg"),
        layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: primitives.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: program.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: nodes.as_entire_binding(),
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use super::*;

    fn identity_frame() -> [[f32; 4]; 4] {
        let mut frame = [[0.0; 4]; 4];
        for (axis, column) in frame.iter_mut().enumerate() {
            column[axis] = 1.0;
        }
        frame
    }

    fn sphere_program(centers: &[[f32; 4]], radius: f32) -> FieldProgram {
        let mut program = FieldProgram {
            primitives: centers
                .iter()
                .map(|&translation| FieldPrimitive {
                    frame: identity_frame(),
                    translation,
                    params: [radius, 0.0, 0.0, 0.0],
                })
                .collect(),
            program: Vec::new(),
            nodes: Vec::new(),
            stack: 2,
            kind: FieldKind::ExactDistance,
        };
        for index in 0..centers.len() as u32 {
            program.program.extend([OP_SPHERE, index]);
            if index > 0 {
                program.program.extend([OP_UNION, 0]);
            }
        }
        program
    }

    #[test]
    fn a_structural_edit_uploads_a_program_without_building_a_pipeline() {
        let gpu = crate::device::noop_context();
        let mut node = FieldMarchNode::new(
            &gpu.device,
            TextureFormat::Rgba8Unorm,
            crate::DepthMode::Off,
            1,
        );
        assert_eq!(node.pipeline_builds(), 1);
        node.set_program(&gpu.queue, &sphere_program(&[[0.0, 0.0, -3.0, 0.0]], 0.5));
        let capacity = node.program_capacity();

        let wide: Vec<[f32; 4]> = (0..400).map(|i| [i as f32, 0.0, -3.0, 0.0]).collect();
        node.set_program(&gpu.queue, &sphere_program(&wide, 0.5));
        assert!(
            node.program_capacity() > capacity,
            "the wider program should have grown the storage buffer"
        );
        assert_eq!(
            node.pipeline_builds(),
            1,
            "a structural edit must not recompile the pipeline"
        );
    }

    #[test]
    fn a_wider_hierarchy_grows_the_node_buffer_without_building_a_pipeline() {
        let gpu = crate::device::noop_context();
        let mut node = FieldMarchNode::new(
            &gpu.device,
            TextureFormat::Rgba8Unorm,
            crate::DepthMode::Off,
            1,
        );
        let mut program = sphere_program(&[[0.0, 0.0, -3.0, 0.0]], 0.5);
        program.nodes = (0..300)
            .map(|i| FieldNode {
                center: [i as f32, 0.0, 0.0, 0.0],
                radius: 0.5,
                start: 0,
                end: 2,
                escape: i + 1,
            })
            .collect();
        let capacity = node.node_capacity();
        node.set_program(&gpu.queue, &program);
        assert!(
            node.node_capacity() > capacity,
            "300 nodes should have grown the hierarchy buffer past {capacity}"
        );
        assert_eq!(node.uniforms().node_len, 300);
        assert_eq!(node.pipeline_builds(), 1);
    }

    #[test]
    fn an_implicit_program_marches_with_the_fixed_step() {
        let gpu = crate::device::noop_context();
        let mut node = FieldMarchNode::new(
            &gpu.device,
            TextureFormat::Rgba8Unorm,
            crate::DepthMode::Off,
            1,
        );
        let mut program = sphere_program(&[[0.0, 0.0, -3.0, 0.0]], 0.5);
        program.kind = FieldKind::Implicit;
        node.set_program(&gpu.queue, &program);
        assert_eq!(node.march_mode(), MarchMode::FixedStep);
        assert_eq!(node.uniforms().kind, MarchMode::FixedStep.code());
    }

    #[test]
    fn a_stable_program_specializes_at_the_configured_boundary_and_not_before() {
        let gpu = crate::device::noop_context();
        let mut node = FieldMarchNode::new(
            &gpu.device,
            TextureFormat::Rgba8Unorm,
            crate::DepthMode::Off,
            1,
        );
        node.specialize_after(3);
        node.set_program(
            &gpu.queue,
            &sphere_program(&[[0.0, 0.0, -3.0, 0.0], [1.0, 0.0, -3.0, 0.0]], 0.5),
        );
        for boundary in 1..3 {
            node.boundary();
            assert!(
                !node.is_specialized(),
                "specialized after {boundary} of 3 boundaries"
            );
        }
        node.boundary();
        assert!(node.is_specialized(), "not specialized after 3 boundaries");
    }

    #[test]
    fn a_primitive_value_edit_does_not_reset_the_stability_count() {
        let gpu = crate::device::noop_context();
        let mut node = FieldMarchNode::new(
            &gpu.device,
            TextureFormat::Rgba8Unorm,
            crate::DepthMode::Off,
            1,
        );
        let centers = [[0.0, 0.0, -3.0, 0.0], [1.0, 0.0, -3.0, 0.0]];
        let mut program = sphere_program(&centers, 0.5);
        program.program = vec![
            OP_SPHERE,
            0,
            OP_SPHERE,
            1,
            OP_SMOOTH_UNION,
            0.25f32.to_bits(),
        ];
        node.set_program(&gpu.queue, &program);
        for _ in 0..4 {
            node.boundary();
        }
        assert_eq!(node.stable_boundaries(), 4);

        let mut moved = program.clone();
        moved.primitives[0].params[0] = 0.75;
        moved.primitives[1].translation = [4.0, 0.0, -3.0, 0.0];
        moved.program.pop();
        moved.program.push(0.9f32.to_bits());
        node.set_program(&gpu.queue, &moved);
        assert_eq!(
            node.stable_boundaries(),
            4,
            "a radius, a pose, and a blend radius are values, not structure"
        );
    }

    #[derive(Default)]
    struct Deferred {
        held: Option<(u64, RenderPipeline)>,
        release: Arc<AtomicBool>,
    }

    impl SpecializationBuilder for Deferred {
        fn submit(&mut self, revision: u64, request: SpecializationRequest) {
            self.held = Some((revision, request.build()));
        }

        fn take(&mut self) -> Option<(u64, RenderPipeline)> {
            if self.release.load(Ordering::Relaxed) {
                self.held.take()
            } else {
                None
            }
        }

        fn discard(&mut self) {}
    }

    #[test]
    fn a_finished_build_swaps_in_without_the_boundary_building_a_pipeline() {
        let gpu = crate::device::noop_context();
        let mut node = FieldMarchNode::new(
            &gpu.device,
            TextureFormat::Rgba8Unorm,
            crate::DepthMode::Off,
            1,
        );
        let release = Arc::new(AtomicBool::new(false));
        node.set_specialization_builder(Box::new(Deferred {
            held: None,
            release: release.clone(),
        }));
        node.specialize_after(1);
        node.set_program(
            &gpu.queue,
            &sphere_program(&[[0.0, 0.0, -3.0, 0.0], [1.0, 0.0, -3.0, 0.0]], 0.5),
        );
        node.boundary();
        release.store(true, Ordering::Relaxed);
        node.boundary();
        assert!(node.is_specialized(), "the finished build never swapped in");
        assert_eq!(
            node.pipeline_builds(),
            1,
            "the boundary built a pipeline instead of taking the finished one"
        );
    }

    #[test]
    fn a_structural_edit_during_a_pending_build_discards_it_and_keeps_the_interpreter() {
        let gpu = crate::device::noop_context();
        let mut node = FieldMarchNode::new(
            &gpu.device,
            TextureFormat::Rgba8Unorm,
            crate::DepthMode::Off,
            1,
        );
        let release = Arc::new(AtomicBool::new(false));
        node.set_specialization_builder(Box::new(Deferred {
            held: None,
            release: release.clone(),
        }));
        node.specialize_after(2);
        node.set_program(&gpu.queue, &sphere_program(&[[0.0, 0.0, -3.0, 0.0]], 0.5));
        node.boundary();
        node.boundary();
        assert!(!node.is_specialized(), "the build has not completed yet");

        node.set_program(
            &gpu.queue,
            &sphere_program(&[[0.0, 0.0, -3.0, 0.0], [2.0, 0.0, -3.0, 0.0]], 0.5),
        );
        release.store(true, Ordering::Relaxed);
        node.boundary();
        assert!(
            !node.is_specialized(),
            "a build from the edited-away program was swapped in"
        );
        assert_eq!(node.pipeline_builds(), 1);
    }

    const PROBE_SIZE: u32 = 64;
    const PROBE_NEAR: f32 = 0.05;
    const PROBE_CENTER: [f32; 4] = [0.0, 0.0, -3.0, 0.0];
    const PROBE_RADIUS: f32 = 0.5;

    fn probe_ray() -> glam::Vec3 {
        let pixel = PROBE_SIZE / 2;
        let uv = |v: u32| ((v as f32 + 0.5) / PROBE_SIZE as f32) * 2.0 - 1.0;
        let tan = FieldMarchUniforms::default().fov_y_tan;
        glam::Vec3::new(uv(pixel) * tan, -uv(pixel) * tan, -1.0).normalize()
    }

    fn probe_expected() -> f32 {
        let center = glam::Vec3::new(PROBE_CENTER[0], PROBE_CENTER[1], PROBE_CENTER[2]);
        let rd = probe_ray();
        let along = rd.dot(center);
        let gap = along * along - center.length_squared() + PROBE_RADIUS * PROBE_RADIUS;
        assert!(gap > 0.0, "the probe ray misses the sphere");
        crate::view::projective_depth(rd * (along - gap.sqrt()), PROBE_NEAR)
    }

    fn probe_depth(device: &Device, queue: &Queue, node: &FieldMarchNode) -> f32 {
        let size = Extent3d {
            width: PROBE_SIZE,
            height: PROBE_SIZE,
            depth_or_array_layers: 1,
        };
        let attachment = |format: TextureFormat, label: &'static str| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let color = attachment(TextureFormat::Rgba8Unorm, "field march probe color");
        let depth = attachment(crate::view::DEPTH_FORMAT, "field march probe depth");
        let color_view = color.create_view(&TextureViewDescriptor::default());
        let depth_view = depth.create_view(&TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&Default::default());
        node.record(
            &mut encoder,
            &color_view,
            Some(&depth_view),
            crate::Viewport::full([PROBE_SIZE, PROBE_SIZE]),
        );
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("field march probe readback"),
            size: (PROBE_SIZE * PROBE_SIZE * 4) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &depth,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::DepthOnly,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(PROBE_SIZE * 4),
                    rows_per_image: None,
                },
            },
            size,
        );
        queue.submit(Some(encoder.finish()));
        readback.slice(..).map_async(MapMode::Read, |_| {});
        device
            .poll(PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("readback poll");
        let depths =
            bytemuck::cast_slice::<u8, f32>(&readback.slice(..).get_mapped_range()).to_vec();
        let center = PROBE_SIZE / 2;
        depths[(center * PROBE_SIZE + center) as usize]
    }

    fn probe_node(device: &Device, queue: &Queue, program: &FieldProgram) -> FieldMarchNode {
        let mut node = FieldMarchNode::new(
            device,
            TextureFormat::Rgba8Unorm,
            crate::DepthMode::ReadWrite {
                format: crate::view::DEPTH_FORMAT,
            },
            1,
        );
        node.uniforms_mut().resolution = [PROBE_SIZE as f32; 2];
        node.uniforms_mut().near = PROBE_NEAR;
        node.set_program(queue, program);
        node
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run with --include-ignored"]
    fn an_interpreted_hit_writes_the_root_eyes_projective_depth_gpu_probe() {
        let (device, queue) = pollster::block_on(request_adapter_device());
        let node = probe_node(
            &device,
            &queue,
            &sphere_program(&[PROBE_CENTER], PROBE_RADIUS),
        );
        let written = probe_depth(&device, &queue, &node);
        let expected = probe_expected();
        assert!(
            (written - expected).abs() <= 5.0e-5,
            "the interpreted hit wrote depth {written}, not the projective depth {expected}"
        );
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run with --include-ignored"]
    fn the_gpu_hierarchy_reaches_the_same_hit_as_the_unculled_program_gpu_probe() {
        let (device, queue) = pollster::block_on(request_adapter_device());
        let centers: Vec<[f32; 4]> = [[1.0, 0.0, -3.0, 0.0], PROBE_CENTER]
            .into_iter()
            .chain((0..64).map(|i| {
                let t = i as f32;
                [
                    (t * 0.37).sin() * 30.0,
                    (t * 0.71).cos() * 30.0,
                    -3.0 - t * 0.5,
                    0.0,
                ]
            }))
            .collect();
        let unculled = sphere_program(&centers, PROBE_RADIUS);
        let mut culled = unculled.clone();
        culled.nodes = leaf_hierarchy(&centers, PROBE_RADIUS);

        let flat = probe_depth(&device, &queue, &probe_node(&device, &queue, &unculled));
        let hierarchy = probe_depth(&device, &queue, &probe_node(&device, &queue, &culled));
        assert!(
            (flat - hierarchy).abs() <= 1.0e-4,
            "the hierarchy marched to depth {hierarchy}, the unculled program to {flat}"
        );
    }

    fn leaf_hierarchy(centers: &[[f32; 4]], radius: f32) -> Vec<FieldNode> {
        centers
            .iter()
            .enumerate()
            .map(|(i, &center)| {
                let start = if i == 0 { 0 } else { (i * 4 - 2) as u32 };
                FieldNode {
                    center,
                    radius,
                    start,
                    end: start + 2,
                    escape: i as u32 + 1,
                }
            })
            .collect()
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run with --include-ignored"]
    fn the_specialized_kernel_and_the_interpreter_agree_on_a_hit_depth_gpu_probe() {
        let (device, queue) = pollster::block_on(request_adapter_device());
        let program = sphere_program(
            &[PROBE_CENTER, [1.4, 0.2, -3.4, 0.0], [-1.1, -0.3, -2.6, 0.0]],
            PROBE_RADIUS,
        );
        let mut node = probe_node(&device, &queue, &program);
        let interpreted = probe_depth(&device, &queue, &node);

        node.specialize_after(1);
        node.boundary();
        assert!(node.is_specialized(), "the node did not specialize");
        let specialized = probe_depth(&device, &queue, &node);
        assert!(
            (interpreted - specialized).abs() <= 1.0e-4,
            "the specialized kernel wrote depth {specialized}, the interpreter {interpreted}"
        );
    }

    async fn request_adapter_device() -> (Device, Queue) {
        let instance = Instance::default();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("wgpu adapter");
        adapter
            .request_device(&DeviceDescriptor {
                label: Some("field-march probe"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                memory_hints: MemoryHints::default(),
                trace: Trace::Off,
                experimental_features: Default::default(),
            })
            .await
            .expect("wgpu device")
    }
}
