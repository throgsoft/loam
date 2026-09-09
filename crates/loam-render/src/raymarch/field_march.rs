use bytemuck::{Pod, Zeroable};
use loam_runtime::domain::FieldKind;
use loam_runtime::field::{
    FieldPrimitive, MAX_POSE_DEPTH, MAX_STACK, OP_BOX, OP_HALFSPACE, OP_HALFSPACE4, OP_HYPERSPHERE,
    OP_INTERSECTION, OP_POP_POSE, OP_PUSH_POSE, OP_SMOOTH_UNION, OP_SPHERE, OP_SUBTRACTION,
    OP_UNION,
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
    pub _pad3: [u32; 2],
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
            _pad3: [0; 2],
        }
    }
}

pub fn field_march_wgsl() -> String {
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
         {body}",
        depth = include_str!("../shader/projective_depth.wgsl"),
        body = include_str!("field_march.wgsl"),
    )
}

const PRIMITIVE_SIZE: u64 = std::mem::size_of::<FieldPrimitive>() as u64;
const INITIAL_PRIMITIVES: usize = 64;
const INITIAL_PROGRAM_WORDS: usize = 256;

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
    pipeline_builds: u32,
    layout: BindGroupLayout,
    uniforms: FieldMarchUniforms,
    uniform_buf: Buffer,
    primitive_buf: Buffer,
    primitive_capacity: usize,
    program_buf: Buffer,
    program_capacity: usize,
    bind_group: BindGroup,
    mode: MarchMode,
    clear_color: Color,
    has_depth: bool,
}

impl FieldMarchNode {
    pub fn new(
        device: &Device,
        surface_format: TextureFormat,
        depth: crate::DepthMode,
        sample_count: u32,
    ) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("field march kernel"),
            source: ShaderSource::Wgsl(field_march_wgsl().into()),
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
        let bind_group = bind(device, &layout, &uniform_buf, &primitive_buf, &program_buf);
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("field march pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("field march pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some(if depth.is_active() {
                    "fs_depth"
                } else {
                    "fs_main"
                }),
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
        });
        Self {
            device: device.clone(),
            pipeline,
            pipeline_builds: 1,
            layout,
            uniforms: FieldMarchUniforms::default(),
            uniform_buf,
            primitive_buf,
            primitive_capacity: INITIAL_PRIMITIVES,
            program_buf,
            program_capacity: INITIAL_PROGRAM_WORDS,
            bind_group,
            mode: MarchMode::ExactDistance,
            clear_color: Color::BLACK,
            has_depth: depth.is_active(),
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

    /// Grows the storage buffers by doubling and rebinds; the pipeline is never rebuilt.
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
        if rebind {
            self.bind_group = bind(
                &self.device,
                &self.layout,
                &self.uniform_buf,
                &self.primitive_buf,
                &self.program_buf,
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
        self.mode = MarchMode::of(program.kind);
        self.uniforms.program_len = program.program.len() as u32;
        self.uniforms.kind = self.mode.code();
        self.flush_uniforms(queue);
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
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.draw(0..3, 0..1);
    }
}

fn bind(
    device: &Device,
    layout: &BindGroupLayout,
    uniforms: &Buffer,
    primitives: &Buffer,
    program: &Buffer,
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
        ],
    })
}

#[cfg(test)]
mod tests {
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

    #[test]
    #[ignore = "requires a working wgpu adapter; run with --include-ignored"]
    fn an_interpreted_hit_writes_the_root_eyes_projective_depth_gpu_probe() {
        let (device, queue) = pollster::block_on(request_adapter_device());
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

        let mut node = FieldMarchNode::new(
            &device,
            TextureFormat::Rgba8Unorm,
            crate::DepthMode::ReadWrite {
                format: crate::view::DEPTH_FORMAT,
            },
            1,
        );
        node.uniforms_mut().resolution = [PROBE_SIZE as f32; 2];
        node.uniforms_mut().near = PROBE_NEAR;
        node.set_program(&queue, &sphere_program(&[PROBE_CENTER], PROBE_RADIUS));

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
        let written = depths[(center * PROBE_SIZE + center) as usize];
        let expected = probe_expected();
        assert!(
            (written - expected).abs() <= 5.0e-5,
            "the interpreted hit wrote depth {written}, not the projective depth {expected}"
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
