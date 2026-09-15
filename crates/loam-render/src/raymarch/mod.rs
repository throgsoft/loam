//! A user shader for [`RayMarchNode`] exports `vs_fullscreen(@builtin(vertex_index))` and
//! `fs_main(@builtin(position))` and binds [`RayMarchUniforms`] at group 0, binding 0.

mod hyperslice4d;
mod polytope_data;
pub mod scene;
pub use hyperslice4d::{
    BodyKind, BodyUniform, Hyperslice4DNode, Hyperslice4DUniforms, HYPERSLICE_KERNEL_WGSL,
    SHAPE_120CELL, SHAPE_16CELL, SHAPE_24CELL, SHAPE_3SPHERE, SHAPE_600CELL, SHAPE_CLIFFORD_TORUS,
    SHAPE_DUOCYLINDER, SHAPE_PENTATOPE, SHAPE_SPHERINDER, SHAPE_TESSERACT,
};
pub use polytope_data::{polytope_extended_sdfs_wgsl, polytope_stub_sdfs_wgsl};

/// `None` for the smooth-surface SDFs.
pub fn polytope4_from_shape_id(shape: u32) -> Option<loam_shape::polytope::Polytope4> {
    use loam_shape::polytope::Polytope4;
    Some(match shape {
        s if s == SHAPE_PENTATOPE => Polytope4::Pentatope,
        s if s == SHAPE_TESSERACT => Polytope4::Tesseract,
        s if s == SHAPE_16CELL => Polytope4::Cell16,
        s if s == SHAPE_24CELL => Polytope4::Cell24,
        s if s == SHAPE_120CELL => Polytope4::Cell120,
        s if s == SHAPE_600CELL => Polytope4::Cell600,
        _ => return None,
    })
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RaymarchShape {
    Polytope(loam_shape::polytope::Polytope4),
    ThreeSphere,
    Duocylinder,
    CliffordTorus,
    Spherinder,
}

impl RaymarchShape {
    pub fn shape_id(&self) -> u32 {
        use loam_shape::polytope::Polytope4;
        match self {
            RaymarchShape::Polytope(p) => match p {
                Polytope4::Pentatope => SHAPE_PENTATOPE,
                Polytope4::Tesseract => SHAPE_TESSERACT,
                Polytope4::Cell16 => SHAPE_16CELL,
                Polytope4::Cell24 => SHAPE_24CELL,
                Polytope4::Cell120 => SHAPE_120CELL,
                Polytope4::Cell600 => SHAPE_600CELL,
            },
            RaymarchShape::ThreeSphere => SHAPE_3SPHERE,
            RaymarchShape::Duocylinder => SHAPE_DUOCYLINDER,
            RaymarchShape::CliffordTorus => SHAPE_CLIFFORD_TORUS,
            RaymarchShape::Spherinder => SHAPE_SPHERINDER,
        }
    }

    pub fn polytope4(&self) -> Option<loam_shape::polytope::Polytope4> {
        match self {
            RaymarchShape::Polytope(p) => Some(*p),
            _ => None,
        }
    }
}

impl From<loam_shape::polytope::Polytope4> for RaymarchShape {
    fn from(p: loam_shape::polytope::Polytope4) -> Self {
        RaymarchShape::Polytope(p)
    }
}

impl From<RaymarchShape> for u32 {
    fn from(s: RaymarchShape) -> u32 {
        s.shape_id()
    }
}

use bytemuck::{Pod, Zeroable};
use wgpu::*;

/// Bind group 0, binding 0, `std140`: every `vec3` pads to 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RayMarchUniforms {
    pub camera_pos: [f32; 3],
    pub _pad0: f32,
    pub camera_forward: [f32; 3],
    pub _pad1: f32,
    pub camera_right: [f32; 3],
    pub _pad2: f32,
    pub camera_up: [f32; 3],
    pub fov_y_tan: f32,
    pub resolution: [f32; 2],
    pub time: f32,
    pub tick: f32,
    /// Four scalar knobs; semantics are up to the user shader.
    pub params: [f32; 4],
    /// Inverse of the eye's isometry on the domain's embedding; the view map composes it first.
    pub eye_inverse: [[f32; 4]; 4],
    /// Near distance of the root projection; see [`crate::view`].
    pub near: f32,
    pub _pad3: [f32; 3],
}

impl Default for RayMarchUniforms {
    fn default() -> Self {
        Self {
            camera_pos: [0.0, 0.0, 3.0],
            _pad0: 0.0,
            camera_forward: [0.0, 0.0, -1.0],
            _pad1: 0.0,
            camera_right: [1.0, 0.0, 0.0],
            _pad2: 0.0,
            camera_up: [0.0, 1.0, 0.0],
            fov_y_tan: (60.0_f32.to_radians() * 0.5).tan(),
            resolution: [1.0, 1.0],
            time: 0.0,
            tick: 0.0,
            params: [0.0; 4],
            eye_inverse: glam::Mat4::IDENTITY.to_cols_array_2d(),
            near: 0.05,
            _pad3: [0.0; 3],
        }
    }
}

pub struct RayMarchNode {
    pipeline: RenderPipeline,
    uniforms: RayMarchUniforms,
    uniform_buf: Buffer,
    bind_group: BindGroup,
    clear_color: Color,
    has_depth: bool,
}

impl RayMarchNode {
    pub fn new(device: &Device, surface_format: TextureFormat, shader: &ShaderModule) -> Self {
        Self::with_depth(device, surface_format, shader, crate::DepthMode::Off)
    }

    /// The user's `fs_main` writes [`crate::view`]'s projective depth as `frag_depth`.
    pub fn with_depth(
        device: &Device,
        surface_format: TextureFormat,
        shader: &ShaderModule,
        depth: crate::DepthMode,
    ) -> Self {
        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("raymarch uniforms"),
            size: std::mem::size_of::<RayMarchUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("raymarch bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("raymarch bg"),
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("raymarch pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("raymarch pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
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
                count: 1,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniforms: RayMarchUniforms::default(),
            uniform_buf,
            bind_group,
            clear_color: Color::BLACK,
            has_depth: depth.is_active(),
        }
    }

    pub fn uniforms(&self) -> &RayMarchUniforms {
        &self.uniforms
    }

    pub fn uniforms_mut(&mut self) -> &mut RayMarchUniforms {
        &mut self.uniforms
    }

    pub fn set_uniforms(&mut self, queue: &Queue, uniforms: RayMarchUniforms) {
        self.uniforms = uniforms;
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));
    }

    pub fn flush_uniforms(&self, queue: &Queue) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));
    }
}

impl RayMarchNode {
    /// Clears the whole attachment before shading the viewport.
    pub fn record_in_viewport(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        viewport: crate::Viewport,
    ) {
        self.record(encoder, view, None, viewport);
    }

    /// Clears both attachments; depth clears to [`crate::view::DEPTH_CLEAR`].
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        viewport: crate::Viewport,
    ) {
        match (self.has_depth, depth_view.is_some()) {
            (true, false) => panic!(
                "RayMarchNode::record: the pipeline has a depth format but no depth view was given"
            ),
            (false, true) => panic!(
                "RayMarchNode::record: the pipeline has no depth format but a depth view was given"
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
            label: Some("raymarch pass"),
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
