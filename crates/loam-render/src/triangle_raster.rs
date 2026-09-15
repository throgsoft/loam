use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use loam_math::{Projection, RasterizableSpace};
use loam_shape::TriangleMesh;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutEntry, BindingType,
    BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, CompareFunction, Device, Features, Limits, LoadOp, Operations,
    PrimitiveTopology, Queue, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
    RenderPassDescriptor, RenderPipeline, ShaderStages, StoreOp, TextureFormat, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexStepMode,
};

use crate::device::{FeatureRequest, GpuContext, MissingGpuCapability};
use crate::material::MaterialSpec;

const TRIANGLE_RASTER_WGSL: &str = include_str!("triangle_raster.wgsl");

const CAMERA_BINDING: [BindGroupLayoutEntry; 1] = [BindGroupLayoutEntry {
    binding: 0,
    visibility: ShaderStages::VERTEX_FRAGMENT,
    ty: BindingType::Buffer {
        ty: BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    },
    count: None,
}];

/// Matches the WGSL `CameraUniform`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TriangleRasterUniforms {
    pub view_projection: [[f32; 4]; 4],
}

impl Default for TriangleRasterUniforms {
    fn default() -> Self {
        Self {
            view_projection: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }
}

/// Matches the vertex attribute layout: position at 0, color at 16.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
pub struct TriangleVertex {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub color: [f32; 4],
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum FragmentShading {
    #[default]
    Flat,
    FaceNormalLambert,
}

impl FragmentShading {
    fn entry_point(self) -> &'static str {
        match self {
            Self::Flat => "fs_flat",
            Self::FaceNormalLambert => "fs_lambert",
        }
    }
}

pub struct TriangleRasterNode {
    pipeline: RenderPipeline,
    uniform_buf: Buffer,
    bind_group: BindGroup,

    vertex_buf: Buffer,
    vertex_capacity: u32,

    index_buf: Buffer,
    index_capacity: u32,
    index_count: u32,

    has_depth: bool,
    vertices_scratch: Vec<TriangleVertex>,
}

impl TriangleRasterNode {
    pub fn new(
        gpu: &GpuContext,
        surface_format: TextureFormat,
        depth: crate::DepthMode,
        convention: crate::DepthConvention,
        shading: FragmentShading,
    ) -> Result<Self, MissingGpuCapability> {
        let device = &gpu.device;
        let vertex_attrs = [
            VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 16,
                shader_location: 1,
            },
        ];
        let material = MaterialSpec {
            label: "triangle_raster",
            shader: TRIANGLE_RASTER_WGSL,
            vertex_entry: "vs_main",
            fragment_entry: shading.entry_point(),
            vertex_layouts: &[VertexBufferLayout {
                array_stride: std::mem::size_of::<TriangleVertex>() as u64,
                step_mode: VertexStepMode::Vertex,
                attributes: &vertex_attrs,
            }],
            resources: &[&CAMERA_BINDING],
            features: FeatureRequest {
                required_features: Features::empty(),
                optional_features: Features::empty(),
                required_limits: Limits::downlevel_webgl2_defaults(),
            },
            topology: PrimitiveTopology::TriangleList,
            blend: Some(BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
            }),
            depth,
            convention,
            standard_z_compare: CompareFunction::Less,
        };
        let built = material.build(gpu, surface_format)?;

        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("triangle_raster uniforms"),
            size: std::mem::size_of::<TriangleRasterUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("triangle_raster bg"),
            layout: &built.pipeline.get_bind_group_layout(0),
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let vertex_buf = device.create_buffer(&BufferDescriptor {
            label: Some("triangle_raster vertex buffer"),
            size: 64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buf = device.create_buffer(&BufferDescriptor {
            label: Some("triangle_raster index buffer"),
            size: 64,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            pipeline: built.pipeline,
            uniform_buf,
            bind_group,
            vertex_buf,
            vertex_capacity: 0,
            index_buf,
            index_capacity: 0,
            index_count: 0,
            has_depth: depth.is_active(),
            vertices_scratch: Vec::new(),
        })
    }

    pub fn set_camera(&self, queue: &Queue, view_projection: Mat4) {
        let uniforms = TriangleRasterUniforms {
            view_projection: view_projection.to_cols_array_2d(),
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    pub fn upload<S, const N: usize>(
        &mut self,
        device: &Device,
        queue: &Queue,
        mesh: &TriangleMesh<N>,
        projection: &Projection<N>,
    ) where
        S: RasterizableSpace<N>,
    {
        let n_vertices = mesh.vertices.len();
        assert_eq!(
            mesh.colors.len(),
            n_vertices,
            "TriangleMesh invariant: colors.len() == vertices.len()"
        );
        self.vertices_scratch.clear();
        let verts = &mut self.vertices_scratch;
        for (v, color) in mesh.vertices.iter().zip(mesh.colors.iter()) {
            let p_native = S::array_to_point(*v);
            let p3 = S::project_point(p_native, projection).unwrap_or(glam::Vec3::NAN);
            verts.push(TriangleVertex {
                position: p3.to_array(),
                _pad0: 0.0,
                color: *color,
            });
        }

        let indices: &[u32] = bytemuck::cast_slice(&mesh.indices);

        if verts.len() as u32 > self.vertex_capacity {
            let new_cap = (verts.len() as u32).next_power_of_two().max(16);
            self.vertex_buf = device.create_buffer(&BufferDescriptor {
                label: Some("triangle_raster vertex buffer"),
                size: (new_cap as u64) * (std::mem::size_of::<TriangleVertex>() as u64),
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertex_capacity = new_cap;
        }
        if indices.len() as u32 > self.index_capacity {
            let new_cap = (indices.len() as u32).next_power_of_two().max(16);
            self.index_buf = device.create_buffer(&BufferDescriptor {
                label: Some("triangle_raster index buffer"),
                size: (new_cap as u64) * (std::mem::size_of::<u32>() as u64),
                usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.index_capacity = new_cap;
        }

        if !verts.is_empty() {
            queue.write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(verts));
        }
        if !indices.is_empty() {
            queue.write_buffer(&self.index_buf, 0, bytemuck::cast_slice(indices));
        }
        self.index_count = indices.len() as u32;
    }

    /// Loads both attachments; uploads between records affect every draw in the same submission.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: Option<&wgpu::TextureView>,
        viewport: Option<&crate::Viewport>,
    ) {
        match (self.has_depth, depth_view.is_some()) {
            (true, false) => {
                panic!(
                    "TriangleRasterNode::record: pipeline was created with a depth format but \
                     no depth view was provided"
                )
            }
            (false, true) => {
                panic!(
                    "TriangleRasterNode::record: pipeline was created without a depth format \
                     but a depth view was provided"
                )
            }
            _ => {}
        }
        if self.index_count == 0 {
            return;
        }
        let depth_attachment = depth_view.map(|dv| RenderPassDepthStencilAttachment {
            view: dv,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        });
        let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("triangle_raster pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if let Some(vp) = viewport {
            vp.apply(&mut rp);
        }
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.set_vertex_buffer(0, self.vertex_buf.slice(..));
        rp.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        rp.draw_indexed(0..self.index_count, 0, 0..1);
    }
}
