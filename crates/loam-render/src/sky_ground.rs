use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::*;

/// Shared with [`crate::raymarch::HYPERSLICE_KERNEL_WGSL`].
pub const SKY_GROUND_WGSL: &str = include_str!("sky_ground.wgsl");

/// Linear light; `below` is the color at ray y = -1, `above` at ray y = +1, mixed linearly in y.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sky {
    pub below: [f32; 3],
    pub above: [f32; 3],
}

pub const DEFAULT_SKY: Sky = Sky {
    below: [0.04, 0.05, 0.10],
    above: [0.10, 0.13, 0.22],
};

impl Sky {
    /// The color at ray y = 0 with alpha 1.
    pub const fn horizon(self) -> Color {
        Color {
            r: 0.5 * (self.below[0] as f64 + self.above[0] as f64),
            g: 0.5 * (self.below[1] as f64 + self.above[1] as f64),
            b: 0.5 * (self.below[2] as f64 + self.above[2] as f64),
            a: 1.0,
        }
    }
}

pub const SKY_HORIZON: Color = DEFAULT_SKY.horizon();

/// Returns linear light.
pub fn linear_from_srgb8(rgb: [u8; 3]) -> [f32; 3] {
    // IEC 61966-2-1 sRGB electro-optical transfer function.
    rgb.map(|byte| {
        let c = f64::from(byte) / 255.0;
        let linear = if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        };
        linear as f32
    })
}

pub const GROUND_DARK_GREY: [f32; 3] = [0.18, 0.20, 0.24];
pub const GROUND_LIGHT_GREY: [f32; 3] = [0.30, 0.32, 0.36];

/// Half the sky is mixed in at `ln 2 / density` world units.
pub const DEFAULT_FOG_PER_UNIT: f32 = 0.0015;

#[derive(Copy, Clone, Debug)]
pub struct Ground {
    pub y: f32,
    pub dark: [f32; 3],
    pub light: [f32; 3],
    pub fog_per_unit: f32,
    /// Distance from the eye before any fog mixes in.
    pub fog_start: f32,
    pub visible: bool,
}

/// Bind group 0, binding 0; matches the node's WGSL `Uniforms`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SkyGroundUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub viewport_origin: [f32; 2],
    pub resolution: [f32; 2],
    pub ground_dark: [f32; 3],
    pub ground_y: f32,
    pub ground_light: [f32; 3],
    pub show_ground: f32,
    pub sky_below: [f32; 3],
    pub fog_per_unit: f32,
    pub sky_above: [f32; 3],
    pub fog_start: f32,
}

impl SkyGroundUniforms {
    /// `view_proj` must be the matrix the raster content over this pass is drawn with.
    pub fn new(view_proj: Mat4, viewport: crate::Viewport, sky: Sky, ground: Ground) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            viewport_origin: [viewport.x as f32, viewport.y as f32],
            resolution: viewport.resolution_f32(),
            ground_dark: ground.dark,
            ground_y: ground.y,
            ground_light: ground.light,
            show_ground: if ground.visible { 1.0 } else { 0.0 },
            sky_below: sky.below,
            fog_per_unit: ground.fog_per_unit,
            sky_above: sky.above,
            fog_start: ground.fog_start,
        }
    }

    fn sky(&self) -> Sky {
        Sky {
            below: self.sky_below,
            above: self.sky_above,
        }
    }
}

const SKY_GROUND_NODE_WGSL: &str = concat!(
    include_str!("sky_ground.wgsl"),
    r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    viewport_origin: vec2<f32>,
    resolution: vec2<f32>,
    ground_dark: vec3<f32>,
    ground_y: f32,
    ground_light: vec3<f32>,
    show_ground: f32,
    sky_below: vec3<f32>,
    fog_per_unit: f32,
    sky_above: vec3<f32>,
    fog_start: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

const HORIZON_EPS: f32 = 1.0e-6;

const LIGHT_DIR: vec3<f32> = vec3<f32>(0.5, 0.85, 0.3);
const AMBIENT: f32 = 0.20;
const DIFFUSE: f32 = 0.85;

struct Fragment {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn unproject(ndc: vec3<f32>) -> vec3<f32> {
    let h = u.inv_view_proj * vec4<f32>(ndc, 1.0);
    return h.xyz / h.w;
}

fn shade(frag_pos: vec4<f32>, near_ndc: f32, far_ndc: f32, background_depth: f32) -> Fragment {
    let uv = (frag_pos.xy - u.viewport_origin) / u.resolution;
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let near = unproject(vec3<f32>(ndc_xy, near_ndc));
    // The infinite reversed projection puts the far point at w = 0, so the direction is formed before the divide.
    let far = u.inv_view_proj * vec4<f32>(ndc_xy, far_ndc, 1.0);
    let rd = normalize(far.xyz - near * far.w);
    // Derivatives need uniform control flow, so the plane footprint is taken before the sky branch returns.
    let plane_t = (u.ground_y - near.y) / select(rd.y, HORIZON_EPS, abs(rd.y) <= HORIZON_EPS);
    let plane = (near + rd * plane_t).xz;
    let footprint = abs(dpdx(plane)) + abs(dpdy(plane));

    var out: Fragment;

    var t: f32 = 0.0;
    var hit = false;
    if (u.show_ground >= 0.5 && abs(rd.y) > HORIZON_EPS) {
        t = (u.ground_y - near.y) / rd.y;
        hit = t > 0.0;
    }

    if (!hit) {
        out.color = vec4<f32>(sky(rd, u.sky_below, u.sky_above), 1.0);
        out.depth = background_depth;
        return out;
    }

    let p_hit = near + rd * t;
    let fog = 1.0 - exp(-max(t - u.fog_start, 0.0) * u.fog_per_unit);
    let base = ground_color(p_hit, footprint, u.ground_dark, u.ground_light, fog);
    let lambert = max(dot(vec3<f32>(0.0, 1.0, 0.0), normalize(LIGHT_DIR)), 0.0);
    let lit = base * (AMBIENT + DIFFUSE * lambert);
    out.color = vec4<f32>(mix(lit, sky(rd, u.sky_below, u.sky_above), fog), 1.0);

    let clip = u.view_proj * vec4<f32>(p_hit, 1.0);
    out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> Fragment {
    return shade(frag_pos, 0.0, 1.0, 1.0);
}

@fragment
fn fs_reversed_z(@builtin(position) frag_pos: vec4<f32>) -> Fragment {
    return shade(frag_pos, 1.0, 0.0, 0.0);
}
"#
);

pub struct SkyGroundNode {
    pipeline: RenderPipeline,
    uniform_buf: Buffer,
    bind_group: BindGroup,
    clear: Color,
    depth_clear: f32,
}

impl SkyGroundNode {
    /// `depth_format` must match the attachments passed to [`Self::record`], which owns the frame's color and depth clear.
    pub fn new(
        device: &Device,
        target_format: TextureFormat,
        depth_format: TextureFormat,
        convention: crate::DepthConvention,
    ) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("sky_ground shader"),
            source: ShaderSource::Wgsl(SKY_GROUND_NODE_WGSL.into()),
        });

        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("sky_ground uniforms"),
            size: std::mem::size_of::<SkyGroundUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("sky_ground bgl"),
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
            label: Some("sky_ground bg"),
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("sky_ground pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("sky_ground pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some(match convention {
                    crate::DepthConvention::StandardZ => "fs_main",
                    crate::DepthConvention::ReversedZ => "fs_reversed_z",
                }),
                targets: &[Some(ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Always,
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
            uniform_buf,
            bind_group,
            clear: SKY_HORIZON,
            depth_clear: match convention {
                crate::DepthConvention::StandardZ => 1.0,
                crate::DepthConvention::ReversedZ => crate::view::DEPTH_CLEAR,
            },
        }
    }

    /// Also sets the color clear to the uniforms' horizon.
    pub fn set_uniforms(&mut self, queue: &Queue, uniforms: &SkyGroundUniforms) {
        self.clear = uniforms.sky().horizon();
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniforms));
    }

    /// Clears both attachments, so it records at the start of a stage the schedule lets clear; `viewport` restricts the shading, not the clear.
    pub fn record(
        &self,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        depth_view: &TextureView,
        viewport: Option<&crate::Viewport>,
    ) {
        let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("sky_ground pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(self.clear),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(self.depth_clear),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if let Some(vp) = viewport {
            vp.apply(&mut rp);
        }
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_validates_and_uniforms_match_host_layout() {
        let module = naga::front::wgsl::parse_str(SKY_GROUND_NODE_WGSL).expect("parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("sky shader validates");
        let ty = module
            .types
            .iter()
            .map(|(_, t)| t)
            .find(|t| t.name.as_deref() == Some("Uniforms"))
            .expect("the node's WGSL declares `Uniforms`");
        let naga::TypeInner::Struct { members, span } = &ty.inner else {
            panic!("`Uniforms` is not a struct");
        };
        assert_eq!(*span as usize, std::mem::size_of::<SkyGroundUniforms>());
        let rust_offsets = [
            (
                "view_proj",
                std::mem::offset_of!(SkyGroundUniforms, view_proj),
            ),
            (
                "inv_view_proj",
                std::mem::offset_of!(SkyGroundUniforms, inv_view_proj),
            ),
            (
                "viewport_origin",
                std::mem::offset_of!(SkyGroundUniforms, viewport_origin),
            ),
            (
                "resolution",
                std::mem::offset_of!(SkyGroundUniforms, resolution),
            ),
            (
                "ground_dark",
                std::mem::offset_of!(SkyGroundUniforms, ground_dark),
            ),
            (
                "ground_y",
                std::mem::offset_of!(SkyGroundUniforms, ground_y),
            ),
            (
                "ground_light",
                std::mem::offset_of!(SkyGroundUniforms, ground_light),
            ),
            (
                "show_ground",
                std::mem::offset_of!(SkyGroundUniforms, show_ground),
            ),
            (
                "sky_below",
                std::mem::offset_of!(SkyGroundUniforms, sky_below),
            ),
            (
                "fog_per_unit",
                std::mem::offset_of!(SkyGroundUniforms, fog_per_unit),
            ),
            (
                "sky_above",
                std::mem::offset_of!(SkyGroundUniforms, sky_above),
            ),
            (
                "fog_start",
                std::mem::offset_of!(SkyGroundUniforms, fog_start),
            ),
        ];
        assert_eq!(members.len(), rust_offsets.len());
        for (member, (name, offset)) in members.iter().zip(rust_offsets) {
            assert_eq!(member.name.as_deref(), Some(name));
            assert_eq!(member.offset as usize, offset, "offset of {name}");
        }
    }

    #[test]
    fn every_srgb_byte_survives_decode_and_the_targets_encode() {
        for byte in 0..=u8::MAX {
            let [linear, _, _] = linear_from_srgb8([byte; 3]);
            let c = f64::from(linear);
            let encoded = if c <= 0.003_130_8 {
                12.92 * c
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            };
            assert_eq!((encoded * 255.0).round() as u8, byte);
        }
    }
}
