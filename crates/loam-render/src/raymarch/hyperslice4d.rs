//! The user assembles the WGSL by concatenating [`HYPERSLICE_KERNEL_WGSL`]
//! with `Scene4::to_hyperslice_wgsl("u.w_slice")` (which defines
//! `loam_scene_sdf`).

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use loam_math::Rotor4;
use wgpu::*;

/// The uniform layout is fixed-size, so raising this is a recompile.
pub const MAX_BODIES: usize = 32;

/// Mirrored as `SHAPE_*` in [`HYPERSLICE_KERNEL_WGSL`]; keep in sync.
pub const SHAPE_PENTATOPE: u32 = 0;
pub const SHAPE_TESSERACT: u32 = 1;
pub const SHAPE_16CELL: u32 = 2;
pub const SHAPE_24CELL: u32 = 3;
pub const SHAPE_120CELL: u32 = 4;
pub const SHAPE_600CELL: u32 = 5;
pub const SHAPE_3SPHERE: u32 = 6;
pub const SHAPE_DUOCYLINDER: u32 = 7;
pub const SHAPE_CLIFFORD_TORUS: u32 = 8;
pub const SHAPE_SPHERINDER: u32 = 9;

/// 80 bytes, matching the kernel's `BodyUniform`; the rotor packs as two `vec4`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BodyUniform {
    pub position: [f32; 4],
    pub kind: f32,
    pub radius_or_shape: f32,
    pub polytope_size: f32,
    pub _pad0: f32,
    pub color: [f32; 3],
    pub _pad1: f32,
    pub rotor: [f32; 8],
}

impl Default for BodyUniform {
    fn default() -> Self {
        Self {
            position: [0.0; 4],
            kind: BodyKind::Invalid as i32 as f32,
            radius_or_shape: 0.0,
            polytope_size: 0.0,
            _pad0: 0.0,
            color: [0.7, 0.7, 0.7],
            _pad1: 0.0,
            rotor: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    }
}

/// Written as `f32`; the shader reads it back as `u32`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BodyKind {
    Sphere = 0,
    Polytope = 1,
    Invalid = 255,
}

impl BodyUniform {
    pub fn sphere(position: [f32; 4], radius: f32, color: [f32; 3]) -> Self {
        Self {
            position,
            kind: BodyKind::Sphere as i32 as f32,
            radius_or_shape: radius,
            color,
            ..Self::default()
        }
    }

    pub fn polytope(
        position: [f32; 4],
        shape_index: u32,
        size: f32,
        rotor: [f32; 8],
        color: [f32; 3],
    ) -> Self {
        Self {
            position,
            kind: BodyKind::Polytope as i32 as f32,
            radius_or_shape: shape_index as f32,
            polytope_size: size,
            color,
            rotor,
            ..Self::default()
        }
    }

    pub fn polytope_with_rotor(
        position: [f32; 4],
        shape_index: u32,
        size: f32,
        rotor: Rotor4,
        color: [f32; 3],
    ) -> Self {
        Self::polytope(position, shape_index, size, rotor.into(), color)
    }
}

/// Bind group 0, binding 0; matches the kernel's `Uniforms`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Hyperslice4DUniforms {
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
    pub w_slice: f32,
    pub body_count: f32,
    /// Framebuffer pixel of the viewport's top-left.
    pub viewport_origin: [f32; 2],
    pub params: [f32; 4],
    pub bodies: [BodyUniform; MAX_BODIES],
}

impl Default for Hyperslice4DUniforms {
    fn default() -> Self {
        Self {
            camera_pos: [0.0, 0.0, 5.0],
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
            w_slice: 0.0,
            body_count: 0.0,
            viewport_origin: [0.0, 0.0],
            params: [0.0; 4],
            bodies: [BodyUniform::default(); MAX_BODIES],
        }
    }
}

/// Prefixed with [`crate::sky_ground::SKY_GROUND_WGSL`]; the user's `Scene4` emit supplies `loam_scene_sdf`.
pub const HYPERSLICE_KERNEL_WGSL: &str = concat!(
    include_str!("../sky_ground.wgsl"),
    r#"
const MAX_BODIES: u32 = 32u;

const BODY_KIND_SPHERE: u32 = 0u;
const BODY_KIND_POLYTOPE: u32 = 1u;

const SHAPE_PENTATOPE: u32 = 0u;
const SHAPE_TESSERACT: u32 = 1u;
const SHAPE_16CELL: u32 = 2u;
const SHAPE_24CELL: u32 = 3u;
const SHAPE_120CELL: u32 = 4u;
const SHAPE_600CELL: u32 = 5u;
const SHAPE_3SPHERE: u32 = 6u;
const SHAPE_DUOCYLINDER: u32 = 7u;
const SHAPE_CLIFFORD_TORUS: u32 = 8u;
const SHAPE_SPHERINDER: u32 = 9u;
const BODY_KIND_INVALID: u32 = 255u;

struct BodyUniform {
    position: vec4<f32>,
    kind: f32,
    radius_or_shape: f32,
    polytope_size: f32,
    _pad0: f32,
    color: vec3<f32>,
    _pad1: f32,
    // Two vec4 so the std140 stride matches Rust's `[f32; 8]`; order [s, xy, xz, xw, yz, yw, zw, xyzw].
    rotor_lo: vec4<f32>,
    rotor_hi: vec4<f32>,
};

struct Uniforms {
    camera_pos: vec3<f32>,
    camera_forward: vec3<f32>,
    camera_right: vec3<f32>,
    camera_up: vec3<f32>,
    fov_y_tan: f32,
    resolution: vec2<f32>,
    time: f32,
    tick: f32,
    w_slice: f32,
    body_count: f32,
    viewport_origin: vec2<f32>,
    params: vec4<f32>,
    bodies: array<BodyUniform, MAX_BODIES>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

fn body_sphere_sdf_4d(p4: vec4<f32>, b: BodyUniform) -> f32 {
    return length(p4 - b.position) - b.radius_or_shape;
}

fn rotor4_inverse_apply(rotor_lo: vec4<f32>, rotor_hi: vec4<f32>, v: vec4<f32>) -> vec4<f32> {
    let rs  = rotor_lo.x;
    let rxy = -rotor_lo.y;
    let rxz = -rotor_lo.z;
    let rxw = -rotor_lo.w;
    let ryz = -rotor_hi.x;
    let ryw = -rotor_hi.y;
    let rzw = -rotor_hi.z;
    let r_i = rotor_hi.w;

    let vx = v.x; let vy = v.y; let vz = v.z; let vw = v.w;

    let p1 = rs * vx - rxy * vy - rxz * vz - rxw * vw;
    let p2 = rs * vy + rxy * vx - ryz * vz - ryw * vw;
    let p3 = rs * vz + rxz * vx + ryz * vy - rzw * vw;
    let p4 = rs * vw + rxw * vx + ryw * vy + rzw * vz;

    let t123 = -rxy * vz + rxz * vy - ryz * vx + r_i * vw;
    let t124 = -rxy * vw + rxw * vy - ryw * vx - r_i * vz;
    let t134 = -rxz * vw + rxw * vz - rzw * vx + r_i * vy;
    let t234 = -ryz * vw + ryw * vz - rzw * vy - r_i * vx;

    let q1 = rs * p1 - rxy * p2 - rxz * p3 - rxw * p4
           - ryz * t123 - ryw * t124 - rzw * t134 + r_i * t234;
    let q2 = rs * p2 + rxy * p1 - ryz * p3 - ryw * p4
           + rxz * t123 + rxw * t124 - rzw * t234 - r_i * t134;
    let q3 = rs * p3 + rxz * p1 + ryz * p2 - rzw * p4
           - rxy * t123 + rxw * t134 + ryw * t234 + r_i * t124;
    let q4 = rs * p4 + rxw * p1 + ryw * p2 + rzw * p3
           - rxy * t124 - rxz * t134 - ryz * t234 - r_i * t123;

    return vec4<f32>(q1, q2, q3, q4);
}

fn pentatope_sdf_local(p: vec4<f32>) -> f32 {
    let t  = 0.55901699437;  // sqrt(5) / 4
    let n0 = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    let n1 = vec4<f32>(-t, -t, -t, 0.25);
    let n2 = vec4<f32>(-t,  t,  t, 0.25);
    let n3 = vec4<f32>( t, -t,  t, 0.25);
    let n4 = vec4<f32>( t,  t, -t, 0.25);
    let r = 0.25;
    var d: f32 = dot(n0, p) - r;
    d = max(d, dot(n1, p) - r);
    d = max(d, dot(n2, p) - r);
    d = max(d, dot(n3, p) - r);
    d = max(d, dot(n4, p) - r);
    return d;
}

fn tesseract_sdf_local(p: vec4<f32>) -> f32 {
    let q = abs(p) - vec4<f32>(0.5, 0.5, 0.5, 0.5);
    let outside = length(max(q, vec4<f32>(0.0, 0.0, 0.0, 0.0)));
    let inside = min(max(max(q.x, q.y), max(q.z, q.w)), 0.0);
    return outside + inside;
}

fn cell16_sdf_local(p: vec4<f32>) -> f32 {
    let q = abs(p);
    return (q.x + q.y + q.z + q.w - 1.0) * 0.5;
}

fn cell24_sdf_local(p: vec4<f32>) -> f32 {
    let inv_sqrt2: f32 = 0.70710678;
    let sqrt2:     f32 = 1.41421356;
    let q = abs(p);
    let tess  = max(max(q.x, q.y), max(q.z, q.w)) - inv_sqrt2;
    let cross = (q.x + q.y + q.z + q.w - sqrt2) * 0.5;
    return max(tess, cross);
}

fn sphere3_sdf_local(p: vec4<f32>) -> f32 {
    return length(p) - 1.0;
}

fn duocylinder_sdf_local(p: vec4<f32>) -> f32 {
    let r = 0.7071068;
    let dxy = length(p.xy) - r;
    let dzw = length(p.zw) - r;
    let outside = length(vec2<f32>(max(dxy, 0.0), max(dzw, 0.0)));
    let inside = min(max(dxy, dzw), 0.0);
    return outside + inside;
}

fn clifford_torus_sdf_local(p: vec4<f32>) -> f32 {
    let r1 = 0.5;
    let r2 = 0.5;
    let tube = 0.2;
    let q1 = length(p.xy) - r1;
    let q2 = length(p.zw) - r2;
    return length(vec2<f32>(q1, q2)) - tube;
}

fn spherinder_sdf_local(p: vec4<f32>) -> f32 {
    let r = 0.7071068;
    let h = 0.7071068;
    let dxyz = length(p.xyz) - r;
    let dw = abs(p.w) - h;
    let outside = length(vec2<f32>(max(dxyz, 0.0), max(dw, 0.0)));
    let inside = min(max(dxyz, dw), 0.0);
    return outside + inside;
}

// Outside 1.5× the circumradius the ball SDF is a Lipschitz-1 lower bound; skips the rotor inverse.
fn body_polytope_sdf_4d(p4: vec4<f32>, b: BodyUniform) -> f32 {
    let size = max(b.polytope_size, 1.0e-6);
    let world_v = p4 - b.position;
    let world_dist2 = dot(world_v, world_v);
    let bound = size * 1.5;
    if (world_dist2 > bound * bound) {
        return sqrt(world_dist2) - size;
    }
    let local_v = rotor4_inverse_apply(b.rotor_lo, b.rotor_hi, world_v);
    let unit_p = local_v / size;
    let shape = u32(b.radius_or_shape + 0.5);
    var d: f32 = 1.0e9;
    if (shape == SHAPE_PENTATOPE) {
        d = pentatope_sdf_local(unit_p);
    } else if (shape == SHAPE_TESSERACT) {
        d = tesseract_sdf_local(unit_p);
    } else if (shape == SHAPE_16CELL) {
        d = cell16_sdf_local(unit_p);
    } else if (shape == SHAPE_24CELL) {
        d = cell24_sdf_local(unit_p);
    } else if (shape == SHAPE_120CELL) {
        d = cell120_sdf_local(unit_p);
    } else if (shape == SHAPE_600CELL) {
        d = cell600_sdf_local(unit_p);
    } else if (shape == SHAPE_3SPHERE) {
        d = sphere3_sdf_local(unit_p);
    } else if (shape == SHAPE_DUOCYLINDER) {
        d = duocylinder_sdf_local(unit_p);
    } else if (shape == SHAPE_CLIFFORD_TORUS) {
        d = clifford_torus_sdf_local(unit_p);
    } else if (shape == SHAPE_SPHERINDER) {
        d = spherinder_sdf_local(unit_p);
    }
    return d * size;
}

fn loam_dynamic_bodies_sdf(p3: vec3<f32>) -> f32 {
    let p4 = vec4<f32>(p3, u.w_slice);
    let body_count = u32(u.body_count + 0.5);
    var sdf: f32 = 1.0e9;
    for (var i: u32 = 0u; i < body_count; i = i + 1u) {
        let b = u.bodies[i];
        let kind = u32(b.kind + 0.5);
        if (kind == BODY_KIND_SPHERE) {
            sdf = min(sdf, body_sphere_sdf_4d(p4, b));
        } else if (kind == BODY_KIND_POLYTOPE) {
            sdf = min(sdf, body_polytope_sdf_4d(p4, b));
        }
    }
    return sdf;
}

fn loam_body_sdf_at(p3: vec3<f32>, body_idx: u32) -> f32 {
    if (body_idx >= MAX_BODIES) { return 1.0e9; }
    let p4 = vec4<f32>(p3, u.w_slice);
    let b = u.bodies[body_idx];
    let kind = u32(b.kind + 0.5);
    if (kind == BODY_KIND_SPHERE) {
        return body_sphere_sdf_4d(p4, b);
    } else if (kind == BODY_KIND_POLYTOPE) {
        return body_polytope_sdf_4d(p4, b);
    }
    return 1.0e9;
}

struct HitInfo {
    dist: f32,
    body_idx: u32,
};

fn loam_total_sdf(p3: vec3<f32>) -> HitInfo {
    let scene_d = loam_scene_sdf(p3);

    let p4 = vec4<f32>(p3, u.w_slice);
    let body_count = u32(u.body_count + 0.5);
    var dyn_d: f32 = 1.0e9;
    var dyn_idx: u32 = MAX_BODIES;
    for (var i: u32 = 0u; i < body_count; i = i + 1u) {
        let b = u.bodies[i];
        let kind = u32(b.kind + 0.5);
        var d: f32 = 1.0e9;
        if (kind == BODY_KIND_SPHERE) {
            d = body_sphere_sdf_4d(p4, b);
        } else if (kind == BODY_KIND_POLYTOPE) {
            d = body_polytope_sdf_4d(p4, b);
        }
        if (d < dyn_d) {
            dyn_d = d;
            dyn_idx = i;
        }
    }

    if (scene_d <= dyn_d) {
        return HitInfo(scene_d, MAX_BODIES);
    }
    return HitInfo(dyn_d, dyn_idx);
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn estimate_normal(p: vec3<f32>, body_idx: u32) -> vec3<f32> {
    let h = 0.001;
    if (body_idx >= MAX_BODIES) {
        let dx = loam_scene_sdf(p + vec3<f32>(h, 0.0, 0.0))
               - loam_scene_sdf(p - vec3<f32>(h, 0.0, 0.0));
        let dy = loam_scene_sdf(p + vec3<f32>(0.0, h, 0.0))
               - loam_scene_sdf(p - vec3<f32>(0.0, h, 0.0));
        let dz = loam_scene_sdf(p + vec3<f32>(0.0, 0.0, h))
               - loam_scene_sdf(p - vec3<f32>(0.0, 0.0, h));
        return normalize(vec3<f32>(dx, dy, dz));
    }
    let dx = loam_body_sdf_at(p + vec3<f32>(h, 0.0, 0.0), body_idx)
           - loam_body_sdf_at(p - vec3<f32>(h, 0.0, 0.0), body_idx);
    let dy = loam_body_sdf_at(p + vec3<f32>(0.0, h, 0.0), body_idx)
           - loam_body_sdf_at(p - vec3<f32>(0.0, h, 0.0), body_idx);
    let dz = loam_body_sdf_at(p + vec3<f32>(0.0, 0.0, h), body_idx)
           - loam_body_sdf_at(p - vec3<f32>(0.0, 0.0, h), body_idx);
    return normalize(vec3<f32>(dx, dy, dz));
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = ((frag_pos.xy - u.viewport_origin) / u.resolution) * 2.0 - vec2<f32>(1.0, 1.0);
    let aspect = u.resolution.x / u.resolution.y;
    let ndc = vec2<f32>(uv.x * aspect, -uv.y);
    let rd = normalize(
        u.camera_forward
        + u.camera_right * (ndc.x * u.fov_y_tan)
        + u.camera_up    * (ndc.y * u.fov_y_tan)
    );
    let ro = u.camera_pos;

    var t: f32 = 0.0;
    let scene_max_t = loam_scene_max_t(ro, rd);
    let max_t = min(60.0, scene_max_t + 1.0);
    var hit = false;
    var hit_idx: u32 = MAX_BODIES + 1u;
    // Every composed SDF is a 1-Lipschitz lower bound, so the full step cannot tunnel.
    let hit_eps = 0.001;
    let min_step = 0.0001;
    for (var i: i32 = 0; i < 384; i = i + 1) {
        let p = ro + rd * t;
        let h = loam_total_sdf(p);
        if (h.dist < hit_eps) {
            hit = true;
            hit_idx = h.body_idx;
            break;
        }
        t = t + max(h.dist, min_step);
        if (t > max_t) { break; }
    }

    if (!hit) {
        discard;
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let p_hit = ro + rd * t;
    if (hit_idx >= MAX_BODIES) {
        if (loam_scene_at(p_hit).kind == LOAM_PRIM_HALFSPACE4D) {
            discard;
            return vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
    }
    let n = estimate_normal(p_hit, hit_idx);
    let light_dir = normalize(vec3<f32>(0.5, 0.85, 0.3));
    let lambert = max(dot(n, light_dir), 0.0);
    let ambient = 0.20;
    var base = vec3<f32>(0.65, 0.65, 0.72);
    if (hit_idx < MAX_BODIES) {
        base = u.bodies[hit_idx].color;
    }
    let lit = base * (ambient + lambert * 0.85);
    let fog = 1.0 - exp(-t * 0.05);
    let final_color = mix(lit, sky(rd), fog * 0.5);
    return vec4<f32>(final_color, 1.0);
}
"#
);

const HYPERSLICE_UNIFORMS_SIZE: u64 = std::mem::size_of::<Hyperslice4DUniforms>() as u64;

fn strip_cell_stride(min_uniform_buffer_offset_alignment: u64) -> u64 {
    HYPERSLICE_UNIFORMS_SIZE.div_ceil(min_uniform_buffer_offset_alignment)
        * min_uniform_buffer_offset_alignment
}

fn strip_cell_uniforms(
    base: &Hyperslice4DUniforms,
    viewport: crate::Viewport,
    w_slice: f32,
    body: &BodyUniform,
) -> Hyperslice4DUniforms {
    let mut cell = *base;
    cell.bodies[0] = *body;
    cell.body_count = 1.0;
    cell.w_slice = w_slice;
    cell.resolution = viewport.resolution_f32();
    cell.viewport_origin = [viewport.x as f32, viewport.y as f32];
    cell
}

struct StripCellUniforms {
    buffer: Buffer,
    bind_groups: Vec<BindGroup>,
    stride: u64,
}

impl StripCellUniforms {
    fn new(device: &Device, layout: &BindGroupLayout, cell_count: usize) -> Self {
        let stride = strip_cell_stride(device.limits().min_uniform_buffer_offset_alignment as u64);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("hyperslice4d strip cell uniforms"),
            size: stride * cell_count as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_groups = (0..cell_count)
            .map(|i| {
                device.create_bind_group(&BindGroupDescriptor {
                    label: Some("hyperslice4d strip cell bg"),
                    layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &buffer,
                            offset: i as u64 * stride,
                            size: BufferSize::new(HYPERSLICE_UNIFORMS_SIZE),
                        }),
                    }],
                })
            })
            .collect();
        Self {
            buffer,
            bind_groups,
            stride,
        }
    }
}

pub struct Hyperslice4DNode {
    pipeline: RenderPipeline,
    uniforms: Hyperslice4DUniforms,
    uniform_buf: Buffer,
    bind_group: BindGroup,
    bind_group_layout: BindGroupLayout,
    clear_color: Color,
    strip_cells: Option<StripCellUniforms>,
}

impl Hyperslice4DNode {
    pub fn new(
        device: &Device,
        surface_format: TextureFormat,
        module: &ShaderModule,
        sample_count: u32,
    ) -> Self {
        let uniform_buf = device.create_buffer(&BufferDescriptor {
            label: Some("hyperslice4d uniforms"),
            size: std::mem::size_of::<Hyperslice4DUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("hyperslice4d bgl"),
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
            label: Some("hyperslice4d bg"),
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("hyperslice4d pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("hyperslice4d pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module,
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
            depth_stencil: None,
            multisample: MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniforms: Hyperslice4DUniforms::default(),
            uniform_buf,
            bind_group,
            bind_group_layout: bgl,
            clear_color: crate::sky_ground::SKY_HORIZON,
            strip_cells: None,
        }
    }

    pub fn uniforms(&self) -> &Hyperslice4DUniforms {
        &self.uniforms
    }
    pub fn uniforms_mut(&mut self) -> &mut Hyperslice4DUniforms {
        &mut self.uniforms
    }

    pub fn set_uniforms(&mut self, queue: &Queue, uniforms: Hyperslice4DUniforms) {
        self.uniforms = uniforms;
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));
    }

    pub fn flush_uniforms(&self, queue: &Queue) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));
    }

    pub fn set_clear_color(&mut self, color: Color) {
        self.clear_color = color;
    }

    /// Does not auto-flush; pair with [`Self::flush_uniforms`].
    pub fn set_bodies(&mut self, bodies: &[BodyUniform]) {
        let n = bodies.len().min(MAX_BODIES);
        self.uniforms.bodies[..n].copy_from_slice(&bodies[..n]);
        self.uniforms.body_count = n as f32;
    }

    pub fn set_body(&mut self, index: usize, body: BodyUniform) {
        if index < MAX_BODIES {
            self.uniforms.bodies[index] = body;
        }
    }

    pub fn set_body_count(&mut self, count: usize) {
        self.uniforms.body_count = count.min(MAX_BODIES) as f32;
    }
}

impl Hyperslice4DNode {
    /// `LoadOp::Load`: the kernel discards unshaded pixels, so record a background ahead of it.
    pub fn record_in_viewport(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        viewport: crate::Viewport,
    ) {
        let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("hyperslice4d pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        viewport.apply(&mut rp);
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.draw(0..3, 0..1);
    }

    /// Each cell gets its own uniform image and bind group: `write_buffer` lands ahead of the whole command buffer, so a shared image would give every cell the last write.
    pub fn record_strip(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        view: &wgpu::TextureView,
        cells: &[(crate::Viewport, f32, BodyUniform)],
    ) -> Result<()> {
        let drawn = cells
            .iter()
            .filter(|(viewport, _, _)| viewport.width != 0 && viewport.height != 0)
            .count();
        if drawn == 0 {
            return Ok(());
        }
        if self
            .strip_cells
            .as_ref()
            .is_none_or(|strip| strip.bind_groups.len() < drawn)
        {
            self.strip_cells = Some(StripCellUniforms::new(
                device,
                &self.bind_group_layout,
                drawn,
            ));
        }
        let strip = self
            .strip_cells
            .as_ref()
            .context("missing strip uniform storage")?;

        {
            let upload_size = BufferSize::new(drawn as u64 * strip.stride)
                .context("empty strip uniform storage")?;
            let mut staging = queue
                .write_buffer_with(&strip.buffer, 0, upload_size)
                .context("mapping the filmstrip's per-cell uniform staging buffer")?;
            let mut slot = 0usize;
            for (viewport, w_slice, body) in cells {
                if viewport.width == 0 || viewport.height == 0 {
                    continue;
                }
                let cell = strip_cell_uniforms(&self.uniforms, *viewport, *w_slice, body);
                let start = slot * strip.stride as usize;
                staging[start..start + HYPERSLICE_UNIFORMS_SIZE as usize]
                    .copy_from_slice(bytemuck::bytes_of(&cell));
                slot += 1;
            }
        }

        {
            let mut rp = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("hyperslice4d strip pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(self.clear_color),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rp.set_pipeline(&self.pipeline);
            let mut slot = 0usize;
            for (viewport, _, _) in cells {
                if viewport.width == 0 || viewport.height == 0 {
                    continue;
                }
                viewport.apply(&mut rp);
                rp.set_bind_group(0, &strip.bind_groups[slot], &[]);
                rp.draw(0..3, 0..1);
                slot += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_validates_with_real_scene_union() {
        use glam::Vec4;
        use loam_scene::{Scene4, SceneNode4};

        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::new(-0.6, 0.7, -1.5, 0.0), 0.7)
                .union(SceneNode4::hypersphere(Vec4::new(0.6, 0.7, -1.5, 0.0), 0.7))
                .union(SceneNode4::hypersphere(Vec4::new(0.0, 1.0, 1.5, 0.0), 1.0))
                .union(SceneNode4::halfspace(Vec4::Y, 0.0)),
        );
        let scene_wgsl = scene.to_hyperslice_wgsl("u.w_slice");
        let polytope = super::super::polytope_data::polytope_extended_sdfs_wgsl();
        let source = format!("{HYPERSLICE_KERNEL_WGSL}\n{polytope}\n{scene_wgsl}");
        let module = naga::front::wgsl::parse_str(&source)
            .expect("hyperslice4d kernel + Scene4 emit should parse as WGSL");
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("hyperslice4d kernel + Scene4 emit should validate");
    }

    #[test]
    fn kernel_validates_with_gated_scene() {
        use glam::Vec4;
        use loam_scene::{Scene4, SceneNode4};

        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::new(0.0, 1.0, 0.0, 0.0), 0.5)
                .union(SceneNode4::halfspace(Vec4::Y, 0.0)),
        );
        let scene_wgsl = scene.to_hyperslice_wgsl_gated("u.w_slice", "u.params.x");
        let polytope = super::super::polytope_data::polytope_extended_sdfs_wgsl();
        let source = format!("{HYPERSLICE_KERNEL_WGSL}\n{polytope}\n{scene_wgsl}");
        let module = naga::front::wgsl::parse_str(&source)
            .expect("gated Scene4 emit should parse against the kernel");
        let flags = naga::valid::ValidationFlags::all();
        let caps = naga::valid::Capabilities::empty();
        naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .expect("gated Scene4 emit should validate against the kernel");
    }

    fn strip_probe_cells() -> Vec<(crate::Viewport, f32, BodyUniform)> {
        crate::Viewport::full([192, 64])
            .split_horizontal(3)
            .zip([0.0_f32, 0.6, 0.9])
            .map(|(viewport, w_slice)| {
                (
                    viewport,
                    w_slice,
                    BodyUniform::sphere([0.0; 4], 1.0, [1.0, 0.0, 0.0]),
                )
            })
            .collect()
    }

    #[test]
    fn strip_cell_stride_is_offset_legal_and_covers_one_image() {
        for alignment in [32_u64, 64, 128, 256] {
            let stride = strip_cell_stride(alignment);
            assert_eq!(stride % alignment, 0, "stride must be an aligned offset");
            assert!(
                stride >= HYPERSLICE_UNIFORMS_SIZE,
                "stride {stride} would overlap the next cell's image",
            );
            assert!(
                stride - HYPERSLICE_UNIFORMS_SIZE < alignment,
                "stride {stride} wastes a whole alignment unit",
            );
        }
    }

    const EMPTY_SCENE_STUB: &str = r#"
const LOAM_PRIM_HYPERSPHERE4D: u32 = 0u;
const LOAM_PRIM_HALFSPACE4D: u32 = 1u;
const LOAM_PRIM_OTHER: u32 = 255u;
struct LoamSceneHit { dist: f32, kind: u32 }
fn loam_scene_at(p: vec3<f32>) -> LoamSceneHit {
    return LoamSceneHit(1.0e9, LOAM_PRIM_OTHER);
}
fn loam_scene_sdf(p: vec3<f32>) -> f32 {
    return loam_scene_at(p).dist;
}
fn loam_scene_max_t(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    return 1.0e9;
}
"#;

    async fn request_device() -> Result<(Device, Queue), String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("request_adapter failed: {e}"))?;
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("hyperslice4d-strip"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            })
            .await
            .map_err(|e| format!("request_device failed: {e}"))
    }

    fn read_back_rgba(
        device: &Device,
        queue: &Queue,
        texture: &Texture,
        size: [u32; 2],
    ) -> Vec<u8> {
        let bytes_per_row = size[0] * 4;
        assert_eq!(
            bytes_per_row % COPY_BYTES_PER_ROW_ALIGNMENT,
            0,
            "probe width chosen so no row padding is needed"
        );
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("hyperslice4d strip readback"),
            size: (bytes_per_row * size[1]) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("hyperslice4d strip readback encoder"),
        });
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device
            .poll(PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("readback poll");
        let data = slice.get_mapped_range().to_vec();
        readback.unmap();
        data
    }

    fn strip_probe_module(device: &Device) -> ShaderModule {
        let polytope = super::super::polytope_data::polytope_stub_sdfs_wgsl();
        device.create_shader_module(ShaderModuleDescriptor {
            label: Some("hyperslice4d strip probe"),
            source: ShaderSource::Wgsl(
                format!("{HYPERSLICE_KERNEL_WGSL}\n{polytope}\n{EMPTY_SCENE_STUB}").into(),
            ),
        })
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run with --include-ignored"]
    fn each_strip_cell_renders_its_own_w_slice_gpu_probe() {
        const SIZE: [u32; 2] = [192, 64];
        let (device, queue) = pollster::block_on(request_device()).expect("wgpu device");

        let module = strip_probe_module(&device);
        let target = device.create_texture(&TextureDescriptor {
            label: Some("hyperslice4d strip probe target"),
            size: Extent3d {
                width: SIZE[0],
                height: SIZE[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&TextureViewDescriptor::default());

        let cells = strip_probe_cells();
        let mut node = Hyperslice4DNode::new(&device, TextureFormat::Rgba8Unorm, &module, 1);
        let mut encoder = device.create_command_encoder(&Default::default());
        node.record_strip(&device, &queue, &mut encoder, &view, &cells)
            .expect("filmstrip should render");
        queue.submit(Some(encoder.finish()));

        let pixels = read_back_rgba(&device, &queue, &target, SIZE);
        let footprints: Vec<usize> = cells
            .iter()
            .map(|(viewport, _, _)| {
                let mut covered = 0;
                for y in viewport.y..viewport.y + viewport.height {
                    for x in viewport.x..viewport.x + viewport.width {
                        let px = ((y * SIZE[0] + x) * 4) as usize;
                        if pixels[px] > pixels[px + 2] {
                            covered += 1;
                        }
                    }
                }
                covered
            })
            .collect();

        assert!(
            footprints[2] > 0,
            "the w = 0.9 slice should still show a body: {footprints:?}"
        );
        assert!(
            footprints[0] > footprints[1] && footprints[1] > footprints[2],
            "footprints should shrink with |w|; equal cells mean one uniform image \
             fed every cell: {footprints:?}"
        );
    }
}
