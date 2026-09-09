struct LoamFieldPrimitive {
    frame: mat4x4<f32>,
    translation: vec4<f32>,
    params: vec4<f32>,
};

struct LoamFieldNode {
    center: vec4<f32>,
    radius: f32,
    start: u32,
    end: u32,
    escape: u32,
};

struct FieldUniforms {
    camera_pos: vec3<f32>,
    camera_forward: vec3<f32>,
    camera_right: vec3<f32>,
    camera_up: vec3<f32>,
    fov_y_tan: f32,
    resolution: vec2<f32>,
    viewport_origin: vec2<f32>,
    params: vec4<f32>,
    near: f32,
    w_slice: f32,
    implicit_step: f32,
    max_t: f32,
    program_len: u32,
    kind: u32,
    node_len: u32,
    cull_tolerance: f32,
};

@group(0) @binding(0) var<uniform> u: FieldUniforms;
@group(0) @binding(1) var<storage, read> loam_field_prims: array<LoamFieldPrimitive>;
@group(0) @binding(2) var<storage, read> loam_field_prog: array<u32>;
@group(0) @binding(3) var<storage, read> loam_field_nodes: array<LoamFieldNode>;

var<private> loam_field_visits: u32 = 0u;
var<private> loam_field_skips: u32 = 0u;
var<private> loam_field_evals: u32 = 0u;

fn loam_field_local(prim: LoamFieldPrimitive, p: vec4<f32>) -> vec4<f32> {
    return transpose(prim.frame) * (p - prim.translation);
}

fn loam_field_primitive(op: u32, index: u32, p: vec4<f32>) -> f32 {
    if (LOAM_FIELD_COUNTING) {
        loam_field_evals = loam_field_evals + 1u;
    }
    let prim = loam_field_prims[index];
    let q = loam_field_local(prim, p);
    let r = prim.params;
    if (op == LOAM_OP_SPHERE) {
        return length(q.xyz) - r.x;
    }
    if (op == LOAM_OP_BOX) {
        let d = abs(q.xyz) - r.xyz;
        return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
    }
    if (op == LOAM_OP_HALFSPACE) {
        return dot(q.xyz, r.xyz);
    }
    if (op == LOAM_OP_HYPERSPHERE) {
        return length(q) - r.x;
    }
    if (op == LOAM_OP_HALFSPACE4) {
        return dot(q, r);
    }
    return LOAM_FIELD_FAR;
}

// Quilez, "Smooth minimum", iquilezles.org/articles/smin, polynomial form.
fn loam_field_smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
