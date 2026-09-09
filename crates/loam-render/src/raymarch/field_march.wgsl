
struct LoamFieldPrimitive {
    frame: mat4x4<f32>,
    translation: vec4<f32>,
    params: vec4<f32>,
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
};

@group(0) @binding(0) var<uniform> u: FieldUniforms;
@group(0) @binding(1) var<storage, read> loam_field_prims: array<LoamFieldPrimitive>;
@group(0) @binding(2) var<storage, read> loam_field_prog: array<u32>;

fn loam_field_local(prim: LoamFieldPrimitive, p: vec4<f32>) -> vec4<f32> {
    return transpose(prim.frame) * (p - prim.translation);
}

fn loam_field_primitive(op: u32, index: u32, p: vec4<f32>) -> f32 {
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

fn loam_field_smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn loam_field_sdf(p3: vec3<f32>) -> f32 {
    var stack: array<f32, LOAM_MAX_STACK>;
    var points: array<vec4<f32>, LOAM_MAX_POSE_DEPTH>;
    var sp: u32 = 0u;
    var pp: u32 = 0u;
    var p = vec4<f32>(p3, u.w_slice);
    for (var i: u32 = 0u; i + 1u < u.program_len; i = i + 2u) {
        let op = loam_field_prog[i];
        let arg = loam_field_prog[i + 1u];
        if (op >= LOAM_OP_SPHERE && op <= LOAM_OP_HALFSPACE4) {
            if (sp >= LOAM_MAX_STACK) {
                return LOAM_FIELD_FAR;
            }
            stack[sp] = loam_field_primitive(op, arg, p);
            sp = sp + 1u;
        } else if (op == LOAM_OP_PUSH_POSE) {
            if (pp >= LOAM_MAX_POSE_DEPTH) {
                return LOAM_FIELD_FAR;
            }
            points[pp] = p;
            pp = pp + 1u;
            p = loam_field_local(loam_field_prims[arg], p);
        } else if (op == LOAM_OP_POP_POSE) {
            if (pp == 0u) {
                return LOAM_FIELD_FAR;
            }
            pp = pp - 1u;
            p = points[pp];
        } else {
            if (sp < 2u) {
                return LOAM_FIELD_FAR;
            }
            let b = stack[sp - 1u];
            let a = stack[sp - 2u];
            sp = sp - 1u;
            var value = min(a, b);
            if (op == LOAM_OP_INTERSECTION) {
                value = max(a, b);
            } else if (op == LOAM_OP_SUBTRACTION) {
                value = max(a, -b);
            } else if (op == LOAM_OP_SMOOTH_UNION) {
                value = loam_field_smooth_min(a, b, bitcast<f32>(arg));
            }
            stack[sp - 1u] = value;
        }
    }
    if (sp == 0u) {
        return LOAM_FIELD_FAR;
    }
    return stack[0];
}

fn loam_field_step(distance: f32) -> f32 {
    if (u.kind == LOAM_FIELD_IMPLICIT) {
        return u.implicit_step;
    }
    return max(distance, 0.0001);
}

fn loam_field_normal(p: vec3<f32>) -> vec3<f32> {
    let h = 0.001;
    let g = vec3<f32>(
        loam_field_sdf(p + vec3<f32>(h, 0.0, 0.0)) - loam_field_sdf(p - vec3<f32>(h, 0.0, 0.0)),
        loam_field_sdf(p + vec3<f32>(0.0, h, 0.0)) - loam_field_sdf(p - vec3<f32>(0.0, h, 0.0)),
        loam_field_sdf(p + vec3<f32>(0.0, 0.0, h)) - loam_field_sdf(p - vec3<f32>(0.0, 0.0, h)),
    );
    if (dot(g, g) < 1.0e-24) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(g);
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

struct Shaded {
    color: vec4<f32>,
    depth: f32,
};

struct Fragment {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

fn shade(frag_pos: vec4<f32>) -> Shaded {
    let uv = ((frag_pos.xy - u.viewport_origin) / u.resolution) * 2.0 - vec2<f32>(1.0, 1.0);
    let aspect = u.resolution.x / u.resolution.y;
    let ndc = vec2<f32>(uv.x * aspect, -uv.y);
    let rd = normalize(
        u.camera_forward
        + u.camera_right * (ndc.x * u.fov_y_tan)
        + u.camera_up * (ndc.y * u.fov_y_tan)
    );
    let ro = u.camera_pos;

    var t: f32 = 0.0;
    var hit = false;
    for (var i: i32 = 0; i < 256; i = i + 1) {
        let d = loam_field_sdf(ro + rd * t);
        if (d < 0.001) {
            hit = true;
            break;
        }
        t = t + loam_field_step(d);
        if (t > u.max_t) {
            break;
        }
    }
    if (!hit) {
        discard;
        return Shaded(vec4<f32>(0.0), 0.0);
    }

    let n = loam_field_normal(ro + rd * t);
    let light_dir = normalize(vec3<f32>(0.5, 0.85, 0.3));
    let lit = u.params.xyz * (0.20 + max(dot(n, light_dir), 0.0) * 0.85);
    let image = vec3<f32>(0.0, 0.0, -t * dot(rd, u.camera_forward));
    return Shaded(vec4<f32>(lit, 1.0), loam_projective_depth(image, u.near));
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    return shade(frag_pos).color;
}

@fragment
fn fs_depth(@builtin(position) frag_pos: vec4<f32>) -> Fragment {
    let shaded = shade(frag_pos);
    return Fragment(shaded.color, shaded.depth);
}
