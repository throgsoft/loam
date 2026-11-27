const NO_BODY: u32 = 0xffffffffu;

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
    near: f32,
    body_offset: f32,
    aspect: f32,
    sky_below: vec3<f32>,
    sky_above: vec3<f32>,
    ground_plane: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> bodies: array<BodyUniform>;

fn body_at(index: u32) -> BodyUniform {
    return bodies[u32(u.body_offset + 0.5) + index];
}

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

// Outside 1.5x the circumradius the ball SDF is a Lipschitz-1 lower bound; skips the rotor inverse.
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

fn loam_body_sdf_at(p3: vec3<f32>, body_idx: u32) -> f32 {
    if (body_idx >= u32(u.body_count + 0.5)) { return 1.0e9; }
    let p4 = vec4<f32>(p3, u.w_slice);
    let b = body_at(body_idx);
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
    var dyn_idx: u32 = NO_BODY;
    for (var i: u32 = 0u; i < body_count; i = i + 1u) {
        let b = body_at(i);
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
        return HitInfo(scene_d, NO_BODY);
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
    if (body_idx == NO_BODY) {
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
    let ndc = vec2<f32>(uv.x * u.aspect, -uv.y);
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
    var hit_idx: u32 = NO_BODY;
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
        return Shaded(vec4<f32>(0.0, 0.0, 0.0, 0.0), 0.0);
    }

    let p_hit = ro + rd * t;
    if (hit_idx == NO_BODY) {
        if (loam_scene_at(p_hit).kind == LOAM_PRIM_HALFSPACE4D) {
            discard;
            return Shaded(vec4<f32>(0.0, 0.0, 0.0, 0.0), 0.0);
        }
    }
    let n = estimate_normal(p_hit, hit_idx);
    let light_dir = normalize(vec3<f32>(0.5, 0.85, 0.3));
    let lambert = max(dot(n, light_dir), 0.0);
    let ambient = 0.20;
    var base = vec3<f32>(0.65, 0.65, 0.72);
    if (hit_idx != NO_BODY) {
        base = body_at(hit_idx).color;
    }
    let lit = base * (ambient + lambert * 0.85);
    let fog = 1.0 - exp(-t * 0.05);
    let final_color = mix(lit, sky(rd, u.sky_below, u.sky_above), fog * 0.5);
    let image = vec3<f32>(0.0, 0.0, -t * dot(rd, u.camera_forward));
    return Shaded(vec4<f32>(final_color, 1.0), loam_projective_depth(image, u.near));
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
