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
