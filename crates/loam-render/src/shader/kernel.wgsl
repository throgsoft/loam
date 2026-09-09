
fn loam_safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if l2 < 1e-12 { return fallback; }
    return v * inverseSqrt(l2);
}

fn loam_march_geodesic(ro: vec3<f32>, rd: vec3<f32>, ball_scale: f32) -> vec4<f32> {
    let scale = max(ball_scale, 1e-5);
    var p = ro * scale;

    // Probe the local Riemannian metric scaling to build a Riemannian-unit tangent. Space-agnostic via the ABI.
    let rd_unit = loam_safe_normalize(rd, vec3<f32>(0.0, 0.0, -1.0));
    let probe_eps = 1e-4;
    let probed     = loam_exp(p, rd_unit * probe_eps);
    let riem_norm  = loam_distance(p, probed) / probe_eps;
    var v = rd_unit / max(riem_norm, 1e-7);

    var t_scene = 0.0;
    var t_arc   = 0.0;
    let hit_eps  = 0.001  * scale;
    let min_step = 0.0001 * scale;

    for (var i = 0; i < 256; i = i + 1) {
        // Escape near the Space boundary. 0.92 buffers the ABI's saturating distance so it doesn't asymptote into a stall before escape fires.
        if loam_origin_distance(p) > LOAM_MAX_ARC * 0.92 {
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }
        let d = loam_scene_sdf(p);
        if d < hit_eps {
            return vec4<f32>(p, t_scene);
        }
        if t_scene > 40.0 || t_arc > LOAM_MAX_ARC {
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }
        let step   = max(d * 0.85, min_step);
        let next_p = loam_exp(p, v * step);
        let next_v = loam_parallel_transport(p, next_p, v);
        p = next_p;
        v       = select(v, next_v, dot(next_v, next_v) > 1e-12);
        t_scene = t_scene + step / scale;
        t_arc   = t_arc   + step;
    }
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
}

fn loam_estimate_normal(p: vec3<f32>, ball_scale: f32) -> vec3<f32> {
    let eps = 0.0012 * max(ball_scale, 1e-5);
    let ex = vec3<f32>(eps, 0.0, 0.0);
    let ey = vec3<f32>(0.0, eps, 0.0);
    let ez = vec3<f32>(0.0, 0.0, eps);
    let g = vec3<f32>(
        loam_scene_sdf(p + ex) - loam_scene_sdf(p - ex),
        loam_scene_sdf(p + ey) - loam_scene_sdf(p - ey),
        loam_scene_sdf(p + ez) - loam_scene_sdf(p - ez),
    );
    return loam_safe_normalize(g, vec3<f32>(0.0, 1.0, 0.0));
}
