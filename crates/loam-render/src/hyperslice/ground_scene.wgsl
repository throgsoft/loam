
const LOAM_PRIM_HALFSPACE4D: u32 = 1u;
struct LoamSceneHit { dist: f32, kind: u32 }
fn loam_scene_at(p: vec3<f32>) -> LoamSceneHit {
    let d = select(1.0e9, dot(p, u.ground_plane.xyz) - u.ground_plane.w, u.params.x >= 0.5);
    return LoamSceneHit(d, LOAM_PRIM_HALFSPACE4D);
}
fn loam_scene_sdf(p: vec3<f32>) -> f32 {
    return loam_scene_at(p).dist;
}
fn loam_scene_max_t(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    var t_max: f32 = 1.0e9;
    if (u.params.x >= 0.5) {
        let dr = dot(rd, u.ground_plane.xyz);
        if (dr < -1.0e-4) {
            let t = (u.ground_plane.w - dot(ro, u.ground_plane.xyz)) / dr;
            if (t > 0.0 && t < t_max) { t_max = t; }
        }
    }
    return t_max;
}
