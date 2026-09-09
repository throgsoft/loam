fn loam_field_sdf(p3: vec3<f32>) -> f32 {
    let p = vec4<f32>(p3, u.w_slice);
    if (u.node_len == 0u) {
        return loam_field_all(p);
    }
    var best = LOAM_FIELD_FAR;
    var i: u32 = 0u;
    loop {
        if (i >= u.node_len) {
            break;
        }
        let node = loam_field_nodes[i];
        if (LOAM_FIELD_COUNTING) {
            loam_field_visits = loam_field_visits + 1u;
        }
        if (node.radius >= 0.0 && length(p - node.center) - node.radius > best + u.cull_tolerance) {
            if (LOAM_FIELD_COUNTING) {
                loam_field_skips = loam_field_skips + 1u;
            }
            i = node.escape;
            continue;
        }
        if (node.end != 0u) {
            best = min(best, loam_field_leaf(node, p));
        }
        i = i + 1u;
    }
    return best;
}
