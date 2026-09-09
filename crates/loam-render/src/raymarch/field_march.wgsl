fn loam_field_range(start: u32, end: u32, point: vec4<f32>) -> f32 {
    var stack: array<f32, LOAM_MAX_STACK>;
    var points: array<vec4<f32>, LOAM_MAX_POSE_DEPTH>;
    var sp: u32 = 0u;
    var pp: u32 = 0u;
    var p = point;
    for (var i: u32 = start; i + 1u < end; i = i + 2u) {
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

fn loam_field_sdf(p3: vec3<f32>) -> f32 {
    let p = vec4<f32>(p3, u.w_slice);
    if (u.node_len == 0u) {
        return loam_field_range(0u, u.program_len, p);
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
            best = min(best, loam_field_range(node.start, node.end, p));
        }
        i = i + 1u;
    }
    return best;
}
