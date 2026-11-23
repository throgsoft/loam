// The leading fields of loam-render's RayMarchUniforms; params.x is the turn angle about +Y in radians.
struct Uniforms {
    camera_pos: vec3<f32>,
    camera_forward: vec3<f32>,
    camera_right: vec3<f32>,
    camera_up: vec3<f32>,
    fov_y_tan: f32,
    resolution: vec2<f32>,
    time: f32,
    tick: f32,
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// The classic Mandelbulb exponent (White and Nylander, 2009).
const POWER: f32 = 8.0;
const ITERATIONS: i32 = 12;
const MARCH_STEPS: i32 = 128;
const HIT_EPSILON: f32 = 0.001;
const FAR: f32 = 20.0;
// Distance at which the fog hides the surface completely.
const FOG_DISTANCE: f32 = 12.0;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    // One triangle twice the viewport's size covers it without a diagonal seam.
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn sky_color(rd: vec3<f32>) -> vec3<f32> {
    let horizon = (rd.y + 1.0) * 0.5;
    return mix(vec3<f32>(0.03, 0.04, 0.10), vec3<f32>(0.08, 0.10, 0.18), horizon);
}

fn turned(p: vec3<f32>) -> vec3<f32> {
    let c = cos(u.params.x);
    let s = sin(u.params.x);
    return vec3<f32>(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}

fn mandelbulb_de(world: vec3<f32>) -> f32 {
    let p = turned(world);
    var z = p;
    var dr = 1.0;
    var r = 0.0;
    for (var i = 0; i < ITERATIONS; i = i + 1) {
        r = length(z);
        // Past radius 2 the orbit escapes to infinity.
        if (r > 2.0) { break; }
        let theta = acos(clamp(z.z / r, -1.0, 1.0));
        let phi = atan2(z.y, z.x);
        dr = pow(r, POWER - 1.0) * POWER * dr + 1.0;
        let zr = pow(r, POWER);
        let nt = theta * POWER;
        let np = phi * POWER;
        z = zr * vec3<f32>(sin(nt) * cos(np), sin(nt) * sin(np), cos(nt)) + p;
    }
    // Escape-time distance estimate 0.5 r ln(r) / |dz| (Hart, Sandin and Kauffman, 1989).
    return 0.5 * log(max(r, 1e-4)) * r / dr;
}

// Distance along the ray to the surface, or -1 on a miss.
fn march(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    var t = 0.0;
    for (var i = 0; i < MARCH_STEPS; i = i + 1) {
        let d = mandelbulb_de(ro + rd * t);
        if (d < HIT_EPSILON) {
            return t;
        }
        if (t > FAR) {
            return -1.0;
        }
        // The estimate can overshoot near the surface, so each step takes nine tenths of it.
        t = t + max(d * 0.9, HIT_EPSILON);
    }
    return -1.0;
}

fn estimate_normal(p: vec3<f32>) -> vec3<f32> {
    let eps = 0.0015;
    let ex = vec3<f32>(eps, 0.0, 0.0);
    let ey = vec3<f32>(0.0, eps, 0.0);
    let ez = vec3<f32>(0.0, 0.0, eps);
    return normalize(vec3<f32>(
        mandelbulb_de(p + ex) - mandelbulb_de(p - ex),
        mandelbulb_de(p + ey) - mandelbulb_de(p - ey),
        mandelbulb_de(p + ez) - mandelbulb_de(p - ez),
    ));
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (pos.xy / u.resolution) * 2.0 - 1.0;
    let aspect = u.resolution.x / u.resolution.y;
    // Pixel y grows downward and view y grows upward.
    let ndc = vec2<f32>(uv.x * aspect, -uv.y);

    let rd = normalize(
        u.camera_forward
        + u.camera_right * ndc.x * u.fov_y_tan
        + u.camera_up * ndc.y * u.fov_y_tan
    );

    let sky = sky_color(rd);
    let t = march(u.camera_pos, rd);
    if (t < 0.0) {
        return vec4<f32>(sky, 1.0);
    }

    let hit = u.camera_pos + rd * t;
    let n = estimate_normal(hit);
    let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
    let diffuse = max(dot(n, sun_dir), 0.0);
    let ambient = 0.15;
    let fog = clamp(t / FOG_DISTANCE, 0.0, 1.0);

    let sky_linear = pow(max(sky, vec3<f32>(0.0)), vec3<f32>(2.2));
    let base = vec3<f32>(0.35 + 0.3 * n.x, 0.55 + 0.2 * n.y, 0.9);
    let lit = base * (diffuse + ambient);
    let shaded = mix(lit, sky_linear, fog);
    return vec4<f32>(pow(max(shaded, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)), 1.0);
}
