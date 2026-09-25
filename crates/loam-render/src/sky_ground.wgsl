
fn sky(rd: vec3<f32>, below: vec3<f32>, above: vec3<f32>) -> vec3<f32> {
    return mix(below, above, (rd.y + 1.0) * 0.5);
}

// Box-filtered checker: Quilez, "Filtering the checkerboard pattern" (2013).
fn ground_color(
    p: vec3<f32>,
    footprint: vec2<f32>,
    dark: vec3<f32>,
    light: vec3<f32>,
    checker_fade: f32,
) -> vec3<f32> {
    let w = footprint + vec2<f32>(1.0e-3);
    let i = 2.0 * (abs(fract((p.xz - 0.5 * w) * 0.5) - 0.5) - abs(fract((p.xz + 0.5 * w) * 0.5) - 0.5)) / w;
    let alt = 0.5 - 0.5 * i.x * i.y;
    return mix(mix(dark, light, alt), 0.5 * (dark + light), checker_fade);
}
