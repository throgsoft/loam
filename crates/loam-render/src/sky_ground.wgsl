
fn sky(rd: vec3<f32>) -> vec3<f32> {
    let t = (rd.y + 1.0) * 0.5;
    return mix(vec3<f32>(0.04, 0.05, 0.10), vec3<f32>(0.10, 0.13, 0.22), t);
}

fn ground_color(
    p: vec3<f32>,
    dark: vec3<f32>,
    light: vec3<f32>,
    checker_fade: f32,
) -> vec3<f32> {
    let g = floor(p.x) + floor(p.z);
    let alt = abs(g - 2.0 * floor(g * 0.5));
    return mix(mix(dark, light, alt), 0.5 * (dark + light), checker_fade);
}
