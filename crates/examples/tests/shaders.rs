use loam::render::shader::validate_wgsl;

#[test]
fn the_mandelbulb_shader_validates() {
    validate_wgsl(include_str!("../shaders/mandelbulb.wgsl"))
        .unwrap_or_else(|error| panic!("mandelbulb: {error}"));
}
