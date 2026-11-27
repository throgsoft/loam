use loam::render::shader::validate_wgsl;

#[test]
fn mandelbulb() {
    validate_wgsl(include_str!("../shaders/mandelbulb.wgsl"))
        .unwrap_or_else(|error| panic!("mandelbulb: {error}"));
}
