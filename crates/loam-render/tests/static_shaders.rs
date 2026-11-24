use loam_render::shader::validate_wgsl;

#[test]
fn standalone_shaders() {
    for (name, source) in [
        ("line", include_str!("../src/line_raster.wgsl")),
        ("point", include_str!("../src/point_raster.wgsl")),
        ("triangle", include_str!("../src/triangle_raster.wgsl")),
        ("composite", include_str!("../src/composite.wgsl")),
    ] {
        validate_wgsl(source).unwrap_or_else(|error| panic!("{name}: {error}"));
    }
}
