#[test]
fn raster_shader_modules_validate() {
    for (name, source) in [
        ("line", include_str!("../src/line_raster.wgsl")),
        ("line_r4", include_str!("../src/line_raster_static_r4.wgsl")),
        ("point", include_str!("../src/point_raster.wgsl")),
        ("triangle", include_str!("../src/triangle_raster.wgsl")),
    ] {
        let module =
            naga::front::wgsl::parse_str(source).unwrap_or_else(|error| panic!("{name}: {error}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    }
}
