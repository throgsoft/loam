use loam_math::{
    BlendedSpace, EuclideanR3, EuclideanR4, FlatTorus3, HyperbolicH3, LensSpace, LinearBlendX,
    SphericalS3, WgslSpace,
};
use loam_render::shader::{assemble_wgsl, validate_wgsl, GEODESIC_MARCH_KERNEL};

#[test]
fn standalone_shader_files_validate() {
    for (name, source) in [
        ("line", include_str!("../src/line_raster.wgsl")),
        ("line_r4", include_str!("../src/line_raster_static_r4.wgsl")),
        ("point", include_str!("../src/point_raster.wgsl")),
        ("triangle", include_str!("../src/triangle_raster.wgsl")),
        ("composite", include_str!("../src/composite.wgsl")),
    ] {
        validate_wgsl(source).unwrap_or_else(|error| panic!("{name}: {error}"));
    }
}

const MARCH_SCENE: &str = r#"
fn loam_scene_sdf(p: vec3<f32>) -> f32 {
    return loam_distance(p, vec3<f32>(0.0, 0.0, 0.0)) - 0.25;
}
"#;

const MARCH_CALLER: &str = r#"
@compute @workgroup_size(1)
fn main() {
    let a = vec3<f32>(0.1, 0.2, 0.3);
    let b = vec3<f32>(0.2, -0.1, 0.05);
    let v = vec3<f32>(0.01, 0.02, -0.03);
    _ = loam_distance(a, b);
    _ = loam_origin_distance(a);
    _ = loam_exp(a, v);
    _ = loam_log(a, b);
    _ = loam_geodesic_step(a, v, 1.0);
    _ = LOAM_MAX_ARC;
    _ = loam_march_geodesic(a, v, 0.2);
    _ = loam_estimate_normal(a, 0.2);
    _ = loam_safe_normalize(v, b);
}
"#;

const R4_CALLER: &str = r#"
@compute @workgroup_size(1)
fn main() {
    let a = vec4<f32>(0.1, 0.2, 0.3, 0.0);
    let b = vec4<f32>(0.2, -0.1, 0.05, 0.4);
    let v = vec4<f32>(0.01, 0.02, -0.03, 0.05);
    _ = loam_distance(a, b);
    _ = loam_origin_distance(a);
    _ = loam_exp(a, v);
    _ = loam_log(a, b);
    _ = loam_geodesic_step(a, v, 1.0);
    _ = LOAM_MAX_ARC;
}
"#;

const LENS_CALLER: &str = r#"
@compute @workgroup_size(1)
fn main() {
    let x = vec4<f32>(0.5, 0.5, 0.5, 0.5);
    let centre = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    _ = loam_lens_wrap(x);
    _ = loam_lens_distance(x, centre);
    _ = loam_lens_nearest_power(x, centre);
}
"#;

fn march_chain(space: &impl WgslSpace) -> String {
    assemble_wgsl(&[
        space.wgsl_impl().as_ref(),
        MARCH_SCENE,
        GEODESIC_MARCH_KERNEL,
        MARCH_CALLER,
    ])
}

#[test]
fn every_space_prelude_validates_under_the_shader_code_that_calls_it() {
    let blend = LinearBlendX::new(-2.0, 2.0).expect("an increasing blend interval");
    for (name, source) in [
        ("EuclideanR3", march_chain(&EuclideanR3)),
        ("HyperbolicH3", march_chain(&HyperbolicH3)),
        ("SphericalS3", march_chain(&SphericalS3)),
        (
            "BlendedSpace<E3, H3>",
            march_chain(&BlendedSpace::new(EuclideanR3, HyperbolicH3, blend)),
        ),
        (
            "FlatTorus3",
            march_chain(&FlatTorus3::new(glam::Vec3::new(2.0, 2.6, 1.7))),
        ),
        (
            "EuclideanR4",
            assemble_wgsl(&[EuclideanR4.wgsl_impl().as_ref(), R4_CALLER]),
        ),
        (
            "LensSpace",
            assemble_wgsl(&[LensSpace::new(5, 2).wgsl_prelude().as_str(), LENS_CALLER]),
        ),
    ] {
        validate_wgsl(&source).unwrap_or_else(|error| panic!("{name}: {error}"));
    }
}
