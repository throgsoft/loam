use glam::Vec4;
use loam::math::EuclideanR4;
use loam::render::raymarch::{
    polytope_extended_sdfs_wgsl, BodyUniform, Hyperslice4DUniforms, HYPERSLICE_KERNEL_WGSL,
};
use loam::render::sky_ground::{Ground, DEFAULT_FOG_PER_UNIT, GROUND_DARK_GREY, GROUND_LIGHT_GREY};
use loam::runtime::Eye;
use loam::runtime::Pose;
use loam::scene::{Scene4, SceneNode4};

use crate::catalog::ShapeEntry;
use crate::consts::FLOOR_Y;

pub(crate) fn shader_source() -> String {
    let scene = Scene4::new(SceneNode4::halfspace(Vec4::Y, FLOOR_Y));
    format!(
        "{kernel}\n{polytope}\n{scene}\n",
        kernel = HYPERSLICE_KERNEL_WGSL,
        polytope = polytope_extended_sdfs_wgsl(),
        scene = scene.to_hyperslice_wgsl_gated("u.w_slice", "u.params.x"),
    )
}

pub(crate) fn ground(visible: bool) -> Ground {
    Ground {
        y: FLOOR_Y,
        dark: GROUND_DARK_GREY,
        light: GROUND_LIGHT_GREY,
        fog_per_unit: DEFAULT_FOG_PER_UNIT,
        visible,
    }
}

pub(crate) fn body_of(entry: &ShapeEntry, pose: &Pose<EuclideanR4>, size: f32) -> BodyUniform {
    BodyUniform::polytope_with_rotor(
        pose.point.to_array(),
        entry.shape.shape_id(),
        size,
        pose.frame,
        entry.body_color,
    )
}

pub(crate) fn uniforms(eye: &Eye, w_slice: f32, floor_visible: bool) -> Hyperslice4DUniforms {
    Hyperslice4DUniforms {
        camera_pos: eye.position,
        camera_forward: eye.forward,
        camera_right: eye.right,
        camera_up: eye.up,
        fov_y_tan: (eye.fov_y * 0.5).tan(),
        w_slice,
        params: [if floor_visible { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        near: eye.near,
        ..Hyperslice4DUniforms::default()
    }
}
