use glam::Vec4;
use loam::render::raymarch::scene::{Scene4, SceneNode4};
use loam::render::raymarch::{polytope_extended_sdfs_wgsl, HYPERSLICE_KERNEL_WGSL};

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
