use glam::Vec4;
use loam::render::raymarch::scene::{Scene4, SceneNode4};
use loam::render::raymarch::{polytope_extended_sdfs_wgsl, HYPERSLICE_KERNEL_WGSL};
use loam::render::sky_ground::{Ground, DEFAULT_FOG_PER_UNIT, GROUND_DARK_GREY, GROUND_LIGHT_GREY};

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
        fog_start: 0.0,
        visible,
    }
}

// `shader_source` gates its floor on `u.params.x`.
pub(crate) fn params(floor_visible: bool) -> [f32; 4] {
    [if floor_visible { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0]
}
