//! Root image space: the eye at the origin looking down -Z. Depth is
//! `near / -z`, reversed-Z in `Depth32Float`, cleared to zero and compared
//! `GreaterEqual`; every pass here writes that quantity.

use glam::{Mat4, Vec3};
use loam_runtime::Eye;
use wgpu::{CompareFunction, TextureFormat};

pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;
pub const DEPTH_COMPARE: CompareFunction = CompareFunction::GreaterEqual;
pub const DEPTH_CLEAR: f32 = 0.0;

pub fn root_projection(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    Mat4::perspective_infinite_reverse_rh(fov_y, aspect, near)
}

pub fn projective_depth(image: Vec3, near: f32) -> f32 {
    near / -image.z
}

pub fn root_view_projection(eye: &Eye) -> Mat4 {
    let basis = Mat4::from_cols(
        Vec3::from(eye.right).extend(0.0),
        Vec3::from(eye.up).extend(0.0),
        (-Vec3::from(eye.forward)).extend(0.0),
        Vec3::from(eye.position).extend(1.0),
    );
    root_projection(eye.fov_y, eye.aspect, eye.near) * basis.inverse()
}
