//! Root image space: the eye at the origin looking down -Z. Depth is
//! `near / -z`, reversed-Z in `Depth32Float`, cleared to zero and compared
//! `GreaterEqual`; every pass here writes that quantity.

use glam::{Mat4, Vec3};
use loam_runtime::Eye;
use wgpu::{
    CompareFunction, Device, Extent3d, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureView, TextureViewDescriptor,
};

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

pub struct DepthBuffer {
    pub view: TextureView,
    pub format: TextureFormat,
    size: (u32, u32),
}

impl DepthBuffer {
    pub fn new(device: &Device, format: TextureFormat, size: (u32, u32)) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("loam-render DepthBuffer"),
            size: Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self { view, format, size }
    }

    pub fn ensure(
        slot: &mut Option<DepthBuffer>,
        device: &Device,
        format: TextureFormat,
        size: (u32, u32),
    ) {
        let needs_recreate = match slot {
            Some(b) => b.format != format || b.size != size,
            None => true,
        };
        if needs_recreate {
            *slot = Some(DepthBuffer::new(device, format, size));
        }
    }
}
