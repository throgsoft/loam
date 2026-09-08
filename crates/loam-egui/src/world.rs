use glam::Vec3;
use loam_camera::Camera;
use loam_math::Space;

/// `viewport` in egui points; `None` outside the frustum.
pub fn world_to_screen<S: Space<Point = Vec3, Vector = Vec3>>(
    camera: &Camera<S>,
    world: Vec3,
    viewport: (u32, u32),
    space: &S,
) -> Option<egui::Pos2> {
    let pixels = camera.pixels_from_world(world, viewport, space)?;
    Some(egui::Pos2::new(pixels.x, pixels.y))
}
