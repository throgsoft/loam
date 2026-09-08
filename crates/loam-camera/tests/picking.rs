use glam::{Vec2, Vec3};
use loam_camera::{Camera, Ray};
use loam_math::EuclideanR3;

#[test]
fn off_axis_pick_uses_fov_and_aspect() {
    let mut camera = Camera::<EuclideanR3>::at_origin();
    camera.fov_y = std::f32::consts::FRAC_PI_2;
    camera.aspect = 2.0;
    let ray = camera.ray_from_ndc(Vec2::splat(0.5));
    let distance = ray
        .intersect_triangle(
            Vec3::new(1.8, 0.8, -2.0),
            Vec3::new(2.2, 0.8, -2.0),
            Vec3::new(2.0, 1.2, -2.0),
        )
        .unwrap();
    assert!((distance - 3.0).abs() < 1e-5);
}

#[test]
fn camera_ray_hits_both_triangle_windings() {
    let camera = Camera::<EuclideanR3>::at_origin();
    let ray = camera.ray_from_ndc(Vec2::ZERO);
    let a = Vec3::new(-1.0, -1.0, -3.0);
    let b = Vec3::new(1.0, -1.0, -3.0);
    let c = Vec3::new(0.0, 1.0, -3.0);
    assert_eq!(ray.intersect_triangle(a, b, c), Some(3.0));
    assert_eq!(ray.intersect_triangle(c, b, a), Some(3.0));
    assert_eq!(ray.intersect_triangle(-a, -b, -c), None);
    assert_eq!(ray.intersect_triangle(a, a, c), None);
    assert_eq!(
        ray.intersect_triangle(a + Vec3::X * 4.0, b + Vec3::X * 4.0, c + Vec3::X * 4.0),
        None
    );
}

#[test]
fn sphere_pick_selects_positive_root() {
    let ray = Ray {
        origin: Vec3::ZERO,
        direction: -Vec3::Z,
    };
    assert_eq!(ray.intersect_sphere(-Vec3::Z * 4.0, 1.0), Some(3.0));
    assert_eq!(ray.intersect_sphere(-Vec3::Z * 0.5, 1.0), Some(1.5));
    assert_eq!(ray.intersect_sphere(Vec3::Z * 4.0, 1.0), None);
    assert_eq!(ray.intersect_sphere(Vec3::new(3.0, 0.0, -4.0), 1.0), None);
    assert_eq!(
        ray.intersect_sphere(Vec3::new(1.0, 0.0, -4.0), 1.0),
        Some(4.0)
    );
}
