use glam::{Vec3, Vec4};
use loam_math::{EuclideanR3, EuclideanR4};
use loam_physics::{euclidean_r3 as r3, euclidean_r4 as r4, Narrowphase};

#[test]
fn sphere_hull_contacts_have_no_fixed_world_size_limit() {
    for scale in [0.1, 1.0, 100.0] {
        let mut np3 = Narrowphase::new();
        r3::register_default_narrowphase(&mut np3);
        let sphere = r3::sphere_body_r3(Vec3::X * (1.5 * scale), Vec3::ZERO, scale, 1.0).unwrap();
        let hull = r3::box_body(Vec3::ZERO, Vec3::ZERO, Vec3::splat(scale), 0.0).unwrap();
        let contact = np3
            .test(&sphere, &hull, &EuclideanR3)
            .expect("overlapping sphere and box");
        assert!((contact.penetration / scale - 0.5).abs() < 0.01);
        assert!(contact.normal.dot(-Vec3::X) > 0.99);

        let mut np4 = Narrowphase::new();
        r4::register_default_narrowphase(&mut np4);
        let sphere = r4::sphere_body_r4(Vec4::X * (1.5 * scale), Vec4::ZERO, scale, 1.0).unwrap();
        let hull = r4::polytope_body_r4(
            Vec4::ZERO,
            Vec4::ZERO,
            r4::tesseract_vertices(2.0 * scale),
            0.0,
        )
        .unwrap();
        let contact = np4
            .test(&sphere, &hull, &EuclideanR4)
            .expect("overlapping ball and tesseract");
        assert!((contact.penetration / scale - 0.5).abs() < 0.02);
        assert!(contact.normal.dot(-Vec4::X) > 0.99);
    }
}
