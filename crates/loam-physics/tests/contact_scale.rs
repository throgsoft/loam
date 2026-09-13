#![cfg(all(feature = "r3", feature = "r4"))]

use glam::{Vec3, Vec4};
use loam_math::{EuclideanR3, EuclideanR4};
use loam_physics::{euclidean_r3 as r3, euclidean_r4 as r4, World};

#[test]
fn sphere_hull_contacts_have_no_fixed_world_size_limit() {
    for scale in [0.1, 1.0, 100.0] {
        let mut w3 = World::new(EuclideanR3);
        r3::register_default_narrowphase(&mut w3.narrowphase);
        let sphere = w3.push_body(
            r3::sphere_body_r3(Vec3::X * (1.5 * scale), Vec3::ZERO, scale, 1.0).unwrap(),
        );
        let hull =
            w3.push_body(r3::box_body(Vec3::ZERO, Vec3::ZERO, Vec3::splat(scale), 0.0).unwrap());
        let contact = w3
            .narrowphase
            .test(
                &w3.bodies()[sphere],
                &w3.bodies()[hull],
                w3.geometry(),
                w3.space(),
            )
            .expect("overlapping sphere and box");
        assert!((contact.penetration / scale - 0.5).abs() < 0.01);
        assert!(contact.normal.dot(-Vec3::X) > 0.99);

        let mut w4 = World::new(EuclideanR4);
        r4::register_default_narrowphase(&mut w4.narrowphase);
        let sphere = w4.push_body(
            r4::sphere_body_r4(Vec4::X * (1.5 * scale), Vec4::ZERO, scale, 1.0).unwrap(),
        );
        let hull = w4.push_body(
            r4::polytope_body_r4(
                Vec4::ZERO,
                Vec4::ZERO,
                r4::tesseract_vertices(2.0 * scale),
                0.0,
            )
            .unwrap(),
        );
        let contact = w4
            .narrowphase
            .test(
                &w4.bodies()[sphere],
                &w4.bodies()[hull],
                w4.geometry(),
                w4.space(),
            )
            .expect("overlapping ball and tesseract");
        assert!((contact.penetration / scale - 0.5).abs() < 0.02);
        assert!(contact.normal.dot(-Vec4::X) > 0.99);
    }
}
