use std::hash::Hasher;

use glam::{Vec2, Vec3, Vec4};
use loam_shape::Shape;

#[derive(Default)]
struct Written(Vec<u8>);

impl Hasher for Written {
    fn write(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    fn finish(&self) -> u64 {
        0
    }
}

fn stream(tag: u8, fields: &[f32]) -> Vec<u8> {
    let mut bytes = vec![tag];
    for field in fields {
        bytes.extend_from_slice(&field.to_bits().to_ne_bytes());
    }
    bytes
}

// A stored geometry table keeps each shape's hash and is refused on load when a shape hashes differently.
#[test]
fn every_shape_feeds_the_hasher_the_stream_stored_geometry_was_hashed_with() {
    let cases = [
        (
            Shape::Sphere {
                center: Vec3::new(1.0, 2.0, 3.0),
                radius: 4.0,
            },
            stream(0, &[1.0, 2.0, 3.0, 4.0]),
        ),
        (
            Shape::HalfSpace {
                normal: Vec3::new(0.0, 1.0, 0.0),
                offset: -2.0,
            },
            stream(1, &[0.0, 1.0, 0.0, -2.0]),
        ),
        (
            Shape::HalfSpace4D {
                normal: Vec4::new(0.0, 0.0, 0.0, 1.0),
                offset: 0.5,
            },
            stream(2, &[0.0, 0.0, 0.0, 1.0, 0.5]),
        ),
        (
            Shape::Box3 {
                half_extents: Vec3::new(1.0, 2.0, 3.0),
            },
            stream(3, &[1.0, 2.0, 3.0]),
        ),
        (
            Shape::Polygon2D {
                vertices: vec![Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0)],
            },
            stream(4, &[1.0, 2.0, 3.0, 4.0]),
        ),
        (
            Shape::ConvexPolytope3D {
                vertices: vec![Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)],
            },
            stream(5, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ),
        (
            Shape::ConvexPolytope4D {
                vertices: vec![Vec4::new(1.0, 2.0, 3.0, 4.0)],
            },
            stream(6, &[1.0, 2.0, 3.0, 4.0]),
        ),
        (
            Shape::HyperSphere4D {
                center: Vec4::new(1.0, 2.0, 3.0, 4.0),
                radius: 5.0,
            },
            stream(7, &[1.0, 2.0, 3.0, 4.0, 5.0]),
        ),
    ];

    for (shape, expected) in cases {
        let mut written = Written::default();
        shape.hash_into(&mut written);
        assert_eq!(written.0, expected, "{shape:?}");
    }
}
