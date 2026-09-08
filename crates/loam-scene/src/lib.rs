//! Emit contract, shared by the 3D and 4D paths: every baked constant goes
//! through `literal::wgsl_f32`, so parsing the emitted literal recovers the
//! exact input bits and the emitter contributes no floor to CPU/GPU parity.

pub mod combinator;
pub mod edit;
mod literal;
pub mod load;
pub mod primitive;
pub mod primitive4;
pub mod scene;
pub mod scene4;

pub use edit::{EditError, NodePath, SceneEdit};
pub use load::SceneLoadError;
pub use loam_shape::Shape;
pub use primitive::Primitive;
pub use primitive4::Primitive4;
pub use scene::{PrimitiveKind, Scene, SceneNode};
pub use scene4::{
    Scene4, SceneNode4, PRIM_KIND_HALFSPACE4D, PRIM_KIND_HYPERSPHERE4D, PRIM_KIND_OTHER,
};

/// Returned by shapes with no closed-form SDF in the emitted dimension.
pub const SENTINEL_DISTANCE: f32 = 1e9;

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn one_shape_per_kind() -> Vec<Shape> {
        use glam::{Vec2, Vec4};
        vec![
            Shape::sphere_at(Vec3::new(0.05, -0.02, 0.03), 0.25),
            Shape::HalfSpace {
                normal: Vec3::Y,
                offset: -0.5,
            },
            Shape::HalfSpace4D {
                normal: Vec4::Y,
                offset: -0.25,
            },
            Shape::Box3 {
                half_extents: Vec3::new(0.4, 0.3, 0.2),
            },
            Shape::Polygon2D {
                vertices: vec![Vec2::ZERO, Vec2::X, Vec2::Y],
            },
            Shape::ConvexPolytope3D {
                vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z],
            },
            Shape::ConvexPolytope4D {
                vertices: vec![Vec4::ZERO, Vec4::X, Vec4::Y, Vec4::Z, Vec4::W],
            },
            Shape::HyperSphere4D {
                center: Vec4::new(0.1, 0.0, -0.1, 0.2),
                radius: 0.3,
            },
        ]
    }

    #[test]
    fn unsupported_primitives_keep_the_distance_sentinel() {
        use loam_math::{
            BlendedSpace, EuclideanR3, HyperbolicH3, LinearBlendX, Space, SphericalS3,
        };
        fn check<S: Space<Point = Vec3, Vector = Vec3>>(space: &S, halfspace_supported: bool) {
            let unsupported = [
                false,
                !halfspace_supported,
                true,
                false,
                true,
                true,
                true,
                true,
            ];
            for (shape, absent) in one_shape_per_kind().into_iter().zip(unsupported) {
                assert_eq!(
                    shape.eval(space, Vec3::new(0.11, -0.07, 0.13)) == SENTINEL_DISTANCE,
                    absent,
                    "{:?}",
                    shape.kind()
                );
            }
        }
        check(&EuclideanR3, true);
        check(&HyperbolicH3, false);
        check(&SphericalS3, false);
        check(
            &BlendedSpace::new(
                EuclideanR3,
                HyperbolicH3,
                LinearBlendX::new(-0.5, 0.5).unwrap(),
            ),
            false,
        );
        for (shape, absent) in one_shape_per_kind()
            .into_iter()
            .zip([true, true, false, true, true, true, true, false])
        {
            assert_eq!(
                shape.eval_4d(glam::Vec4::new(0.11, -0.07, 0.13, 0.05)) == SENTINEL_DISTANCE,
                absent,
                "{:?}",
                shape.kind()
            );
        }
    }

    #[test]
    fn halfspace_eval_follows_the_chart_flatness_gate() {
        use loam_math::{EuclideanR3, HyperbolicH3};
        let plane = Shape::HalfSpace {
            normal: Vec3::Y,
            offset: -0.5,
        };
        let p = Vec3::new(0.0, 0.25, 0.0);
        assert!((plane.eval(&EuclideanR3, p) - 0.75).abs() < 1e-6);
        assert_eq!(plane.eval(&HyperbolicH3, p), SENTINEL_DISTANCE);
        let torus = loam_math::FlatTorus3::cube(2.0);
        assert_eq!(plane.eval(&torus, p), SENTINEL_DISTANCE);
        assert_eq!(plane.eval(&torus, p + Vec3::Y * 2.0), SENTINEL_DISTANCE);
    }

    fn deterministic_pair_samples(seed: u32, count: usize, extent: f32) -> Vec<(Vec3, Vec3)> {
        let mut state = seed;
        let mut next_f32 = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        (0..count)
            .map(|_| {
                let a = Vec3::new(
                    next_f32() * extent,
                    next_f32() * extent,
                    next_f32() * extent,
                );
                let b = Vec3::new(
                    next_f32() * extent,
                    next_f32() * extent,
                    next_f32() * extent,
                );
                (a, b)
            })
            .collect()
    }

    fn deterministic_samples(seed: u32, count: usize, extent: f32) -> Vec<Vec3> {
        let mut state = seed;
        let mut next_f32 = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        (0..count)
            .map(|_| {
                Vec3::new(
                    next_f32() * extent,
                    next_f32() * extent,
                    next_f32() * extent,
                )
            })
            .collect()
    }

    fn assert_lipschitz_1_under_space_metric<S, F>(label: &str, space: &S, sdf: F, extent: f32)
    where
        S: loam_math::Space<Point = Vec3, Vector = Vec3>,
        F: Fn(Vec3) -> f32,
    {
        for (a, b) in deterministic_pair_samples(0xABCD_1234, 256, extent) {
            let separation = space.distance(a, b);
            if separation < 1e-6 {
                continue;
            }
            let delta = (sdf(a) - sdf(b)).abs();
            assert!(
                delta <= separation * (1.0 + 1e-5),
                "{label}: |sdf({a:?}) - sdf({b:?})| = {delta} exceeds d(a, b) = {separation}",
            );
        }
    }

    #[test]
    fn scene_eval_is_lipschitz_1_under_the_space_metric() {
        use loam_math::{EuclideanR3, HyperbolicH3, SphericalS3};
        let scene = Scene::new(
            SceneNode::sphere(Vec3::new(0.1, 0.0, 0.0), 0.2)
                .smooth_union(SceneNode::cube(0.15), 0.06)
                .union(SceneNode::sphere(Vec3::new(-0.2, 0.1, 0.0), 0.12))
                .subtract(SceneNode::sphere(Vec3::new(0.0, 0.2, 0.0), 0.08))
                .intersect(SceneNode::plane(Vec3::Y, -0.6)),
        );

        assert_lipschitz_1_under_space_metric(
            "E3",
            &EuclideanR3,
            |p| scene.eval(&EuclideanR3, p),
            1.0,
        );
        let scene = Scene::new(
            SceneNode::sphere(Vec3::new(0.1, 0.0, 0.0), 0.2)
                .smooth_union(SceneNode::sphere(Vec3::new(-0.2, 0.1, 0.0), 0.12), 0.06)
                .subtract(SceneNode::sphere(Vec3::new(0.0, 0.2, 0.0), 0.08)),
        );
        assert_lipschitz_1_under_space_metric(
            "H3",
            &HyperbolicH3,
            |p| scene.eval(&HyperbolicH3, p),
            0.3,
        );
        assert_lipschitz_1_under_space_metric(
            "S3",
            &SphericalS3,
            |p| scene.eval(&SphericalS3, p),
            0.3,
        );
    }

    #[test]
    fn sphere_eval_is_zero_on_the_geodesic_surface_in_every_space() {
        use loam_math::{EuclideanR3, HyperbolicH3, Space, SphericalS3};

        fn check<S: Space<Point = Vec3, Vector = Vec3>>(space: &S, label: &str) {
            let center = Vec3::new(0.05, -0.03, 0.02);
            let radius = 0.2_f32;
            let shape = Shape::sphere_at(center, radius);
            assert!(
                (shape.eval(space, center) + radius).abs() < 1e-6,
                "{label}: centre must read -radius",
            );
            for direction in [Vec3::X, Vec3::Y, Vec3::Z, -Vec3::X, Vec3::ONE.normalize()] {
                let probe = direction * 0.1;
                let probe_arc = space.distance(center, space.exp(center, probe));
                let at_arc = |arc: f32| space.exp(center, probe * (arc / probe_arc));

                assert!(
                    shape.eval(space, at_arc(radius)).abs() < 1e-5,
                    "{label}: the geodesic sphere of radius r must be the zero set",
                );
                assert!(
                    shape.eval(space, at_arc(radius * 2.0)) > 0.0,
                    "{label}: sign outside",
                );
                assert!(
                    shape.eval(space, at_arc(radius * 0.5)) < 0.0,
                    "{label}: sign inside",
                );
            }
        }

        check(&EuclideanR3, "E3");
        check(&HyperbolicH3, "H3");
        check(&SphericalS3, "S3");
    }

    #[test]
    fn box_and_halfspace_eval_zero_sets_and_signs_in_e3() {
        use loam_math::EuclideanR3;
        let half_extents = Vec3::splat(0.4);
        let box3 = Shape::Box3 { half_extents };
        assert!((box3.eval(&EuclideanR3, Vec3::ZERO) + 0.4).abs() < 1e-6);
        assert!(box3.eval(&EuclideanR3, Vec3::new(0.4, 0.0, 0.0)).abs() < 1e-6);
        assert!((box3.eval(&EuclideanR3, Vec3::new(1.4, 0.0, 0.0)) - 1.0).abs() < 1e-5);
        let corner_offset = Vec3::splat(1.0);
        assert!(
            (box3.eval(&EuclideanR3, half_extents + corner_offset) - corner_offset.length()).abs()
                < 1e-5
        );

        let plane = Shape::HalfSpace {
            normal: Vec3::Y,
            offset: -0.5,
        };
        assert!(plane.eval(&EuclideanR3, Vec3::new(0.0, -0.5, 0.0)).abs() < 1e-6);
        assert!((plane.eval(&EuclideanR3, Vec3::Y) - 1.5).abs() < 1e-5);
        assert!(plane.eval(&EuclideanR3, Vec3::new(0.0, -1.0, 0.0)) < 0.0);
    }

    #[test]
    fn union_commutes_and_difference_does_not() {
        use loam_math::EuclideanR3;
        let left = SceneNode::sphere(Vec3::new(-0.1, 0.0, 0.0), 0.3);
        let right = SceneNode::sphere(Vec3::new(0.1, 0.0, 0.0), 0.3);
        let union_lr = Scene::new(left.clone().union(right.clone()));
        let union_rl = Scene::new(right.clone().union(left.clone()));
        let diff_lr = Scene::new(left.clone().subtract(right.clone()));
        let diff_rl = Scene::new(right.subtract(left));

        let mut asymmetric = 0usize;
        for p in deterministic_samples(0x0BAD_F00D, 256, 0.6) {
            assert_eq!(
                union_lr.eval(&EuclideanR3, p),
                union_rl.eval(&EuclideanR3, p)
            );
            if diff_lr.eval(&EuclideanR3, p) != diff_rl.eval(&EuclideanR3, p) {
                asymmetric += 1;
            }
        }
        assert!(
            asymmetric > 0,
            "A minus B and B minus A must differ somewhere in the sampled volume",
        );
    }

    #[test]
    fn smooth_union_underestimates_min_and_converges_to_it() {
        use loam_math::EuclideanR3;
        let left = SceneNode::sphere(Vec3::new(-0.2, 0.0, 0.0), 0.25);
        let right = SceneNode::sphere(Vec3::new(0.2, 0.0, 0.0), 0.25);
        let hard = Scene::new(left.clone().union(right.clone()));
        let samples = deterministic_samples(0x51DF_00D5, 512, 0.7);

        for k in [0.25_f32, 0.05, 1e-3] {
            let soft = Scene::new(left.clone().smooth_union(right.clone(), k));
            for &p in &samples {
                let smooth = soft.eval(&EuclideanR3, p);
                let sharp = hard.eval(&EuclideanR3, p);
                assert!(
                    smooth <= sharp + 1e-6,
                    "k={k}: smooth_union {smooth} must not exceed min {sharp} at {p:?}",
                );
                assert!(
                    sharp - smooth <= k * 0.25 + 1e-6,
                    "k={k}: gap {} exceeds the k/4 worst case at {p:?}",
                    sharp - smooth,
                );
            }
        }
    }

    #[test]
    fn smooth_union_matches_min_outside_the_blend_band() {
        use loam_math::EuclideanR3;
        let left = SceneNode::sphere(Vec3::new(-0.5, 0.0, 0.0), 0.2);
        let right = SceneNode::sphere(Vec3::new(0.5, 0.0, 0.0), 0.2);
        let soft = Scene::new(left.clone().smooth_union(right.clone(), 0.02));
        let hard = Scene::new(left.union(right));

        let p = Vec3::new(-0.5, 0.0, 0.0);
        assert_eq!(soft.eval(&EuclideanR3, p), hard.eval(&EuclideanR3, p));
    }

    #[test]
    fn hyperslice_eval_tracks_distance_and_kind_through_combinators() {
        use glam::Vec4;
        use scene4::{PRIM_KIND_HALFSPACE4D, PRIM_KIND_HYPERSPHERE4D, PRIM_KIND_OTHER};

        let ball = SceneNode4::hypersphere(Vec4::ZERO, 0.5);
        let floor = SceneNode4::halfspace(Vec4::Y, -0.4);

        let union = Scene4::new(ball.clone().union(floor.clone()));
        let (dist, kind) = union.eval_at(Vec3::new(0.0, 0.6, 0.0), 0.0, true);
        assert!((dist - 0.1).abs() < 1e-6);
        assert_eq!(kind, PRIM_KIND_HYPERSPHERE4D);
        let (_, kind) = union.eval_at(Vec3::new(2.0, -0.3, 0.0), 0.0, true);
        assert_eq!(kind, PRIM_KIND_HALFSPACE4D);

        let intersection = Scene4::new(ball.clone().intersect(floor.clone()));
        let (dist, kind) = intersection.eval_at(Vec3::new(0.0, 0.6, 0.0), 0.0, true);
        assert!((dist - 1.0).abs() < 1e-6);
        assert_eq!(kind, PRIM_KIND_HALFSPACE4D);

        let difference = Scene4::new(ball.subtract(floor));
        let (_, kind) = difference.eval_at(Vec3::ZERO, 0.0, true);
        assert_eq!(kind, PRIM_KIND_OTHER);
    }

    #[test]
    fn hyperslice_radius_shrinks_with_the_slice_coordinate() {
        use glam::Vec4;
        let radius = 0.5_f32;
        let scene = Scene4::new(SceneNode4::hypersphere(Vec4::ZERO, radius));
        for w in [0.0_f32, 0.2, 0.4] {
            let sliced_radius = (radius * radius - w * w).sqrt();
            assert!(
                scene
                    .eval(Vec3::new(sliced_radius, 0.0, 0.0), w, true)
                    .abs()
                    < 1e-6,
                "w={w}: sliced surface point must read zero",
            );
        }
        assert!(scene.eval(Vec3::ZERO, 0.6, true) > 0.0);
    }

    #[test]
    fn hyperslice_gate_off_returns_the_sentinel_for_halfspaces_only() {
        use glam::Vec4;
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::ZERO, 0.5).union(SceneNode4::halfspace(Vec4::Y, -0.4)),
        );
        let just_above_floor = Vec3::new(3.0, -0.39, 0.0);
        assert!(scene.eval(just_above_floor, 0.0, true) < 0.02);
        let ball_only = Scene4::new(SceneNode4::hypersphere(Vec4::ZERO, 0.5));
        assert_eq!(
            scene.eval(just_above_floor, 0.0, false),
            ball_only.eval(just_above_floor, 0.0, true),
        );
    }

    #[test]
    fn scene4_eval_is_lipschitz_1_in_flat_r4() {
        use glam::Vec4;
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::new(0.1, 0.0, -0.1, 0.05), 0.4)
                .union(SceneNode4::halfspace(Vec4::Y, -0.5))
                .subtract(SceneNode4::hypersphere(Vec4::new(0.3, 0.0, 0.0, 0.0), 0.15)),
        );
        let mut state: u32 = 0x5555_3333;
        let mut next_f32 = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for _ in 0..256 {
            let a = Vec4::new(next_f32(), next_f32(), next_f32(), next_f32()) * 2.0;
            let b = Vec4::new(next_f32(), next_f32(), next_f32(), next_f32()) * 2.0;
            let separation = (a - b).length();
            if separation < 1e-6 {
                continue;
            }
            let delta =
                (scene.eval(a.truncate(), a.w, true) - scene.eval(b.truncate(), b.w, true)).abs();
            assert!(
                delta <= separation * (1.0 + 1e-5),
                "|sdf({a:?}) - sdf({b:?})| = {delta} exceeds |a - b| = {separation}",
            );
        }
    }

    const BEYOND_ABSTRACT_INT: f32 = 1.0e19;

    fn assert_naga_accepts(source: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("WGSL parse failed: {e}\n--- source ---\n{source}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("WGSL validation failed: {e:?}\n--- source ---\n{source}"));
    }

    #[test]
    fn scene3_beyond_abstract_int_range_emits_wgsl_naga_accepts() {
        use loam_math::{EuclideanR3, WgslSpace};
        let magnitude = BEYOND_ABSTRACT_INT;
        let scene = Scene::new(
            SceneNode::sphere(Vec3::new(magnitude, -magnitude, 0.0), magnitude)
                .union(SceneNode::plane(Vec3::Y, -magnitude))
                .smooth_union(SceneNode::box_(Vec3::splat(magnitude)), magnitude),
        );
        let probe = format!(
            "{prelude}\n{scene}\n\
             @compute @workgroup_size(1) fn main() {{\n\
             \t_ = loam_scene_sdf(vec3<f32>(0.0));\n\
             }}\n",
            prelude = EuclideanR3.wgsl_impl(),
            scene = scene.to_wgsl(&EuclideanR3),
        );
        assert_naga_accepts(&probe);
    }

    #[test]
    fn scene4_beyond_abstract_int_range_emits_wgsl_naga_accepts() {
        use glam::Vec4;
        let magnitude = BEYOND_ABSTRACT_INT;
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::splat(magnitude), magnitude)
                .union(SceneNode4::halfspace(Vec4::Y, -magnitude)),
        );
        let native = format!(
            "{scene}\n\
             @compute @workgroup_size(1) fn main() {{\n\
             \t_ = loam_scene_sdf_4d(vec4<f32>(0.0));\n\
             }}\n",
            scene = scene.to_wgsl_4d(),
        );
        assert_naga_accepts(&native);

        let hyperslice = format!(
            "{scene}\n\
             @compute @workgroup_size(1) fn main() {{\n\
             \t_ = loam_scene_sdf(vec3<f32>(0.0));\n\
             \t_ = loam_scene_max_t(vec3<f32>(0.0), vec3<f32>(0.0, -1.0, 0.0));\n\
             }}\n",
            scene = scene.to_hyperslice_wgsl("0.0"),
        );
        assert_naga_accepts(&hyperslice);
    }

    #[test]
    #[should_panic(expected = "non-finite")]
    fn scene3_rejects_a_non_finite_constant() {
        use loam_math::EuclideanR3;
        let scene = Scene::new(SceneNode::sphere(Vec3::ZERO, f32::INFINITY));
        let _ = scene.to_wgsl(&EuclideanR3);
    }

    #[test]
    #[should_panic(expected = "non-finite")]
    fn scene4_rejects_a_non_finite_constant() {
        use glam::Vec4;
        let scene = Scene4::new(SceneNode4::halfspace(Vec4::Y, f32::NAN));
        let _ = scene.to_wgsl_4d();
    }
}
