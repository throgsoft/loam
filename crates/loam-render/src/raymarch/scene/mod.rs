//! Emit contract, shared by the 3D and 4D paths: every baked constant goes
//! through `literal::wgsl_f32`, so parsing the emitted literal recovers the
//! exact input bits and the emitter contributes no floor to CPU/GPU parity.

mod literal;
pub mod primitive;
pub mod primitive4;
pub mod scene3;
pub mod scene4;
mod shapes;

pub use loam_shape::Shape;
pub use primitive::Primitive;
pub use primitive4::Primitive4;
pub use scene3::{PrimitiveKind, Scene, SceneNode};
pub use scene4::{
    Scene4, SceneNode4, PRIM_KIND_HALFSPACE4D, PRIM_KIND_HYPERSPHERE4D, PRIM_KIND_OTHER,
};

/// Returned by shapes with no closed-form SDF in the emitted dimension.
pub const SENTINEL_DISTANCE: f32 = 1e9;

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec3, Vec4};

    #[test]
    fn hyperslice_eval_reports_the_sliced_distance_and_kind_and_is_lipschitz_1() {
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
                "R4: |sdf({a:?}) - sdf({b:?})| = {delta} exceeds |a - b| = {separation}",
            );
        }
    }
}
