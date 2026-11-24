use glam::{Vec3, Vec4};
use loam_math::{Plane4, Rotor};
use loam_render::hypergimbal::{Hypergimbal, RingDrag};

const WIDGET: Hypergimbal = Hypergimbal {
    center: Vec3::new(0.3, -0.7, 1.1),
    scale: 0.8,
};

#[test]
fn ring_drag() {
    for (plane, a, b) in [
        (Plane4::Xy, Vec4::X, Vec4::Y),
        (Plane4::Zw, Vec4::Z, Vec4::W),
    ] {
        let ring = WIDGET.ring(plane);
        let eye = ring.center + ring.normal() * 3.0 + ring.u * 0.4;
        let on_circle = |theta: f32| a * theta.cos() + b * theta.sin();
        let toward = |point: Vec4| WIDGET.project(point).expect("off the pole") - eye;

        for (theta, delta) in [(-2.0_f32, 0.9_f32), (0.3, -1.1), (1.7, 2.2)] {
            let (start, end) = (on_circle(theta), on_circle(theta + delta));
            let drag = RingDrag::press(ring, eye, toward(start)).expect("the press hit the plane");
            let rotor = drag
                .rotor(eye, toward(end))
                .expect("the cursor hit the plane");
            let turned = rotor.apply(start);
            assert!(
                (turned - end).length() < 1e-3,
                "{plane:?} drag from {theta} by {delta} turned the grab to {turned}, not {end}"
            );
            assert_eq!(
                drag.rotor(eye, ring.u),
                None,
                "{plane:?} drag made a rotor from a ray along the ring's plane"
            );
        }
        assert_eq!(
            RingDrag::press(ring, eye, ring.u),
            None,
            "{plane:?} press along the ring's plane started a drag"
        );
    }
}
