use std::f32::consts::FRAC_PI_2;

use glam::Vec4;
use loam_math::{Bivector, Plane4, Rotor4};

#[test]
fn near_half_turn_and_reflection() {
    let flip = Rotor4::from_rotation_arc(Vec4::W, -Vec4::W);
    let tilted = |plane: Plane4| (plane.unit_bivector() * (FRAC_PI_2 * 0.05)).exp() * flip;
    let rotors = [
        tilted(Plane4::Xy),
        tilted(Plane4::Xz),
        tilted(Plane4::Yz),
        Rotor4 {
            s: -4.37e-8,
            xy: -2.19e-10,
            xw: 0.999_987_5,
            yw: -0.005,
            ..Rotor4::IDENTITY
        },
    ];
    // Entries are at most 1 and the rebuild is a few hundred f32 operations, so 1e-5 is about 80 ulps.
    let rounding = 1e-5;
    for rotor in rotors {
        let frame = rotor.to_mat4();
        let rebuilt = Rotor4::from_mat4(&frame)
            .unwrap_or_else(|| panic!("{rotor:?} is a rotation and was rejected"));
        assert!(
            (rebuilt.norm_squared() - 1.0).abs() < rounding,
            "{rotor:?} rebuilt as {rebuilt:?}, which is not unit"
        );
        for (slot, (built, given)) in rebuilt
            .to_mat4()
            .iter()
            .flatten()
            .zip(frame.iter().flatten())
            .enumerate()
        {
            assert!(
                (built - given).abs() < rounding,
                "{rotor:?} slot {slot}: rebuilt {built} against {given}"
            );
        }
    }

    let mut mirrored = Rotor4::IDENTITY.to_mat4();
    mirrored[1][1] = -1.0;
    assert_eq!(Rotor4::from_mat4(&mirrored), None);
}
