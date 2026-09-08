//! Host rotation drives unowned slots; timeline-owned slots retain separate rotors.

use loam_math::Rotor4;

use crate::state::{active_plane_angle, compose_active_rotor};

// Slots past the mask remain host-owned.
pub(crate) fn is_directed(directed: &[bool], slot: usize) -> bool {
    directed.get(slot).copied().unwrap_or(false)
}

const DEFAULT_ACTIVE: [bool; 6] = [false, false, true, false, false, false];

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SlotSpin {
    /// Base angles exclude the accumulated active spin, in radians.
    pub(crate) base_angles: [f32; 6],
    pub(crate) active: [bool; 6],
}

impl Default for SlotSpin {
    fn default() -> Self {
        Self {
            base_angles: [0.0; 6],
            active: DEFAULT_ACTIVE,
        }
    }
}

impl SlotSpin {
    pub(crate) fn angle_at(&self, plane_idx: usize, t: f32) -> f32 {
        active_plane_angle(self.base_angles[plane_idx], self.active[plane_idx], t)
    }

    pub(crate) fn active_rotor_at(&self, t: f32) -> Rotor4 {
        compose_active_rotor(&self.base_angles, &self.active, t)
    }
}

pub(crate) struct SlotSpins {
    spin: SlotSpin,
    rotor: Rotor4,
    /// Timeline-owned slots retain their sampled orientations.
    rotors: Vec<Rotor4>,
}

impl SlotSpins {
    pub(crate) fn new(slots: usize) -> Self {
        Self {
            spin: SlotSpin::default(),
            rotor: Rotor4::IDENTITY,
            rotors: vec![Rotor4::IDENTITY; slots.max(1)],
        }
    }

    #[cfg(test)]
    pub(crate) fn uniform(slots: usize, rotor: Rotor4) -> Self {
        let mut spins = Self::new(slots);
        spins.set_row_rotor(rotor, &[]);
        spins
    }

    pub(crate) fn sync(&mut self, slots: usize) {
        self.rotors.resize(slots.max(1), self.rotor);
    }

    pub(crate) fn rotor(&self, slot: usize) -> Rotor4 {
        self.rotors[slot]
    }

    pub(crate) fn row_rotor(&self) -> Rotor4 {
        self.rotor
    }

    pub(crate) fn spin(&self) -> &SlotSpin {
        &self.spin
    }

    pub(crate) fn spin_mut(&mut self) -> &mut SlotSpin {
        &mut self.spin
    }

    pub(crate) fn recompose_active(&mut self, t: f32, directed: &[bool]) {
        self.set_row_rotor(self.spin.active_rotor_at(t), directed);
    }

    pub(crate) fn set_row_rotor(&mut self, rotor: Rotor4, directed: &[bool]) {
        self.rotor = rotor;
        for (slot, held) in self.rotors.iter_mut().enumerate() {
            if !is_directed(directed, slot) {
                *held = rotor;
            }
        }
    }

    pub(crate) fn any_unowned(&self, directed: &[bool]) -> bool {
        (0..self.rotors.len()).any(|slot| !is_directed(directed, slot))
    }

    pub(crate) fn set_rotor(&mut self, slot: usize, rotor: Rotor4) {
        if let Some(held) = self.rotors.get_mut(slot) {
            *held = rotor;
        }
    }

    pub(crate) fn rotors_differ_from(&self, uploaded: &[Rotor4]) -> bool {
        self.rotors != uploaded
    }

    pub(crate) fn record_rotors(&self, out: &mut Vec<Rotor4>) {
        out.clear();
        out.extend_from_slice(&self.rotors);
    }

    // Clear base angles before the next active recompose.
    pub(crate) fn clear_orientation(&mut self) {
        self.spin.base_angles = [0.0; 6];
        self.rotor = Rotor4::IDENTITY;
        self.rotors.fill(Rotor4::IDENTITY);
    }

    pub(crate) fn reset(&mut self) {
        self.spin = SlotSpin::default();
        self.clear_orientation();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::{Bivector, Plane4};

    #[test]
    fn a_directed_slot_is_skipped_by_the_ui_clock_and_its_neighbours_are_not() {
        let mut spins = SlotSpins::new(3);
        let held = compose_active_rotor(&[0.9, 0.0, 0.0, 0.0, 0.0, 0.0], &[false; 6], 0.0);
        spins.set_rotor(1, held);

        spins.recompose_active(2.5, &[false, true]);
        assert_eq!(spins.rotor(1), held, "the directed slot was recomposed");
        assert_ne!(spins.rotor(0), Rotor4::IDENTITY, "slot 0 was skipped");
        assert_eq!(
            spins.rotor(2),
            spins.rotor(0),
            "the mask tail is the clock's"
        );

        assert!(spins.any_unowned(&[false, true]));
        assert!(spins.any_unowned(&[true, true]), "slot 2 is past the mask");
        assert!(!spins.any_unowned(&[true; 3]));
    }

    #[test]
    fn a_slot_that_arrives_mid_spin_joins_the_rows_orientation() {
        let mut spins = SlotSpins::new(2);
        spins.recompose_active(3.0, &[]);
        let turning = spins.row_rotor();
        assert_ne!(turning, Rotor4::IDENTITY);

        spins.sync(4);
        for slot in 0..4 {
            assert_eq!(spins.rotor(slot), turning, "slot {slot} joined at identity");
        }

        spins.sync(1);
        assert!(!spins.rotors_differ_from(&[turning]), "the row was rebuilt");
    }

    #[test]
    fn clearing_orientation_survives_the_next_recompose() {
        let mut spins = SlotSpins::new(2);
        spins.spin_mut().base_angles = [0.4, -0.2, 1.1, 0.0, 0.3, -0.9];
        spins.recompose_active(2.0, &[]);

        spins.clear_orientation();
        spins.recompose_active(0.0, &[]);
        for slot in 0..2 {
            assert_eq!(spins.rotor(slot), Rotor4::IDENTITY, "slot {slot} came back");
        }
        assert_eq!(
            spins.spin().active,
            DEFAULT_ACTIVE,
            "clearing the orientation also cleared the plane mask"
        );
    }

    #[test]
    fn a_turn_of_the_row_or_of_one_directed_slot_makes_the_upload_stale() {
        let mut spins = SlotSpins::new(4);
        let mut uploaded = Vec::new();
        spins.record_rotors(&mut uploaded);
        assert!(!spins.rotors_differ_from(&uploaded));

        spins.spin_mut().base_angles[1] = 0.3;
        spins.recompose_active(0.0, &[]);
        assert!(
            spins.rotors_differ_from(&uploaded),
            "turning the row left the upload gate closed"
        );
        spins.record_rotors(&mut uploaded);

        let quarter = (Plane4::Xw.unit_bivector() * std::f32::consts::FRAC_PI_4).exp();
        for slot in 0..4 {
            spins.set_rotor(slot, quarter);
            assert!(
                spins.rotors_differ_from(&uploaded),
                "posing slot {slot} alone left the upload gate closed"
            );
            spins.record_rotors(&mut uploaded);
            assert!(!spins.rotors_differ_from(&uploaded));
        }

        spins.sync(3);
        assert!(spins.rotors_differ_from(&uploaded));
    }
}
