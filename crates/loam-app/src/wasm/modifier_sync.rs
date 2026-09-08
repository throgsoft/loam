//! Browser modifier flags correct key transitions lost during OS shortcuts.

use winit::keyboard::KeyCode;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModifierFlags {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    pub meta: bool,
}

#[derive(Debug, Default)]
pub struct ModifierSync {
    applied: ModifierFlags,
}

impl ModifierSync {
    pub fn key_event(
        &mut self,
        input: &mut loam_input::InputState,
        code: Option<KeyCode>,
        pressed: bool,
        flags: ModifierFlags,
    ) -> winit::event::ElementState {
        use winit::event::ElementState;
        use winit::keyboard::PhysicalKey;
        let state = |pressed| {
            if pressed {
                ElementState::Pressed
            } else {
                ElementState::Released
            }
        };
        self.reconcile(flags, |code, pressed| {
            input.key_input(PhysicalKey::Code(code), state(pressed))
        });
        if let Some(code) = code {
            input.key_input(PhysicalKey::Code(code), state(pressed));
        }
        state(pressed)
    }

    /// A cleared flag releases both physical keys, including a missed keyup.
    pub fn reconcile(&mut self, flags: ModifierFlags, mut emit: impl FnMut(KeyCode, bool)) {
        let pairs = [
            (
                flags.ctrl,
                self.applied.ctrl,
                KeyCode::ControlLeft,
                KeyCode::ControlRight,
            ),
            (
                flags.shift,
                self.applied.shift,
                KeyCode::ShiftLeft,
                KeyCode::ShiftRight,
            ),
            (
                flags.alt,
                self.applied.alt,
                KeyCode::AltLeft,
                KeyCode::AltRight,
            ),
            (
                flags.meta,
                self.applied.meta,
                KeyCode::SuperLeft,
                KeyCode::SuperRight,
            ),
        ];
        for (now, before, left, right) in pairs {
            match (before, now) {
                (false, true) => emit(left, true),
                (true, false) => {
                    emit(left, false);
                    emit(right, false);
                }
                _ => {}
            }
        }
        self.applied = flags;
    }
}
