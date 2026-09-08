use glam::Vec2;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::{KeyCode, PhysicalKey};

pub const SCROLL_PIXELS_PER_LINE: f32 = 50.0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ButtonState {
    pub down: bool,
    /// `None` while up, or when the press came with no known cursor position.
    pub press_pos: Option<Vec2>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MouseButtons {
    pub left: ButtonState,
    pub right: ButtonState,
    pub middle: ButtonState,
}

impl MouseButtons {
    fn slot(&mut self, button: MouseButton) -> Option<&mut ButtonState> {
        match button {
            MouseButton::Left => Some(&mut self.left),
            MouseButton::Right => Some(&mut self.right),
            MouseButton::Middle => Some(&mut self.middle),
            _ => None,
        }
    }
}

/// Derived from held physical keys.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Modifiers {
    pub shift: bool,
    pub control: bool,
    pub alt: bool,
    pub super_key: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameInput {
    /// OS-clamped cursor delta (`CursorMoved`); stops at the screen edge.
    pub mouse_delta: Vec2,
    /// Raw device delta (`MouseMotion`); accumulates past the screen edge.
    pub mouse_raw_delta: Vec2,
    pub scroll_lines: f32,
    pub buttons: MouseButtons,
    /// Physical pixels; `None` until `CursorMoved` and after the cursor leaves.
    pub cursor_pos: Option<Vec2>,
    pub modifiers: Modifiers,
    /// W = +1, S = −1.
    pub move_forward: f32,
    /// D = +1, A = −1.
    pub move_right: f32,
    /// Space = +1, Left/Right Shift = −1.
    pub move_up: f32,
}

#[derive(Debug, Default)]
pub struct InputState {
    frame: FrameInput,
    held_keys: std::collections::HashSet<KeyCode>,
}

impl InputState {
    pub fn cursor_moved(&mut self, x: f64, y: f64) {
        let pos = Vec2::new(x as f32, y as f32);
        if let Some(last) = self.frame.cursor_pos {
            self.frame.mouse_delta += pos - last;
        }
        self.frame.cursor_pos = Some(pos);
    }

    /// Call on `CursorLeft` and focus loss.
    pub fn cursor_invalidated(&mut self) {
        self.frame.cursor_pos = None;
    }

    /// Call on focus loss.
    pub fn release_buttons(&mut self) {
        self.held_keys.clear();
        self.frame = FrameInput::default();
    }

    pub fn mouse_input(&mut self, button: MouseButton, state: ElementState) {
        let cursor = self.frame.cursor_pos;
        let pressed = state == ElementState::Pressed;
        if let Some(slot) = self.frame.buttons.slot(button) {
            slot.down = pressed;
            slot.press_pos = if pressed { cursor } else { None };
        }
    }

    pub fn mouse_wheel(&mut self, delta: MouseScrollDelta) {
        self.frame.scroll_lines += match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / SCROLL_PIXELS_PER_LINE,
        };
    }

    pub fn key_input(&mut self, physical_key: PhysicalKey, state: ElementState) {
        if let PhysicalKey::Code(code) = physical_key {
            match state {
                ElementState::Pressed => {
                    self.held_keys.insert(code);
                }
                ElementState::Released => {
                    self.held_keys.remove(&code);
                }
            }
        }
    }

    /// Resets the deltas; held state persists into the next frame.
    pub fn take_frame(&mut self) -> FrameInput {
        let held = &self.held_keys;
        self.frame.move_forward = axis(held, KeyCode::KeyW, KeyCode::KeyS);
        self.frame.move_right = axis(held, KeyCode::KeyD, KeyCode::KeyA);
        self.frame.move_up = held.contains(&KeyCode::Space) as u8 as f32
            - either(held, KeyCode::ShiftLeft, KeyCode::ShiftRight) as u8 as f32;
        self.frame.modifiers = Modifiers {
            shift: either(held, KeyCode::ShiftLeft, KeyCode::ShiftRight),
            control: either(held, KeyCode::ControlLeft, KeyCode::ControlRight),
            alt: either(held, KeyCode::AltLeft, KeyCode::AltRight),
            super_key: either(held, KeyCode::SuperLeft, KeyCode::SuperRight),
        };

        let frame = self.frame;
        self.frame.mouse_delta = Vec2::ZERO;
        self.frame.mouse_raw_delta = Vec2::ZERO;
        self.frame.scroll_lines = 0.0;
        frame
    }

    /// Source is winit's `DeviceEvent::MouseMotion`.
    #[doc(hidden)]
    pub fn accumulate_raw_motion(&mut self, dx: f64, dy: f64) {
        self.frame.mouse_raw_delta += Vec2::new(dx as f32, dy as f32);
    }
}

fn axis(held: &std::collections::HashSet<KeyCode>, pos: KeyCode, neg: KeyCode) -> f32 {
    let p = held.contains(&pos) as u8 as f32;
    let n = held.contains(&neg) as u8 as f32;
    p - n
}

fn either(held: &std::collections::HashSet<KeyCode>, left: KeyCode, right: KeyCode) -> bool {
    held.contains(&left) || held.contains(&right)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32) {
        assert!(
            (a - b).abs() <= 1e-5,
            "expected {a} to be within 1e-5 of {b}"
        );
    }

    #[test]
    fn cursor_invalidated_prevents_jump_delta() {
        let mut input = InputState::default();
        input.cursor_moved(100.0, 100.0);
        input.cursor_invalidated();
        input.cursor_moved(500.0, 500.0);
        let frame = input.take_frame();
        assert_eq!(frame.mouse_delta, Vec2::ZERO);
    }

    #[test]
    fn pixel_delta_scroll_uses_pixels_per_line() {
        let mut input = InputState::default();
        input.mouse_wheel(MouseScrollDelta::PixelDelta(
            winit::dpi::PhysicalPosition::new(0.0, SCROLL_PIXELS_PER_LINE as f64),
        ));
        let frame = input.take_frame();
        assert_close(frame.scroll_lines, 1.0);
    }

    #[test]
    fn take_frame_keeps_button_state_and_clears_deltas() {
        let mut input = InputState::default();
        input.mouse_input(MouseButton::Left, ElementState::Pressed);
        input.cursor_moved(10.0, 20.0);
        input.cursor_moved(15.0, 18.0);
        input.mouse_wheel(MouseScrollDelta::LineDelta(0.0, 2.0));
        input.accumulate_raw_motion(8.0, -3.0);

        let frame = input.take_frame();
        assert!(frame.buttons.left.down);
        assert_eq!(frame.mouse_delta, Vec2::new(5.0, -2.0));
        assert_eq!(frame.scroll_lines, 2.0);
        assert_eq!(frame.mouse_raw_delta, Vec2::new(8.0, -3.0));

        let frame = input.take_frame();
        assert!(frame.buttons.left.down);
        assert_eq!(frame.mouse_delta, Vec2::ZERO);
        assert_eq!(frame.scroll_lines, 0.0);
        assert_eq!(frame.mouse_raw_delta, Vec2::ZERO);
        assert_eq!(frame.cursor_pos, Some(Vec2::new(15.0, 18.0)));
    }

    #[test]
    fn button_press_anchors_at_cursor_and_release_drops_the_anchor() {
        let mut input = InputState::default();
        input.cursor_moved(30.0, 40.0);
        input.mouse_input(MouseButton::Left, ElementState::Pressed);
        input.cursor_moved(90.0, 40.0);

        let frame = input.take_frame();
        assert!(frame.buttons.left.down);
        assert_eq!(frame.buttons.left.press_pos, Some(Vec2::new(30.0, 40.0)));
        assert_eq!(frame.cursor_pos, Some(Vec2::new(90.0, 40.0)));

        input.mouse_input(MouseButton::Left, ElementState::Released);
        let frame = input.take_frame();
        assert!(!frame.buttons.left.down);
        assert_eq!(frame.buttons.left.press_pos, None);
    }

    #[test]
    fn press_with_unknown_cursor_has_no_anchor() {
        let mut input = InputState::default();
        input.mouse_input(MouseButton::Right, ElementState::Pressed);
        assert_eq!(input.take_frame().buttons.right.press_pos, None);

        input.mouse_input(MouseButton::Right, ElementState::Released);
        input.cursor_moved(5.0, 5.0);
        input.cursor_invalidated();
        input.mouse_input(MouseButton::Right, ElementState::Pressed);
        let frame = input.take_frame();
        assert!(frame.buttons.right.down);
        assert_eq!(frame.buttons.right.press_pos, None);
    }

    #[test]
    fn each_button_carries_its_own_state_and_anchor() {
        let mut input = InputState::default();
        input.cursor_moved(1.0, 2.0);
        input.mouse_input(MouseButton::Right, ElementState::Pressed);
        input.cursor_moved(3.0, 4.0);
        input.mouse_input(MouseButton::Middle, ElementState::Pressed);

        let frame = input.take_frame();
        assert!(!frame.buttons.left.down);
        assert_eq!(frame.buttons.right.press_pos, Some(Vec2::new(1.0, 2.0)));
        assert_eq!(frame.buttons.middle.press_pos, Some(Vec2::new(3.0, 4.0)));

        input.mouse_input(MouseButton::Right, ElementState::Released);
        let frame = input.take_frame();
        assert!(!frame.buttons.right.down);
        assert!(frame.buttons.middle.down);
    }

    #[test]
    fn focus_loss_clears_held_keys_buttons_and_pending_motion() {
        let mut input = InputState::default();
        input.cursor_moved(11.0, 12.0);
        input.mouse_input(MouseButton::Left, ElementState::Pressed);
        input.mouse_input(MouseButton::Right, ElementState::Pressed);
        input.mouse_input(MouseButton::Middle, ElementState::Pressed);
        for key in [
            KeyCode::KeyW,
            KeyCode::KeyD,
            KeyCode::Space,
            KeyCode::AltLeft,
            KeyCode::ShiftRight,
            KeyCode::ControlLeft,
            KeyCode::SuperRight,
        ] {
            input.key_input(PhysicalKey::Code(key), ElementState::Pressed);
        }
        let held = input.take_frame();
        assert_eq!(held.move_forward, 1.0);
        assert!(held.modifiers.alt);
        input.cursor_moved(31.0, 22.0);
        input.accumulate_raw_motion(12.0, 3.0);
        input.mouse_wheel(MouseScrollDelta::LineDelta(0.0, 2.0));
        input.release_buttons();

        let frame = input.take_frame();
        assert_eq!(frame.buttons, MouseButtons::default());
        assert_eq!(frame.modifiers, Modifiers::default());
        assert_eq!(
            [frame.move_forward, frame.move_right, frame.move_up],
            [0.0; 3]
        );
        assert_eq!(frame.mouse_delta, Vec2::ZERO);
        assert_eq!(frame.mouse_raw_delta, Vec2::ZERO);
        assert_eq!(frame.scroll_lines, 0.0);
        input.cursor_moved(300.0, 200.0);
        assert_eq!(input.take_frame().mouse_delta, Vec2::ZERO);
    }

    #[test]
    fn releasing_one_modifier_keeps_its_held_pair_active() {
        let shift = Modifiers {
            shift: true,
            ..Modifiers::default()
        };
        let control = Modifiers {
            control: true,
            ..Modifiers::default()
        };
        let alt = Modifiers {
            alt: true,
            ..Modifiers::default()
        };
        let super_key = Modifiers {
            super_key: true,
            ..Modifiers::default()
        };
        let pairs = [
            (KeyCode::ShiftLeft, KeyCode::ShiftRight, shift),
            (KeyCode::ControlLeft, KeyCode::ControlRight, control),
            (KeyCode::AltLeft, KeyCode::AltRight, alt),
            (KeyCode::SuperLeft, KeyCode::SuperRight, super_key),
        ];
        for (left, right, expected) in pairs {
            for key in [left, right] {
                let mut input = InputState::default();
                input.key_input(PhysicalKey::Code(key), ElementState::Pressed);
                assert_eq!(
                    input.take_frame().modifiers,
                    expected,
                    "{key:?} must set exactly {expected:?}"
                );
            }
            let mut input = InputState::default();
            input.key_input(PhysicalKey::Code(left), ElementState::Pressed);
            input.key_input(PhysicalKey::Code(right), ElementState::Pressed);
            input.key_input(PhysicalKey::Code(left), ElementState::Released);
            assert_eq!(input.take_frame().modifiers, expected);
            assert_eq!(input.take_frame().modifiers, expected);
            input.key_input(PhysicalKey::Code(right), ElementState::Released);
            assert_eq!(input.take_frame().modifiers, Modifiers::default());
        }
    }
}
