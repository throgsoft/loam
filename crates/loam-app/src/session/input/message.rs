use std::time::Duration;

// A browser reports one wheel notch as 100 pixels and winit as one line, so a notch is one line on both.
const WHEEL_PIXELS_PER_LINE: f32 = 100.0;

pub(crate) fn wheel_lines(pixels: [f32; 2]) -> [f32; 2] {
    pixels.map(|p| p / WHEEL_PIXELS_PER_LINE)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerPhase {
    Down,
    Move,
    Up,
    Cancel,
}

#[derive(Debug)]
pub enum InputMessage {
    /// Physical canvas size and the DPR measured with it.
    Resize {
        width: u32,
        height: u32,
        dpr: f32,
    },

    /// Canvas CSS size, sent on every frame it changes; `Resize` follows once it settles.
    Viewport {
        width: f32,
        height: f32,
    },

    /// Canvas-local CSS position and accumulated DOM `movementX/Y` deltas.
    MouseMove {
        x: f32,
        y: f32,
        buttons: u8,
        dx: f32,
        dy: f32,
        time: Duration,
    },

    /// `button` is `MouseEvent.button` (0=primary, 1=middle, 2=secondary).
    MouseButton {
        x: f32,
        y: f32,
        button: u8,
        pressed: bool,
        time: Duration,
    },

    /// Lines in DOM convention: positive is right/down.
    MouseWheel {
        dx: f32,
        dy: f32,
    },

    /// DOM `code` identifies the physical key; `key` carries logical text.
    Key {
        code: String,
        key: String,
        pressed: bool,
        repeat: bool,
        ctrl: bool,
        shift: bool,
        alt: bool,
        meta: bool,
    },

    Focus(bool),

    Visibility(bool),

    Start,

    /// Browser-confirmed lock state, including releases by Esc or focus loss.
    PointerLockChanged {
        locked: bool,
        released: bool,
    },

    /// Canvas-local CSS position and the DOM `timeStamp`.
    Pointer {
        id: u64,
        x: f32,
        y: f32,
        phase: PointerPhase,
        time: Duration,
    },

    /// A page `loam-message` event, delivered whether or not the embed is paused.
    Host {
        topic: String,
        values: Vec<f32>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wheel_notch() {
        // A browser in pixel mode reports one wheel notch as a deltaY of 100.
        assert_eq!(wheel_lines([0.0, 100.0]), [0.0, 1.0]);
    }
}
