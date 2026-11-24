use std::time::Duration;

// Pixels in one wheel line when a native pixel scroll delta is converted to lines.
pub const SCROLL_PIXELS_PER_LINE: f32 = 50.0;

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
