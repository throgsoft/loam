//! Input transitions preserve arrival order; overflow releases held input and keeps the latest control state.

use std::cell::RefCell;
use std::collections::VecDeque;

#[derive(Debug)]
pub enum InputMessage {
    /// Physical canvas size and the DPR measured with it.
    Resize {
        width: u32,
        height: u32,
        dpr: f32,
    },

    /// Canvas-local CSS position and accumulated DOM `movementX/Y` deltas.
    MouseMove {
        x: f32,
        y: f32,
        buttons: u8,
        dx: f32,
        dy: f32,
    },

    /// `button` is `MouseEvent.button` (0=primary, 1=middle, 2=secondary).
    MouseButton {
        x: f32,
        y: f32,
        button: u8,
        pressed: bool,
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
    PointerLockChanged(bool),
}

pub const MESSAGE_QUEUE_CAPACITY: usize = 256;

thread_local! {
    static MESSAGE_QUEUE: RefCell<VecDeque<InputMessage>> = const { RefCell::new(VecDeque::new()) };
}

fn control_kind(msg: &InputMessage) -> Option<usize> {
    match msg {
        InputMessage::Resize { .. } => Some(0),
        InputMessage::Visibility(_) => Some(1),
        InputMessage::Start => Some(2),
        InputMessage::PointerLockChanged(_) => Some(3),
        InputMessage::Focus(_) => Some(4),
        _ => None,
    }
}

pub fn enqueue(msg: InputMessage) {
    MESSAGE_QUEUE.with(|q| {
        let mut q = q.borrow_mut();
        if q.len() >= MESSAGE_QUEUE_CAPACITY {
            let mut latest = [None; 5];
            for (index, queued) in q.iter().enumerate() {
                if let Some(kind) = control_kind(queued) {
                    latest[kind] = Some(index);
                }
            }
            let mut index = 0;
            q.retain(|queued| {
                let keep = control_kind(queued).is_some_and(|kind| latest[kind] == Some(index));
                index += 1;
                keep
            });
            q.push_front(InputMessage::Focus(false));
            tracing::warn!("input queue overflow: released held input and discarded stale events");
        }
        q.push_back(msg);
    });
}

pub fn drain_messages_into(batch: &mut VecDeque<InputMessage>) {
    batch.clear();
    MESSAGE_QUEUE.with(|q| std::mem::swap(&mut *q.borrow_mut(), batch));
}

/// Converts DOM coordinates to `FrameInput::cursor_pos` units.
pub fn physical_cursor(x: f32, y: f32, device_pixel_ratio: f32) -> (f64, f64) {
    (
        (x * device_pixel_ratio) as f64,
        (y * device_pixel_ratio) as f64,
    )
}

pub fn pointer_button(
    input: &mut loam_input::InputState,
    x: f32,
    y: f32,
    dpr: f32,
    button: winit::event::MouseButton,
    state: winit::event::ElementState,
) {
    let (x, y) = physical_cursor(x, y, dpr);
    input.cursor_moved(x, y);
    input.mouse_input(button, state);
}
