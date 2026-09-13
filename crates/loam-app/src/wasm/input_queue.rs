//! Input transitions preserve arrival order; overflow releases held input and keeps the latest control state.
//! A pointer move replaces its pointer's newest queued move unless a transition sits between them.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::time::Duration;

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
        InputMessage::PointerLockChanged { .. } => Some(3),
        InputMessage::Focus(_) => Some(4),
        _ => None,
    }
}

fn coalesced_move(q: &VecDeque<InputMessage>, msg: &InputMessage) -> Option<usize> {
    let InputMessage::Pointer {
        id,
        phase: PointerPhase::Move,
        ..
    } = msg
    else {
        return None;
    };
    q.iter()
        .enumerate()
        .rev()
        .map_while(|(index, queued)| match queued {
            InputMessage::Pointer {
                id: queued_id,
                phase: PointerPhase::Move,
                ..
            } => Some((index, queued_id == id)),
            _ => None,
        })
        .find(|(_, same_pointer)| *same_pointer)
        .map(|(index, _)| index)
}

pub fn enqueue(msg: InputMessage) {
    MESSAGE_QUEUE.with(|q| {
        let mut q = q.borrow_mut();
        if let Some(index) = coalesced_move(&q, &msg) {
            q.remove(index);
        }
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
