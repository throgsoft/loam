//! Input transitions preserve arrival order; overflow releases held input and keeps the latest control state.
//! A pointer move replaces its pointer's newest queued move unless a transition sits between them.

use super::message::{InputMessage, PointerPhase};
use std::cell::RefCell;
use std::collections::VecDeque;

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
        InputMessage::Viewport { .. } => Some(5),
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
        let msg = match msg {
            InputMessage::Host { topic, values } if !values.is_empty() => {
                let earlier = q
                    .iter_mut()
                    .rev()
                    .filter_map(|queued| match queued {
                        InputMessage::Host { topic, values } => Some((topic, values)),
                        _ => None,
                    })
                    .take_while(|(_, queued_values)| !queued_values.is_empty())
                    .find(|(queued_topic, _)| **queued_topic == topic)
                    .map(|(_, queued_values)| queued_values);
                if let Some(earlier) = earlier {
                    *earlier = values;
                    return;
                }
                InputMessage::Host { topic, values }
            }
            msg => msg,
        };
        if let Some(index) = coalesced_move(&q, &msg) {
            q.remove(index);
        }
        if q.len() >= MESSAGE_QUEUE_CAPACITY {
            let mut latest = [None; 6];
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
