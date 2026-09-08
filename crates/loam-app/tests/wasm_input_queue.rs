#[path = "../src/wasm/input_queue.rs"]
#[allow(dead_code)]
mod input_queue;

use input_queue::{drain_messages_into, enqueue, InputMessage, MESSAGE_QUEUE_CAPACITY};
use std::collections::VecDeque;

#[test]
fn overflow_releases_held_input_before_new_events() {
    enqueue(InputMessage::Resize {
        width: 400,
        height: 300,
        dpr: 1.0,
    });
    enqueue(InputMessage::PointerLockChanged(false));
    enqueue(InputMessage::Resize {
        width: 800,
        height: 600,
        dpr: 2.0,
    });
    enqueue(InputMessage::PointerLockChanged(true));
    enqueue(InputMessage::Start);
    enqueue(InputMessage::Visibility(true));
    for _ in 0..MESSAGE_QUEUE_CAPACITY - 6 {
        enqueue(InputMessage::MouseWheel { dx: 0.0, dy: 1.0 });
    }
    enqueue(InputMessage::MouseWheel { dx: 0.0, dy: 2.0 });
    let mut batch = VecDeque::new();
    drain_messages_into(&mut batch);
    assert_eq!(batch.len(), 6);
    assert!(matches!(
        batch.pop_front(),
        Some(InputMessage::Focus(false))
    ));
    assert!(matches!(
        batch.pop_front(),
        Some(InputMessage::Resize {
            width: 800,
            height: 600,
            dpr: 2.0
        })
    ));
    assert!(matches!(
        batch.pop_front(),
        Some(InputMessage::PointerLockChanged(true))
    ));
    assert!(matches!(batch.pop_front(), Some(InputMessage::Start)));
    assert!(matches!(
        batch.pop_front(),
        Some(InputMessage::Visibility(true))
    ));
    assert!(matches!(
        batch.pop_front(),
        Some(InputMessage::MouseWheel { dy: 2.0, .. })
    ));
    drain_messages_into(&mut batch);
    assert!(batch.is_empty());
}
