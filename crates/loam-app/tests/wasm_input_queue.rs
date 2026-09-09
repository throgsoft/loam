#[path = "../src/wasm/input_queue.rs"]
#[allow(dead_code)]
mod input_queue;

use input_queue::{drain_messages_into, enqueue, InputMessage, MESSAGE_QUEUE_CAPACITY};
use loam_input::PointerPhase;
use std::collections::VecDeque;
use std::time::Duration;

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

fn pointer(id: u64, phase: PointerPhase, x: f32, millis: u64) -> InputMessage {
    InputMessage::Pointer {
        id,
        x,
        y: 0.0,
        phase,
        time: Duration::from_millis(millis),
    }
}

#[test]
fn coalesced_pointer_motion_keeps_the_latest_sample_and_never_crosses_a_release() {
    enqueue(pointer(1, PointerPhase::Down, 0.0, 0));
    enqueue(pointer(1, PointerPhase::Move, 10.0, 10));
    enqueue(pointer(2, PointerPhase::Move, 5.0, 12));
    enqueue(pointer(1, PointerPhase::Move, 20.0, 20));
    enqueue(pointer(1, PointerPhase::Up, 20.0, 25));
    enqueue(pointer(1, PointerPhase::Move, 30.0, 30));
    enqueue(pointer(1, PointerPhase::Move, 40.0, 40));
    let mut batch = VecDeque::new();
    drain_messages_into(&mut batch);
    let got: Vec<(u64, PointerPhase, f32, u128)> = batch
        .iter()
        .map(|msg| match msg {
            InputMessage::Pointer {
                id, x, phase, time, ..
            } => (*id, *phase, *x, time.as_millis()),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(
        got,
        [
            (1, PointerPhase::Down, 0.0, 0),
            (2, PointerPhase::Move, 5.0, 12),
            (1, PointerPhase::Move, 20.0, 20),
            (1, PointerPhase::Up, 20.0, 25),
            (1, PointerPhase::Move, 40.0, 40),
        ]
    );
}
