use loam_app::session::input::{apply, InputMap};
use loam_app::wasm::input_queue::{
    drain_messages_into, enqueue, InputMessage, PointerPhase, MESSAGE_QUEUE_CAPACITY,
};
use loam_runtime::PointerPhase as RuntimePointerPhase;
use std::collections::VecDeque;
use std::time::Duration;

#[test]
fn overflow_releases_held_input_before_new_events() {
    enqueue(InputMessage::Resize {
        width: 400,
        height: 300,
        dpr: 1.0,
    });
    enqueue(InputMessage::PointerLockChanged {
        locked: false,
        released: false,
    });
    enqueue(InputMessage::Resize {
        width: 800,
        height: 600,
        dpr: 2.0,
    });
    enqueue(InputMessage::PointerLockChanged {
        locked: true,
        released: false,
    });
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
        Some(InputMessage::PointerLockChanged { locked: true, .. })
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

#[test]
fn overflow_discards_pointer_bursts_but_cancels_active_drag_and_keeps_controls() {
    let bindings = loam_runtime::Bindings::new();
    let mut map = InputMap::default();
    let mut batch = VecDeque::new();
    apply(
        &mut map,
        &bindings,
        &pointer(7, PointerPhase::Down, 1.0, 1),
        false,
    );
    let _ = map.take();

    for burst in 0..MESSAGE_QUEUE_CAPACITY * 2 {
        let id = burst as u64;
        enqueue(pointer(id, PointerPhase::Down, 1.0, 1));
        enqueue(pointer(id, PointerPhase::Up, 3.0, 3));
    }
    drain_messages_into(&mut batch);
    assert!(batch.len() <= MESSAGE_QUEUE_CAPACITY);
    batch.clear();

    enqueue(InputMessage::Resize {
        width: 1920,
        height: 1080,
        dpr: 2.0,
    });
    enqueue(InputMessage::Visibility(false));
    for burst in 0..MESSAGE_QUEUE_CAPACITY - 2 {
        let id = (1000 + burst) as u64;
        let phase = if burst % 2 == 0 {
            PointerPhase::Down
        } else {
            PointerPhase::Up
        };
        enqueue(pointer(id, phase, 4.0, 4));
    }
    enqueue(pointer(99_999, PointerPhase::Move, 7.0, 7));

    drain_messages_into(&mut batch);
    assert!(batch.len() <= MESSAGE_QUEUE_CAPACITY);
    assert!(matches!(batch.front(), Some(InputMessage::Focus(false))));
    assert!(batch.iter().any(|message| matches!(
        message,
        InputMessage::Resize {
            width: 1920,
            height: 1080,
            dpr: 2.0
        }
    )));
    assert!(batch
        .iter()
        .any(|message| matches!(message, InputMessage::Visibility(false))));
    let pointers: Vec<_> = batch
        .iter()
        .filter_map(|message| match message {
            InputMessage::Pointer { id, phase, .. } => Some((*id, *phase)),
            _ => None,
        })
        .collect();
    assert_eq!(
        pointers,
        [(99_999, PointerPhase::Move)],
        "overflow retained stale pointer transitions"
    );

    for message in &batch {
        apply(&mut map, &bindings, message, false);
    }
    let input = map.take();
    assert_eq!(map.size(), (1920, 1080));
    assert_eq!(map.scale(), 2.0);
    assert!(input
        .pointers
        .iter()
        .any(|pointer| { pointer.id == 7 && pointer.phase == RuntimePointerPhase::Cancelled }));
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
fn a_streaming_host_topic_never_purges_a_queued_one_shot_message() {
    enqueue(InputMessage::Host {
        topic: "reset".into(),
        values: Vec::new(),
    });
    for i in 0..=MESSAGE_QUEUE_CAPACITY {
        enqueue(InputMessage::Host {
            topic: "scroll".into(),
            values: vec![i as f32],
        });
    }
    let mut batch = VecDeque::new();
    drain_messages_into(&mut batch);
    let got: Vec<(&str, &[f32])> = batch
        .iter()
        .map(|msg| match msg {
            InputMessage::Host { topic, values } => (topic.as_str(), values.as_slice()),
            other => panic!("{other:?}"),
        })
        .collect();
    let last = MESSAGE_QUEUE_CAPACITY as f32;
    assert_eq!(got, [("reset", &[][..]), ("scroll", &[last][..])]);
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
