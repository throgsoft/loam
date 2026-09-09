#[path = "../src/session/animation.rs"]
#[allow(dead_code)]
mod animation;

use animation::Next;

#[test]
fn a_recovered_device_schedules_the_frame_its_loss_branch_returned_from() {
    let mut animated = 0;
    let mut scheduled = 0;
    let mut lost = true;
    for _ in 0..2 {
        let next = animation::frame(false, lost, || {
            animated += 1;
            Ok(())
        });
        let next = match next {
            Next::Recover => {
                lost = false;
                animation::recovered(false, false, Ok(()))
            }
            other => other,
        };
        match next {
            Next::Frame => scheduled += 1,
            other => panic!("the worker stopped its animation loop at {other:?}"),
        }
    }
    assert_eq!((animated, scheduled), (1, 2));
    assert!(matches!(
        animation::recovered(false, false, Err("no device".into())),
        Next::Failed(_)
    ));
}
