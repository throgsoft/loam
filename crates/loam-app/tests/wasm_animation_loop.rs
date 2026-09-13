#[path = "../src/session/animation.rs"]
#[allow(dead_code)]
mod animation;

use animation::{Lifecycle, Next};

#[test]
fn recovery_schedules_once_after_success_and_never_while_paused_or_failed() {
    let mut animated = 0;
    let mut scheduled = 0;
    let mut lost = true;
    let mut lifecycle = Lifecycle::Ready;
    for _ in 0..2 {
        let next = animation::frame(false, &mut lifecycle, lost, || {
            animated += 1;
            Ok(())
        });
        let next = match next {
            Next::Recover => {
                lost = false;
                assert_eq!(animation::resumed(true, true, false, lifecycle), Next::Idle);
                animation::recovered(&mut lifecycle, false, false, Ok(()))
            }
            other => other,
        };
        match next {
            Next::Frame => scheduled += 1,
            other => panic!("the worker stopped its animation loop at {other:?}"),
        }
    }
    assert_eq!((animated, scheduled), (1, 2));

    lifecycle = Lifecycle::Recovering;
    assert_eq!(
        animation::recovered(&mut lifecycle, true, false, Ok(())),
        Next::Idle
    );
    assert_eq!(
        animation::resumed(true, true, false, lifecycle),
        Next::Frame
    );

    lifecycle = Lifecycle::Recovering;
    assert!(matches!(
        animation::recovered(&mut lifecycle, false, false, Err("no device".into())),
        Next::Failed(_)
    ));
    assert_eq!(animation::resumed(true, true, false, lifecycle), Next::Idle);

    lifecycle = Lifecycle::Ready;
    assert!(matches!(
        animation::frame(false, &mut lifecycle, false, || Err(
            loam_runtime::host::HostError::Host("stopped".into())
        )),
        Next::Failed(_)
    ));
    assert_eq!(animation::resumed(true, true, false, lifecycle), Next::Idle);
}
