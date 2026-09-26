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
                assert_eq!(
                    animation::resumed(true, false, true, false, lifecycle),
                    Next::Idle
                );
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
        animation::resumed(true, false, true, false, lifecycle),
        Next::Frame
    );

    lifecycle = Lifecycle::Recovering;
    assert!(matches!(
        animation::recovered(&mut lifecycle, false, false, Err("no device".into())),
        Next::Failed(_)
    ));
    assert_eq!(
        animation::resumed(true, false, true, false, lifecycle),
        Next::Idle
    );

    lifecycle = Lifecycle::Ready;
    assert!(matches!(
        animation::frame(false, &mut lifecycle, false, || Err(
            loam_runtime::host::HostError::Host("stopped".into())
        )),
        Next::Failed(_)
    ));
    assert_eq!(
        animation::resumed(true, false, true, false, lifecycle),
        Next::Idle
    );
}

#[test]
fn showing_the_tab_does_not_restart_a_paused_embed_and_resuming_does_not_restart_a_hidden_tab() {
    let unhalt = |before: (bool, bool), after: (bool, bool)| {
        animation::resumed(
            before.0 || before.1,
            after.0 || after.1,
            true,
            false,
            Lifecycle::Ready,
        )
    };
    let (paused, hidden) = (true, true);
    assert_eq!(unhalt((paused, hidden), (paused, false)), Next::Idle);
    assert_eq!(unhalt((paused, hidden), (false, hidden)), Next::Idle);
    assert_eq!(unhalt((false, hidden), (false, false)), Next::Frame);
    assert_eq!(unhalt((paused, false), (false, false)), Next::Frame);
    assert_eq!(unhalt((false, false), (false, false)), Next::Idle);
}
