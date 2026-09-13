use loam_runtime::host::HostError;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Next {
    Idle,
    Frame,
    Recover,
    Failed(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lifecycle {
    Ready,
    Recovering,
    Failed,
}

pub(crate) fn frame(
    paused: bool,
    lifecycle: &mut Lifecycle,
    lost: bool,
    animate: impl FnOnce() -> Result<(), HostError>,
) -> Next {
    if paused || *lifecycle != Lifecycle::Ready {
        return Next::Idle;
    }
    if lost {
        *lifecycle = Lifecycle::Recovering;
        return Next::Recover;
    }
    match animate() {
        Ok(()) => Next::Frame,
        Err(error) => {
            *lifecycle = Lifecycle::Failed;
            Next::Failed(format!("frame failed: {error:?}"))
        }
    }
}

pub(crate) fn resumed(
    was_paused: bool,
    started: bool,
    pending: bool,
    lifecycle: Lifecycle,
) -> Next {
    if was_paused && started && !pending && lifecycle == Lifecycle::Ready {
        Next::Frame
    } else {
        Next::Idle
    }
}

pub(crate) fn recovered(
    lifecycle: &mut Lifecycle,
    paused: bool,
    pending: bool,
    outcome: Result<(), String>,
) -> Next {
    match outcome {
        Err(message) => {
            *lifecycle = Lifecycle::Failed;
            Next::Failed(message)
        }
        Ok(()) => {
            *lifecycle = Lifecycle::Ready;
            if paused || pending {
                Next::Idle
            } else {
                Next::Frame
            }
        }
    }
}
