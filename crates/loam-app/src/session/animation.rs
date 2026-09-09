use loam_runtime::host::HostError;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Next {
    Idle,
    Frame,
    Recover,
    Failed(String),
}

pub(crate) fn frame(
    paused: bool,
    lost: bool,
    animate: impl FnOnce() -> Result<(), HostError>,
) -> Next {
    if paused {
        return Next::Idle;
    }
    if lost {
        return Next::Recover;
    }
    match animate() {
        Ok(()) => Next::Frame,
        Err(error) => Next::Failed(format!("frame failed: {error:?}")),
    }
}

pub(crate) fn recovered(paused: bool, pending: bool, outcome: Result<(), String>) -> Next {
    match outcome {
        Err(message) => Next::Failed(message),
        Ok(()) if paused || pending => Next::Idle,
        Ok(()) => Next::Frame,
    }
}
