/// Mirrors `winit::window::CursorGrabMode` so winit stays out of demo code.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum GrabMode {
    #[default]
    None,
    Confined,
    /// Motion arrives as raw device delta (`FrameInput::mouse_raw_delta`).
    Locked,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CursorState {
    pub grab: GrabMode,
    pub visible: bool,
}

impl CursorState {
    pub const RELEASED: Self = Self {
        grab: GrabMode::None,
        visible: true,
    };
}

impl Default for CursorState {
    fn default() -> Self {
        Self::RELEASED
    }
}

impl crate::Runtime {
    pub fn request_grab_mode(&self, mode: GrabMode) {
        self.0.pending_grab.set(Some(mode));
    }
    pub fn request_cursor_visible(&self, visible: bool) {
        self.0.pending_visible.set(Some(visible));
    }
    pub fn request_warp_to_center(&self) {
        self.0.warp_center.set(true);
    }
    pub fn request_grab(&self) {
        self.request_grab_mode(GrabMode::Locked);
        self.request_cursor_visible(false);
    }
    pub fn request_release(&self) {
        self.request_grab_mode(GrabMode::None);
        self.request_cursor_visible(true);
    }
    pub(crate) fn take_cursor_request(&self) -> (Option<GrabMode>, Option<bool>) {
        (self.0.pending_grab.take(), self.0.pending_visible.take())
    }
    pub(crate) fn take_warp_center(&self) -> bool {
        self.0.warp_center.replace(false)
    }
    pub(crate) fn mark_cursor_applied(&self, grab: GrabMode, visible: bool) {
        self.0.cursor.set(CursorState { grab, visible });
    }
    pub fn cursor_state(&self) -> CursorState {
        self.0.cursor.get()
    }
}

pub(crate) fn apply_grab<E>(
    requested: GrabMode,
    mut apply: impl FnMut(GrabMode) -> Result<(), E>,
) -> Result<GrabMode, E> {
    match apply(requested) {
        Ok(()) => Ok(requested),
        Err(error) => {
            let fallback = match requested {
                GrabMode::Locked => GrabMode::Confined,
                GrabMode::Confined => GrabMode::Locked,
                GrabMode::None => return Err(error),
            };
            apply(fallback).map(|()| fallback)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn platform_result_determines_cursor_state() {
        let fallback = apply_grab(GrabMode::Locked, |mode| {
            if mode == GrabMode::Confined {
                Ok(())
            } else {
                Err(())
            }
        });
        assert_eq!(fallback, Ok(GrabMode::Confined));
        assert!(apply_grab(GrabMode::Locked, |_| Err(())).is_err());
        let mut attempts = 0;
        assert!(apply_grab(GrabMode::None, |_| {
            attempts += 1;
            Err(())
        })
        .is_err());
        assert_eq!(attempts, 1);
    }
}
