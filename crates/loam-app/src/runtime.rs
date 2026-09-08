use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Duration;

use crate::capture_types::CaptureRequest;
use crate::command::CommandQueue;
use crate::cursor::{CursorState, GrabMode};
use loam_egui::HistoryLine;

/// Clones address one runner; callbacks submit requests on its event-loop thread.
#[derive(Clone, Default)]
pub struct Runtime(pub(crate) Rc<RuntimeState>);

#[derive(Default)]
pub(crate) struct RuntimeState {
    pub commands: RefCell<CommandQueue>,
    pub output: RefCell<Vec<HistoryLine>>,
    pub cursor: Cell<CursorState>,
    pub pending_grab: Cell<Option<GrabMode>>,
    pub pending_visible: Cell<Option<bool>>,
    pub warp_center: Cell<bool>,
    pub period: Cell<Option<Duration>>,
    pub vsync: Cell<Option<bool>>,
    pub exit: Cell<bool>,
    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    pub captures: RefCell<Vec<CaptureRequest>>,
    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    pub capture_status: RefCell<Option<String>>,
    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    pub capture_panel: Cell<bool>,
}

impl Runtime {
    pub fn request_exit(&self) {
        self.0.exit.set(true);
    }

    pub(crate) fn exit_requested(&self) -> bool {
        self.0.exit.get()
    }

    pub fn capture(&self, request: CaptureRequest) {
        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
        self.0.captures.borrow_mut().push(request);
        #[cfg(any(not(feature = "capture"), target_arch = "wasm32"))]
        let _ = request;
    }

    pub fn capture_status(&self) -> Option<String> {
        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
        {
            self.0.capture_status.borrow().clone()
        }
        #[cfg(any(not(feature = "capture"), target_arch = "wasm32"))]
        {
            None
        }
    }

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    pub(crate) fn take_captures(&self) -> Vec<CaptureRequest> {
        std::mem::take(&mut *self.0.captures.borrow_mut())
    }

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    pub(crate) fn publish_capture_status(&self, status: Option<String>) {
        *self.0.capture_status.borrow_mut() = status;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_egui::Console;

    #[test]
    fn console_controls_do_not_cross_runners() {
        let first = Runtime::default();
        let second = Runtime::default();
        let mut console = Console::<()>::new();
        crate::fps::register_command(&mut console, &first);
        crate::vsync::register_command(&mut console, &first);
        for line in ["fps 30", "vsync off", "fps NaN", "vsync invalid"] {
            crate::command::run_on_console(&mut console, line, &mut ());
        }
        assert!((first.target_fps() - 30.0).abs() < 0.01);
        assert_eq!(first.take_vsync_request(), Some(false));
        assert_eq!(first.take_vsync_request(), None);
        assert_eq!(second.target_period(), None);
        assert_eq!(second.take_vsync_request(), None);
        first.request_exit();
        first.request_grab();
        assert!(!second.exit_requested());
        assert_eq!(second.take_cursor_request(), (None, None));
        assert_eq!(second.cursor_state(), CursorState::RELEASED);
    }
}
