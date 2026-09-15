#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CursorPolicy {
    Hold,
    #[default]
    Toggle,
}

#[derive(Default)]
pub(crate) struct CursorCapture {
    enabled: bool,
    policy: CursorPolicy,
    wanted: bool,
    locked: bool,
    alt: [bool; 2],
    focused: bool,
    request: Option<bool>,
}

impl CursorCapture {
    pub(crate) fn new() -> Self {
        Self {
            focused: true,
            ..Self::default()
        }
    }

    pub(crate) fn capture(&mut self, enabled: bool, policy: CursorPolicy) {
        let policy_changed = self.policy != policy;
        self.policy = policy;
        if self.enabled != enabled {
            self.enabled = enabled;
            self.set_wanted(enabled && (policy != CursorPolicy::Hold || !self.alt_held()));
        } else if enabled && policy_changed && policy == CursorPolicy::Hold {
            self.set_wanted(!self.alt_held());
        }
    }

    pub(crate) fn alt(&mut self, index: usize, pressed: bool) {
        let was_held = self.alt_held();
        self.alt[index] = pressed;
        let held = self.alt_held();
        if !self.enabled || was_held == held {
            return;
        }
        match self.policy {
            CursorPolicy::Hold => self.set_wanted(!held),
            CursorPolicy::Toggle if held => self.set_wanted(!self.wanted),
            CursorPolicy::Toggle => {}
        }
    }

    pub(crate) fn focus(&mut self, focused: bool) {
        if self.focused == focused {
            return;
        }
        self.focused = focused;
        if focused {
            if self.enabled && self.policy == CursorPolicy::Hold {
                self.wanted = true;
            }
            if self.wanted {
                self.request = Some(true);
            }
        } else {
            self.alt = [false; 2];
            self.request = Some(false);
        }
    }

    pub(crate) fn applied(&mut self, locked: bool) {
        self.locked = locked;
    }

    pub(crate) fn suspend(&mut self) {
        self.enabled = false;
        self.wanted = false;
        self.request = Some(false);
    }

    #[cfg(any(target_arch = "wasm32", test))]
    pub(crate) fn released(&mut self) {
        self.locked = false;
        self.wanted = false;
        self.request = None;
    }

    pub(crate) fn locked(&self) -> bool {
        self.locked
    }

    pub(crate) fn take_request(&mut self) -> Option<bool> {
        self.request.take()
    }

    fn alt_held(&self) -> bool {
        self.alt[0] || self.alt[1]
    }

    fn set_wanted(&mut self, wanted: bool) {
        if self.wanted == wanted {
            return;
        }
        self.wanted = wanted;
        if self.focused || !wanted {
            self.request = Some(wanted);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alt_edges_and_platform_release_preserve_explicit_cursor_control() {
        let mut cursor = CursorCapture::new();
        cursor.capture(true, CursorPolicy::Toggle);
        assert_eq!(cursor.take_request(), Some(true));
        assert!(!cursor.locked());

        cursor.applied(true);
        cursor.alt(0, true);
        assert_eq!(cursor.take_request(), Some(false));
        cursor.alt(1, true);
        cursor.alt(0, false);
        cursor.alt(1, false);
        assert_eq!(cursor.take_request(), None);

        cursor.capture(true, CursorPolicy::Hold);
        assert_eq!(cursor.take_request(), Some(true));
        cursor.alt(1, true);
        assert_eq!(cursor.take_request(), Some(false));
        cursor.alt(1, false);
        assert_eq!(cursor.take_request(), Some(true));

        cursor.capture(false, CursorPolicy::Toggle);
        cursor.capture(true, CursorPolicy::Toggle);
        assert_eq!(cursor.take_request(), Some(true));
        cursor.applied(true);
        cursor.released();
        cursor.capture(true, CursorPolicy::Toggle);
        assert_eq!(cursor.take_request(), None);
        cursor.alt(0, true);
        assert_eq!(cursor.take_request(), Some(true));
    }
}
