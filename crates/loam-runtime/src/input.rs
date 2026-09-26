use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActionId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActionEvent {
    pub action: ActionId,
    pub pressed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerPhase {
    Began,
    Moved,
    Ended,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerButton {
    Primary,
    Secondary,
    Middle,
}

/// Positions use NDC; deltas use logical pixels; both are y-up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pointer {
    pub id: u32,
    pub button: Option<PointerButton>,
    pub ndc: [f32; 2],
    pub delta: [f32; 2],
    pub phase: PointerPhase,
    pub time: f64,
}

/// Messages the embedding page sent since the previous boundary, in arrival order.
#[derive(Clone, Debug, Default)]
pub struct HostMessages {
    text: String,
    entries: Vec<(Range<usize>, Range<usize>)>,
    values: Vec<f32>,
}

impl HostMessages {
    pub fn push(&mut self, topic: &str, values: &[f32]) {
        let text = self.text.len()..self.text.len() + topic.len();
        let range = self.values.len()..self.values.len() + values.len();
        self.text.push_str(topic);
        self.values.extend_from_slice(values);
        self.entries.push((text, range));
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.entries.clear();
        self.values.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &[f32])> {
        self.entries
            .iter()
            .map(|(text, values)| (&self.text[text.clone()], &self.values[values.clone()]))
    }

    pub fn latest(&self, topic: &str) -> Option<&[f32]> {
        self.entries
            .iter()
            .rev()
            .find(|(text, _)| &self.text[text.clone()] == topic)
            .map(|(_, values)| &self.values[values.clone()])
    }
}

/// Everything the host converted since the previous boundary.
#[derive(Clone, Debug, Default)]
pub struct Input {
    pub pointers: Vec<Pointer>,
    pub actions: Vec<ActionEvent>,
    pub held: Vec<ActionId>,
    pub scroll: [f32; 2],
    /// Captured mouse motion in logical pixels, y-up.
    pub look: [f32; 2],
    pub cursor_locked: bool,
    /// Seconds on the pointer clock when the host gathered this boundary.
    pub time: f64,
    pub host: HostMessages,
}

impl Input {
    pub fn pressed(&self, action: ActionId) -> bool {
        self.actions
            .iter()
            .any(|event| event.action == action && event.pressed)
    }

    pub fn is_held(&self, action: ActionId) -> bool {
        self.held.contains(&action)
    }

    pub fn began(&self) -> Option<&Pointer> {
        self.pointers
            .iter()
            .find(|pointer| pointer.phase == PointerPhase::Began)
    }

    pub fn drag(&self, button: PointerButton) -> [f32; 2] {
        self.pointers
            .iter()
            .filter(|pointer| {
                pointer.phase == PointerPhase::Moved && pointer.button == Some(button)
            })
            .fold([0.0; 2], |sum, pointer| {
                [sum[0] + pointer.delta[0], sum[1] + pointer.delta[1]]
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    Space,
    Escape,
    Letter(char),
    Digit(u8),
}

#[derive(Clone, Debug, Default)]
pub struct Bindings {
    keys: Vec<(Key, ActionId)>,
}

impl Bindings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn key(mut self, key: Key, action: ActionId) -> Self {
        self.keys.push((key, action));
        self
    }

    pub fn action(&self, key: Key) -> Option<ActionId> {
        self.keys
            .iter()
            .find(|(bound, _)| *bound == key)
            .map(|(_, action)| *action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_messages_keep_arrival_order_and_latest_returns_the_newest_per_topic() {
        let mut host = HostMessages::default();
        host.push("scroll", &[0.25]);
        host.push("reset", &[]);
        host.push("scroll", &[0.5, 1.0]);

        let seen: Vec<(&str, &[f32])> = host.iter().collect();
        assert_eq!(
            seen,
            [
                ("scroll", &[0.25][..]),
                ("reset", &[][..]),
                ("scroll", &[0.5, 1.0][..])
            ]
        );
        assert_eq!(host.latest("scroll"), Some(&[0.5, 1.0][..]));
        assert_eq!(host.latest("reset"), Some(&[][..]));
        assert_eq!(host.latest("scrol"), None);
    }
}
