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

/// One value for mouse and touch; `ndc` is y-up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pointer {
    pub id: u32,
    pub ndc: [f32; 2],
    pub delta: [f32; 2],
    pub phase: PointerPhase,
    pub time: f64,
}

/// Everything the host converted since the previous boundary.
#[derive(Clone, Debug, Default)]
pub struct Input {
    pub pointers: Vec<Pointer>,
    pub actions: Vec<ActionEvent>,
    pub held: Vec<ActionId>,
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

    pub fn drag(&self) -> [f32; 2] {
        self.pointers
            .iter()
            .filter(|pointer| pointer.phase == PointerPhase::Moved)
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

/// Host configuration over pointer, key, and gamepad; nothing in the session names a key.
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
