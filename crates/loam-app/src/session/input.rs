use web_time::Instant;

use loam_runtime::{ActionEvent, Bindings, Input, Key, Pointer, PointerPhase};

use crate::wasm::input_queue::InputMessage;

pub struct InputMap {
    input: Input,
    spare: Input,
    width: u32,
    height: u32,
    scale: f32,
    cursor: [f32; 2],
    dragging: bool,
    started: Instant,
}

impl Default for InputMap {
    fn default() -> Self {
        Self {
            input: Input::default(),
            spare: Input::default(),
            width: 1,
            height: 1,
            scale: 1.0,
            cursor: [0.0; 2],
            dragging: false,
            started: Instant::now(),
        }
    }
}

impl InputMap {
    pub fn resize(&mut self, width: u32, height: u32, scale: f32) {
        self.width = width.max(1);
        self.height = height.max(1);
        if scale.is_finite() && scale > 0.0 {
            self.scale = scale;
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn cursor(&self) -> [f32; 2] {
        self.cursor
    }

    pub fn ndc(&self, physical_x: f64, physical_y: f64) -> [f32; 2] {
        [
            (physical_x as f32 / self.width as f32) * 2.0 - 1.0,
            1.0 - (physical_y as f32 / self.height as f32) * 2.0,
        ]
    }

    pub fn css_ndc(&self, css_x: f32, css_y: f32) -> [f32; 2] {
        self.ndc(f64::from(css_x * self.scale), f64::from(css_y * self.scale))
    }

    pub fn action(&mut self, bindings: &Bindings, key: Key, pressed: bool) {
        let Some(action) = bindings.action(key) else {
            return;
        };
        if pressed == self.input.held.contains(&action) {
            return;
        }
        if pressed {
            self.input.held.push(action);
        } else {
            self.input.held.retain(|held| *held != action);
        }
        self.input.actions.push(ActionEvent { action, pressed });
    }

    pub fn release_all(&mut self) {
        for action in std::mem::take(&mut self.input.held) {
            self.input.actions.push(ActionEvent {
                action,
                pressed: false,
            });
        }
        self.dragging = false;
    }

    pub fn moved(&mut self, ndc: [f32; 2]) {
        let delta = [ndc[0] - self.cursor[0], ndc[1] - self.cursor[1]];
        self.cursor = ndc;
        if self.dragging {
            self.push(ndc, delta, PointerPhase::Moved);
        }
    }

    pub fn button(&mut self, ndc: [f32; 2], pressed: bool) {
        self.cursor = ndc;
        self.dragging = pressed;
        let phase = if pressed {
            PointerPhase::Began
        } else {
            PointerPhase::Ended
        };
        self.push(ndc, [0.0; 2], phase);
    }

    pub fn touch(&mut self, id: u32, ndc: [f32; 2], phase: PointerPhase) {
        let delta = match phase {
            PointerPhase::Moved => [ndc[0] - self.cursor[0], ndc[1] - self.cursor[1]],
            _ => [0.0; 2],
        };
        self.cursor = ndc;
        self.push_with(id, ndc, delta, phase);
    }

    fn push(&mut self, ndc: [f32; 2], delta: [f32; 2], phase: PointerPhase) {
        self.push_with(0, ndc, delta, phase);
    }

    fn push_with(&mut self, id: u32, ndc: [f32; 2], delta: [f32; 2], phase: PointerPhase) {
        self.input.pointers.push(Pointer {
            id,
            ndc,
            delta,
            phase,
            time: self.started.elapsed().as_secs_f64(),
        });
    }

    pub fn take(&mut self) -> Input {
        std::mem::replace(&mut self.input, std::mem::take(&mut self.spare))
    }

    pub fn reclaim(&mut self, mut reclaimed: Input) {
        reclaimed.pointers.clear();
        reclaimed.actions.clear();
        self.input.held.clear();
        self.input.held.extend_from_slice(&reclaimed.held);
        reclaimed.held.clear();
        self.spare = reclaimed;
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn winit_key(event: &winit::event::KeyEvent) -> Option<Key> {
    use winit::keyboard::{Key as Logical, NamedKey};

    match &event.logical_key {
        Logical::Named(NamedKey::Space) => Some(Key::Space),
        Logical::Named(NamedKey::Escape) => Some(Key::Escape),
        Logical::Character(text) => match text.chars().next() {
            Some(letter) if letter.is_ascii_digit() => Some(Key::Digit(letter as u8 - b'0')),
            Some(letter) => Some(Key::Letter(letter.to_ascii_lowercase())),
            None => None,
        },
        _ => None,
    }
}

pub fn dom_key(code: &str) -> Option<Key> {
    match code {
        "Space" => Some(Key::Space),
        "Escape" => Some(Key::Escape),
        _ => {
            let digit = code
                .strip_prefix("Digit")
                .or_else(|| code.strip_prefix("Numpad"));
            if let Some(digit) = digit.and_then(|rest| rest.parse::<u8>().ok()) {
                return Some(Key::Digit(digit));
            }
            let mut letters = code.strip_prefix("Key")?.chars();
            let letter = letters.next().filter(char::is_ascii_alphabetic)?;
            letters
                .next()
                .is_none()
                .then(|| Key::Letter(letter.to_ascii_lowercase()))
        }
    }
}

fn dom_phase(phase: loam_input::PointerPhase) -> PointerPhase {
    match phase {
        loam_input::PointerPhase::Down => PointerPhase::Began,
        loam_input::PointerPhase::Move => PointerPhase::Moved,
        loam_input::PointerPhase::Up => PointerPhase::Ended,
        loam_input::PointerPhase::Cancel => PointerPhase::Cancelled,
    }
}

pub fn apply(map: &mut InputMap, bindings: &Bindings, message: &InputMessage) {
    match message {
        InputMessage::Resize { width, height, dpr } => map.resize(*width, *height, *dpr),
        InputMessage::MouseMove { x, y, .. } => {
            let ndc = map.css_ndc(*x, *y);
            map.moved(ndc);
        }
        InputMessage::MouseButton {
            x,
            y,
            button,
            pressed,
        } => {
            if *button == 0 {
                let ndc = map.css_ndc(*x, *y);
                map.button(ndc, *pressed);
            }
        }
        InputMessage::Key { code, pressed, .. } => {
            if let Some(key) = dom_key(code) {
                map.action(bindings, key, *pressed);
            }
        }
        InputMessage::Focus(false) => map.release_all(),
        InputMessage::Pointer {
            id, x, y, phase, ..
        } => {
            let ndc = map.css_ndc(*x, *y);
            map.touch(*id as u32, ndc, dom_phase(*phase));
        }
        InputMessage::Focus(true)
        | InputMessage::MouseWheel { .. }
        | InputMessage::Visibility(_)
        | InputMessage::Start
        | InputMessage::PointerLockChanged(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use loam_runtime::{Access, ActionId, Ctx, Phase, Session, SimConfig};

    use super::*;

    mod alloc_probe {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        thread_local! {
            static BYTES: Cell<usize> = const { Cell::new(0) };
        }

        pub struct Counting;

        // SAFETY: Methods preserve System contracts; const TLS and wrapping Cell updates cannot unwind.
        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                let _ = BYTES.try_with(|bytes| bytes.set(bytes.get().wrapping_add(layout.size())));
                // SAFETY: The caller supplies a valid nonzero allocation layout.
                unsafe { System.alloc(layout) }
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                // SAFETY: The caller supplies a live System allocation and its original layout.
                unsafe { System.dealloc(ptr, layout) }
            }

            unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                let _ = BYTES.try_with(|bytes| bytes.set(bytes.get().wrapping_add(new_size)));
                // SAFETY: The caller supplies a live System allocation, its layout, and a valid new size.
                unsafe { System.realloc(ptr, layout, new_size) }
            }
        }

        pub fn bytes_allocated_by(body: impl FnOnce()) -> usize {
            let before = BYTES.with(Cell::get);
            body();
            BYTES.with(Cell::get).wrapping_sub(before)
        }
    }

    #[global_allocator]
    static COUNTING_ALLOCATOR: alloc_probe::Counting = alloc_probe::Counting;

    const WALK: ActionId = ActionId(0);

    loam_runtime::stores! {
        #[derive(Default)]
        pub struct Watched {
            walked: Value<bool>,
        }
    }

    fn key_message(code: &str, pressed: bool) -> InputMessage {
        InputMessage::Key {
            code: code.to_string(),
            key: code.to_lowercase(),
            pressed,
            repeat: false,
            ctrl: false,
            shift: false,
            alt: false,
            meta: false,
        }
    }

    #[test]
    fn a_browser_key_message_reaches_the_session_as_its_bound_action() {
        let bindings = Bindings::new().key(Key::Letter('w'), WALK);
        let mut session = Session::new(Watched::default(), SimConfig::default());
        session.system(
            Phase::Dispatch,
            "watch",
            Access::new(),
            |ctx: Ctx<'_, Watched>| {
                if ctx.input.pressed(WALK) {
                    ctx.app.walked.set(true);
                }
            },
        );

        let mut map = InputMap::default();
        apply(
            &mut map,
            &bindings,
            &InputMessage::Resize {
                width: 800,
                height: 600,
                dpr: 2.0,
            },
        );
        apply(&mut map, &bindings, &key_message("KeyW", true));
        session.boundary(map.take()).expect("boundary");

        assert!(
            *session.app.walked.get(),
            "the DOM key code never became the action bound to it"
        );
    }

    #[test]
    fn a_warmed_frame_of_host_input_conversion_asks_the_allocator_for_nothing() {
        let bindings = Bindings::new().key(Key::Letter('w'), WALK);
        let mut session = Session::new(Watched::default(), SimConfig::default());
        let mut map = InputMap::default();
        map.resize(800, 600, 1.0);
        let cycle = |map: &mut InputMap, session: &mut Session<Watched>| {
            map.action(&bindings, Key::Letter('w'), true);
            map.button([0.1, 0.2], true);
            map.moved([0.2, 0.2]);
            map.button([0.2, 0.2], false);
            map.action(&bindings, Key::Letter('w'), false);
            let input = map.take();
            session.boundary(input).expect("boundary");
            session.tick().expect("tick");
            map.reclaim(session.take_input());
        };
        for _ in 0..8 {
            cycle(&mut map, &mut session);
        }

        let bytes = alloc_probe::bytes_allocated_by(|| {
            for _ in 0..16 {
                cycle(&mut map, &mut session);
            }
        });
        assert_eq!(
            bytes, 0,
            "16 warmed frames of input asked the allocator for {bytes} bytes"
        );
    }
}
