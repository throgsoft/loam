use web_time::Instant;

use loam_runtime::{
    ActionEvent, Bindings, Input, Key, Pointer, PointerButton, PointerPhase as RuntimePointerPhase,
};

use crate::wasm::input_queue::{InputMessage, PointerPhase};

/// Converts host events into the session's `Input` and recycles the buffers the session hands back.
pub struct InputMap {
    input: Input,
    spare: Input,
    held_keys: Vec<(Key, loam_runtime::ActionId)>,
    width: u32,
    height: u32,
    scale: f32,
    cursor: [f32; 2],
    look: [f32; 2],
    cursor_locked: bool,
    mouse_buttons: [bool; 3],
    active_touches: Vec<u32>,
    started: Instant,
}

#[cfg(any(target_arch = "wasm32", test))]
pub(crate) struct TouchCapture {
    active: Vec<CapturedTouch>,
    pointer: Option<u64>,
}

#[cfg(any(target_arch = "wasm32", test))]
struct CapturedTouch {
    id: u64,
    captured: bool,
    pos: [f32; 2],
}

#[cfg(any(target_arch = "wasm32", test))]
impl Default for TouchCapture {
    fn default() -> Self {
        Self {
            active: Vec::with_capacity(8),
            pointer: None,
        }
    }
}

#[cfg(any(target_arch = "wasm32", test))]
impl TouchCapture {
    pub(crate) fn route(
        &mut self,
        id: u64,
        phase: PointerPhase,
        over_ui: bool,
        pos: [f32; 2],
    ) -> bool {
        match phase {
            PointerPhase::Down => {
                if let Some(touch) = self.active.iter_mut().find(|touch| touch.id == id) {
                    touch.pos = pos;
                    return touch.captured;
                }
                if over_ui && self.pointer.is_none() {
                    self.pointer = Some(id);
                }
                self.active.push(CapturedTouch {
                    id,
                    captured: over_ui,
                    pos,
                });
                over_ui
            }
            PointerPhase::Move => {
                let Some(touch) = self.active.iter_mut().find(|touch| touch.id == id) else {
                    return false;
                };
                touch.pos = pos;
                touch.captured
            }
            PointerPhase::Up | PointerPhase::Cancel => {
                let captured = self
                    .active
                    .iter()
                    .position(|touch| touch.id == id)
                    .map(|index| self.active.remove(index).captured)
                    .unwrap_or(false);
                if self.pointer == Some(id) {
                    self.pointer = None;
                }
                captured
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn cancel_all(&mut self) -> impl Iterator<Item = (u64, [f32; 2])> + '_ {
        self.pointer = None;
        self.active
            .drain(..)
            .filter_map(|touch| touch.captured.then_some((touch.id, touch.pos)))
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn is_pointer(&self, id: u64) -> bool {
        self.pointer == Some(id)
    }
}

impl Default for InputMap {
    fn default() -> Self {
        Self {
            input: Input::default(),
            spare: Input::default(),
            held_keys: Vec::new(),
            width: 1,
            height: 1,
            scale: 1.0,
            cursor: [0.0; 2],
            look: [0.0; 2],
            cursor_locked: false,
            mouse_buttons: [false; 3],
            active_touches: Vec::with_capacity(8),
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

    pub fn cursor_locked(&self) -> bool {
        self.cursor_locked
    }

    pub fn set_cursor_locked(&mut self, locked: bool) {
        if self.cursor_locked == locked {
            return;
        }
        self.cursor_locked = locked;
        self.look = [0.0; 2];
        self.release_pointers();
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
        let held = self.held_keys.iter().position(|(held, _)| *held == key);
        if pressed {
            if held.is_some() {
                return;
            }
            let Some(action) = bindings.action(key) else {
                return;
            };
            let starts_action = !self
                .held_keys
                .iter()
                .any(|(_, held_action)| *held_action == action);
            self.held_keys.push((key, action));
            if starts_action {
                self.input.actions.push(ActionEvent {
                    action,
                    pressed: true,
                });
            }
        } else {
            let Some(held) = held else {
                return;
            };
            let (_, action) = self.held_keys.remove(held);
            if !self
                .held_keys
                .iter()
                .any(|(_, held_action)| *held_action == action)
            {
                self.input.actions.push(ActionEvent {
                    action,
                    pressed: false,
                });
            }
        }
    }

    pub(crate) fn host_action(
        &mut self,
        bindings: &Bindings,
        key: Key,
        pressed: bool,
        consumed: bool,
    ) {
        if !consumed || !pressed {
            self.action(bindings, key, pressed);
        }
    }

    pub fn release_all(&mut self) {
        while let Some((_, action)) = self.held_keys.pop() {
            if !self
                .held_keys
                .iter()
                .any(|(_, held_action)| *held_action == action)
            {
                self.input.actions.push(ActionEvent {
                    action,
                    pressed: false,
                });
            }
        }
        self.release_pointers();
    }

    fn release_pointers(&mut self) {
        let time = self.started.elapsed().as_secs_f64();
        for button in [
            PointerButton::Primary,
            PointerButton::Secondary,
            PointerButton::Middle,
        ] {
            let index = Self::button_index(button);
            if self.mouse_buttons[index] {
                self.push(
                    self.cursor,
                    [0.0; 2],
                    Some(button),
                    RuntimePointerPhase::Cancelled,
                    time,
                );
            }
        }
        self.mouse_buttons = [false; 3];
        for id in std::mem::take(&mut self.active_touches) {
            self.push_with(
                id,
                self.cursor,
                [0.0; 2],
                Some(PointerButton::Primary),
                RuntimePointerPhase::Cancelled,
                time,
            );
        }
    }

    pub fn raw_motion(&mut self, physical_dx: f32, physical_dy: f32) {
        if !self.cursor_locked {
            return;
        }
        self.look[0] += physical_dx / self.scale;
        self.look[1] -= physical_dy / self.scale;
    }

    pub fn moved(&mut self, ndc: [f32; 2]) {
        self.moved_at(ndc, self.started.elapsed().as_secs_f64());
    }

    fn moved_at(&mut self, ndc: [f32; 2], time: f64) {
        let delta = [ndc[0] - self.cursor[0], ndc[1] - self.cursor[1]];
        self.cursor = ndc;
        let mut moved = false;
        for button in [
            PointerButton::Primary,
            PointerButton::Secondary,
            PointerButton::Middle,
        ] {
            if self.mouse_buttons[Self::button_index(button)] {
                self.push(ndc, delta, Some(button), RuntimePointerPhase::Moved, time);
                moved = true;
            }
        }
        if !moved {
            self.push(ndc, delta, None, RuntimePointerPhase::Moved, time);
        }
    }

    pub fn sync_mouse_buttons(&mut self, buttons: u8) {
        self.mouse_buttons = [buttons & 1 != 0, buttons & 2 != 0, buttons & 4 != 0];
    }

    pub fn button(&mut self, ndc: [f32; 2], button: PointerButton, pressed: bool) {
        self.button_at(ndc, button, pressed, self.started.elapsed().as_secs_f64());
    }

    fn button_at(&mut self, ndc: [f32; 2], button: PointerButton, pressed: bool, time: f64) {
        self.cursor = ndc;
        self.mouse_buttons[Self::button_index(button)] = pressed;
        let phase = if pressed {
            RuntimePointerPhase::Began
        } else {
            RuntimePointerPhase::Ended
        };
        self.push(ndc, [0.0; 2], Some(button), phase, time);
    }

    #[cfg(any(not(target_arch = "wasm32"), test))]
    pub(crate) fn host_button(
        &mut self,
        ndc: [f32; 2],
        button: PointerButton,
        pressed: bool,
        consumed: bool,
    ) {
        let held = self.mouse_buttons[Self::button_index(button)];
        if !consumed || (!pressed && held) {
            self.button(ndc, button, pressed);
        }
    }

    pub fn touch(&mut self, id: u32, ndc: [f32; 2], phase: RuntimePointerPhase) {
        self.touch_at(id, ndc, phase, self.started.elapsed().as_secs_f64());
    }

    fn touch_at(&mut self, id: u32, ndc: [f32; 2], phase: RuntimePointerPhase, time: f64) {
        let delta = match phase {
            RuntimePointerPhase::Moved => [ndc[0] - self.cursor[0], ndc[1] - self.cursor[1]],
            _ => [0.0; 2],
        };
        self.cursor = ndc;
        self.push_with(id, ndc, delta, Some(PointerButton::Primary), phase, time);
    }

    fn host_touch_at(
        &mut self,
        id: u32,
        ndc: [f32; 2],
        phase: RuntimePointerPhase,
        consumed: bool,
        time: f64,
    ) {
        let held = self.active_touches.contains(&id);
        match phase {
            RuntimePointerPhase::Began if !consumed => {
                if !held {
                    self.active_touches.push(id);
                }
                self.touch_at(id, ndc, phase, time);
            }
            RuntimePointerPhase::Moved if !consumed => self.touch_at(id, ndc, phase, time),
            RuntimePointerPhase::Ended | RuntimePointerPhase::Cancelled if !consumed || held => {
                self.active_touches.retain(|active| *active != id);
                self.touch_at(id, ndc, phase, time);
            }
            _ => {}
        }
    }

    pub fn wheel(&mut self, delta: [f32; 2]) {
        self.input.scroll[0] += delta[0];
        self.input.scroll[1] += delta[1];
    }

    fn button_index(button: PointerButton) -> usize {
        match button {
            PointerButton::Primary => 0,
            PointerButton::Secondary => 1,
            PointerButton::Middle => 2,
        }
    }

    fn push(
        &mut self,
        ndc: [f32; 2],
        delta: [f32; 2],
        button: Option<PointerButton>,
        phase: RuntimePointerPhase,
        time: f64,
    ) {
        self.push_with(0, ndc, delta, button, phase, time);
    }

    fn push_with(
        &mut self,
        id: u32,
        ndc: [f32; 2],
        delta: [f32; 2],
        button: Option<PointerButton>,
        phase: RuntimePointerPhase,
        time: f64,
    ) {
        self.input.pointers.push(Pointer {
            id,
            button,
            ndc,
            delta: [
                delta[0] * self.width as f32 / (2.0 * self.scale),
                delta[1] * self.height as f32 / (2.0 * self.scale),
            ],
            phase,
            time,
        });
    }

    pub fn take(&mut self) -> Input {
        self.input.held.clear();
        for (_, action) in &self.held_keys {
            if !self.input.held.contains(action) {
                self.input.held.push(*action);
            }
        }
        self.input.look = std::mem::take(&mut self.look);
        self.input.cursor_locked = self.cursor_locked;
        std::mem::replace(&mut self.input, std::mem::take(&mut self.spare))
    }

    pub fn reclaim(&mut self, mut reclaimed: Input) {
        reclaimed.pointers.clear();
        reclaimed.actions.clear();
        reclaimed.held.clear();
        reclaimed.scroll = [0.0; 2];
        reclaimed.look = [0.0; 2];
        reclaimed.cursor_locked = false;
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

#[cfg(not(target_arch = "wasm32"))]
pub fn winit_alt(event: &winit::event::KeyEvent) -> Option<usize> {
    use winit::keyboard::{KeyCode, PhysicalKey};

    match event.physical_key {
        PhysicalKey::Code(KeyCode::AltLeft) => Some(0),
        PhysicalKey::Code(KeyCode::AltRight) => Some(1),
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

fn dom_phase(phase: PointerPhase) -> RuntimePointerPhase {
    match phase {
        PointerPhase::Down => RuntimePointerPhase::Began,
        PointerPhase::Move => RuntimePointerPhase::Moved,
        PointerPhase::Up => RuntimePointerPhase::Ended,
        PointerPhase::Cancel => RuntimePointerPhase::Cancelled,
    }
}

pub fn apply(map: &mut InputMap, bindings: &Bindings, message: &InputMessage, consumed: bool) {
    match message {
        InputMessage::Resize { width, height, dpr } => map.resize(*width, *height, *dpr),
        InputMessage::MouseMove {
            x,
            y,
            buttons,
            dx,
            dy,
            time,
        } => {
            if consumed {
                return;
            }
            if map.cursor_locked() {
                map.raw_motion(*dx * map.scale(), *dy * map.scale());
            } else {
                let ndc = map.css_ndc(*x, *y);
                map.sync_mouse_buttons(*buttons);
                map.moved_at(ndc, time.as_secs_f64());
            }
        }
        InputMessage::MouseButton {
            x,
            y,
            button,
            pressed,
            time,
        } => {
            let button = match button {
                0 => Some(PointerButton::Primary),
                1 => Some(PointerButton::Middle),
                2 => Some(PointerButton::Secondary),
                _ => None,
            };
            if let Some(button) = button {
                let ndc = map.css_ndc(*x, *y);
                let held = map.mouse_buttons[InputMap::button_index(button)];
                if !consumed || (!pressed && held) {
                    map.button_at(ndc, button, *pressed, time.as_secs_f64());
                }
            }
        }
        InputMessage::Key { code, pressed, .. } => {
            if let Some(key) = dom_key(code) {
                map.host_action(bindings, key, *pressed, consumed);
            }
        }
        InputMessage::Focus(false) => map.release_all(),
        InputMessage::Pointer {
            id,
            x,
            y,
            phase,
            time,
        } => {
            let ndc = map.css_ndc(*x, *y);
            map.host_touch_at(
                *id as u32,
                ndc,
                dom_phase(*phase),
                consumed,
                time.as_secs_f64(),
            );
        }
        InputMessage::MouseWheel { dx, dy } if !consumed => map.wheel([-*dx, -*dy]),
        InputMessage::MouseWheel { .. } => {}
        InputMessage::Focus(true)
        | InputMessage::Visibility(_)
        | InputMessage::Start
        | InputMessage::PointerLockChanged { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use loam_runtime::{ActionId, Ctx, Phase, Session, SimConfig};

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
        session.system(Phase::Dispatch, "watch", |ctx: Ctx<'_, Watched>| {
            if ctx.input.pressed(WALK) {
                ctx.app.walked.set(true);
            }
            Ok(())
        });

        let mut map = InputMap::default();
        apply(
            &mut map,
            &bindings,
            &InputMessage::Resize {
                width: 800,
                height: 600,
                dpr: 2.0,
            },
            false,
        );
        apply(&mut map, &bindings, &key_message("KeyW", true), false);
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
            map.button([0.1, 0.2], PointerButton::Primary, true);
            map.moved([0.2, 0.2]);
            map.button([0.2, 0.2], PointerButton::Primary, false);
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

    #[test]
    fn camera_turns_do_not_depend_on_viewport_shape_or_pixel_scale() {
        for (width, height, scale) in [(640, 480, 1.0), (1280, 480, 1.0), (1280, 960, 2.0)] {
            let mut map = InputMap::default();
            map.resize(width, height, scale);
            map.button(map.css_ndc(50.0, 50.0), PointerButton::Secondary, true);
            map.moved(map.css_ndc(70.0, 70.0));
            let input = map.take();
            let mut orbit = loam_runtime::Orbit::around([0.0; 3], 5.0);
            orbit.drag(input.drag(PointerButton::Secondary));
            assert!((orbit.yaw + 0.12).abs() < 1e-6);
            assert!((orbit.pitch + 0.12).abs() < 1e-6);

            map.set_cursor_locked(true);
            map.raw_motion(20.0 * scale, 20.0 * scale);
            let mut camera = loam_runtime::FreeCamera::default();
            camera.look(map.take().look);
            let [x, y, z] = camera.eye.forward;
            assert!(((-x).atan2(-z) + 0.04).abs() < 1e-6);
            assert!((y.asin() + 0.04).abs() < 1e-6);
        }
    }

    #[test]
    fn raw_look_requires_a_confirmed_lock_and_survives_a_leased_input() {
        let mut map = InputMap::default();
        map.resize(800, 400, 2.0);
        map.raw_motion(20.0, 10.0);
        map.set_cursor_locked(true);
        map.raw_motion(20.0, 10.0);

        let first = map.take();
        assert_eq!(first.look, [10.0, -5.0]);
        assert!(first.cursor_locked);

        map.raw_motion(40.0, -20.0);
        map.reclaim(first);
        let second = map.take();
        assert_eq!(second.look, [20.0, 10.0]);
    }

    #[test]
    fn a_consumed_release_clears_the_action_held_before_the_overlay_took_focus() {
        let bindings = Bindings::new().key(Key::Letter('w'), WALK);
        let mut map = InputMap::default();
        map.host_action(&bindings, Key::Letter('w'), true, false);
        let first = map.take();

        map.host_action(&bindings, Key::Letter('w'), false, true);
        map.reclaim(first);
        let second = map.take();

        assert!(second
            .actions
            .iter()
            .any(|event| event.action == WALK && !event.pressed));
        assert!(!second.held.contains(&WALK));
    }

    #[test]
    fn consumed_pointer_releases_clear_scene_holds() {
        let mut map = InputMap::default();
        map.host_button([0.0; 2], PointerButton::Primary, true, false);
        let first = map.take();
        map.reclaim(first);

        map.host_button([0.0; 2], PointerButton::Primary, false, true);
        let released = map.take();

        assert_eq!(released.pointers.len(), 1);
        assert_eq!(released.pointers[0].phase, RuntimePointerPhase::Ended);
    }

    #[test]
    fn browser_touch_capture_keeps_its_first_owner_and_source_times() {
        let bindings = Bindings::new();
        let mut map = InputMap::default();
        map.resize(800, 600, 1.0);
        let mut capture = TouchCapture::default();

        for (message, over_ui) in [
            (touch_message(7, PointerPhase::Down, 100, 40.0), true),
            (touch_message(7, PointerPhase::Move, 120, 50.0), false),
            (touch_message(7, PointerPhase::Up, 140, 60.0), false),
        ] {
            let InputMessage::Pointer {
                id, x, y, phase, ..
            } = &message
            else {
                unreachable!();
            };
            let consumed = capture.route(*id, *phase, over_ui, [*x, *y]);
            apply(&mut map, &bindings, &message, consumed);
        }
        assert!(map.take().pointers.is_empty());

        for (message, over_ui) in [
            (touch_message(8, PointerPhase::Down, 200, 100.0), false),
            (touch_message(8, PointerPhase::Move, 225, 110.0), true),
            (touch_message(8, PointerPhase::Up, 250, 120.0), true),
        ] {
            let InputMessage::Pointer {
                id, x, y, phase, ..
            } = &message
            else {
                unreachable!();
            };
            let consumed = capture.route(*id, *phase, over_ui, [*x, *y]);
            apply(&mut map, &bindings, &message, consumed);
        }
        let pointers = map.take().pointers;
        assert_eq!(pointers.len(), 3);
        assert_eq!(pointers[0].phase, RuntimePointerPhase::Began);
        assert_eq!(pointers[1].phase, RuntimePointerPhase::Moved);
        assert_eq!(pointers[2].phase, RuntimePointerPhase::Ended);
        assert_eq!(
            pointers
                .iter()
                .map(|pointer| pointer.time)
                .collect::<Vec<_>>(),
            [0.2, 0.225, 0.25]
        );
    }

    fn touch_message(id: u64, phase: PointerPhase, millis: u64, x: f32) -> InputMessage {
        InputMessage::Pointer {
            id,
            x,
            y: 20.0,
            phase,
            time: std::time::Duration::from_millis(millis),
        }
    }
}
