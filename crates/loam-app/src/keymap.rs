use loam_egui::egui;

pub fn mouse_button_egui(button: u8) -> Option<egui::PointerButton> {
    match button {
        0 => Some(egui::PointerButton::Primary),
        1 => Some(egui::PointerButton::Middle),
        2 => Some(egui::PointerButton::Secondary),
        3 => Some(egui::PointerButton::Extra1),
        4 => Some(egui::PointerButton::Extra2),
        _ => None,
    }
}

pub fn keycode_egui(code: &str) -> Option<egui::Key> {
    if let Some(k) = letter_egui(code) {
        return Some(k);
    }
    if let Some(k) = digit_egui(code) {
        return Some(k);
    }
    if let Some(k) = function_egui(code) {
        return Some(k);
    }
    match code {
        "Space" => Some(egui::Key::Space),
        "Enter" => Some(egui::Key::Enter),
        "Escape" => Some(egui::Key::Escape),
        "Tab" => Some(egui::Key::Tab),
        "Backspace" => Some(egui::Key::Backspace),
        "Delete" => Some(egui::Key::Delete),
        "ArrowUp" => Some(egui::Key::ArrowUp),
        "ArrowDown" => Some(egui::Key::ArrowDown),
        "ArrowLeft" => Some(egui::Key::ArrowLeft),
        "ArrowRight" => Some(egui::Key::ArrowRight),
        "Home" => Some(egui::Key::Home),
        "End" => Some(egui::Key::End),
        "PageUp" => Some(egui::Key::PageUp),
        "PageDown" => Some(egui::Key::PageDown),
        "Backquote" => Some(egui::Key::Backtick),
        "Minus" => Some(egui::Key::Minus),
        "Equal" => Some(egui::Key::Equals),
        _ => None,
    }
}

fn letter_egui(code: &str) -> Option<egui::Key> {
    match code {
        "KeyA" => Some(egui::Key::A),
        "KeyB" => Some(egui::Key::B),
        "KeyC" => Some(egui::Key::C),
        "KeyD" => Some(egui::Key::D),
        "KeyE" => Some(egui::Key::E),
        "KeyF" => Some(egui::Key::F),
        "KeyG" => Some(egui::Key::G),
        "KeyH" => Some(egui::Key::H),
        "KeyI" => Some(egui::Key::I),
        "KeyJ" => Some(egui::Key::J),
        "KeyK" => Some(egui::Key::K),
        "KeyL" => Some(egui::Key::L),
        "KeyM" => Some(egui::Key::M),
        "KeyN" => Some(egui::Key::N),
        "KeyO" => Some(egui::Key::O),
        "KeyP" => Some(egui::Key::P),
        "KeyQ" => Some(egui::Key::Q),
        "KeyR" => Some(egui::Key::R),
        "KeyS" => Some(egui::Key::S),
        "KeyT" => Some(egui::Key::T),
        "KeyU" => Some(egui::Key::U),
        "KeyV" => Some(egui::Key::V),
        "KeyW" => Some(egui::Key::W),
        "KeyX" => Some(egui::Key::X),
        "KeyY" => Some(egui::Key::Y),
        "KeyZ" => Some(egui::Key::Z),
        _ => None,
    }
}

fn digit_egui(code: &str) -> Option<egui::Key> {
    match code {
        "Digit0" => Some(egui::Key::Num0),
        "Digit1" => Some(egui::Key::Num1),
        "Digit2" => Some(egui::Key::Num2),
        "Digit3" => Some(egui::Key::Num3),
        "Digit4" => Some(egui::Key::Num4),
        "Digit5" => Some(egui::Key::Num5),
        "Digit6" => Some(egui::Key::Num6),
        "Digit7" => Some(egui::Key::Num7),
        "Digit8" => Some(egui::Key::Num8),
        "Digit9" => Some(egui::Key::Num9),
        _ => None,
    }
}

fn function_egui(code: &str) -> Option<egui::Key> {
    match code {
        "F1" => Some(egui::Key::F1),
        "F2" => Some(egui::Key::F2),
        "F3" => Some(egui::Key::F3),
        "F4" => Some(egui::Key::F4),
        "F5" => Some(egui::Key::F5),
        "F6" => Some(egui::Key::F6),
        "F7" => Some(egui::Key::F7),
        "F8" => Some(egui::Key::F8),
        "F9" => Some(egui::Key::F9),
        "F10" => Some(egui::Key::F10),
        "F11" => Some(egui::Key::F11),
        "F12" => Some(egui::Key::F12),
        _ => None,
    }
}
