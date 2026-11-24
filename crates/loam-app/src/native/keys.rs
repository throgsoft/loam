use crate::host::Key;

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

pub fn winit_alt(event: &winit::event::KeyEvent) -> Option<usize> {
    use winit::keyboard::{KeyCode, PhysicalKey};

    match event.physical_key {
        PhysicalKey::Code(KeyCode::AltLeft) => Some(0),
        PhysicalKey::Code(KeyCode::AltRight) => Some(1),
        _ => None,
    }
}
