use anyhow::Result;
use std::time::Duration;
use wasm_bindgen::{JsCast, JsValue};

use crate::session::input::message::{InputMessage, PointerPhase};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Init,
    Start,
    Pause,
    Resume,
    HostPause,
    HostResume,
    Visibility,
    Resize,
    Viewport,
    MouseMove,
    MouseButton,
    MouseWheel,
    Key,
    Focus,
    PointerLockChanged,
    Pointer,
    HostMessage,
    Ready,
    PreviewReady,
    Presented,
    Error,
    CursorLock,
    HostPost,
}

impl Kind {
    const ALL: [Self; 23] = [
        Self::Init,
        Self::Start,
        Self::Pause,
        Self::Resume,
        Self::HostPause,
        Self::HostResume,
        Self::Visibility,
        Self::Resize,
        Self::Viewport,
        Self::MouseMove,
        Self::MouseButton,
        Self::MouseWheel,
        Self::Key,
        Self::Focus,
        Self::PointerLockChanged,
        Self::Pointer,
        Self::HostMessage,
        Self::Ready,
        Self::PreviewReady,
        Self::Presented,
        Self::Error,
        Self::CursorLock,
        Self::HostPost,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Init => "init",
            Self::Start => "start",
            Self::Pause => "pause",
            Self::Resume => "resume",
            Self::HostPause => "host_pause",
            Self::HostResume => "host_resume",
            Self::Visibility => "visibility",
            Self::Resize => "resize",
            Self::Viewport => "viewport",
            Self::MouseMove => "mouse_move",
            Self::MouseButton => "mouse_button",
            Self::MouseWheel => "mouse_wheel",
            Self::Key => "key",
            Self::Focus => "focus",
            Self::PointerLockChanged => "pointer_lock_changed",
            Self::Pointer => "pointer",
            Self::HostMessage => "host_message",
            Self::Ready => "ready",
            Self::PreviewReady => "preview_ready",
            Self::Presented => "presented",
            Self::Error => "error",
            Self::CursorLock => "cursor_lock",
            Self::HostPost => "host_post",
        }
    }

    fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }

    pub(crate) fn of(data: &JsValue) -> Option<Self> {
        read_string_field(data, "kind").and_then(|name| Self::parse(&name))
    }

    pub(crate) fn message(self) -> js_sys::Object {
        let message = js_sys::Object::new();
        let _ = js_sys::Reflect::set(
            &message,
            &JsValue::from_str("kind"),
            &JsValue::from_str(self.name()),
        );
        message
    }
}

/// `Ok(None)` covers the kinds that carry no input, which the caller handles, and unknown kinds, which are dropped.
pub fn parse_non_init(data: &JsValue) -> Result<Option<InputMessage>> {
    let name = read_string_field(data, "kind")
        .ok_or_else(|| anyhow::anyhow!("postMessage missing 'kind' field"))?;
    let Some(kind) = Kind::parse(&name) else {
        return Ok(None);
    };

    let msg = match kind {
        Kind::Resize => InputMessage::Resize {
            width: read_u32_field(data, "width").unwrap_or(0),
            height: read_u32_field(data, "height").unwrap_or(0),
            dpr: read_device_pixel_ratio(data),
        },
        Kind::Viewport => InputMessage::Viewport {
            width: read_f32_field(data, "width").unwrap_or(0.0),
            height: read_f32_field(data, "height").unwrap_or(0.0),
        },
        Kind::MouseMove => InputMessage::MouseMove {
            x: read_f32_field(data, "x").unwrap_or(0.0),
            y: read_f32_field(data, "y").unwrap_or(0.0),
            buttons: read_u32_field(data, "buttons").unwrap_or(0) as u8,
            dx: read_f32_field(data, "dx").unwrap_or(0.0),
            dy: read_f32_field(data, "dy").unwrap_or(0.0),
            time: read_event_time(data),
        },
        Kind::MouseButton => InputMessage::MouseButton {
            x: read_f32_field(data, "x").unwrap_or(0.0),
            y: read_f32_field(data, "y").unwrap_or(0.0),
            button: read_u32_field(data, "button").unwrap_or(0) as u8,
            pressed: read_bool_field(data, "pressed").unwrap_or(false),
            time: read_event_time(data),
        },
        Kind::MouseWheel => InputMessage::MouseWheel {
            dx: read_f32_field(data, "dx").unwrap_or(0.0),
            dy: read_f32_field(data, "dy").unwrap_or(0.0),
        },
        Kind::Key => InputMessage::Key {
            code: read_string_field(data, "code").unwrap_or_default(),
            key: read_string_field(data, "key").unwrap_or_default(),
            pressed: read_bool_field(data, "pressed").unwrap_or(false),
            repeat: read_bool_field(data, "repeat").unwrap_or(false),
            ctrl: read_bool_field(data, "ctrl").unwrap_or(false),
            shift: read_bool_field(data, "shift").unwrap_or(false),
            alt: read_bool_field(data, "alt").unwrap_or(false),
            meta: read_bool_field(data, "meta").unwrap_or(false),
        },
        Kind::Focus => InputMessage::Focus(read_bool_field(data, "focused").unwrap_or(false)),
        Kind::Visibility => {
            InputMessage::Visibility(read_bool_field(data, "visible").unwrap_or(false))
        }
        Kind::Start => InputMessage::Start,
        Kind::PointerLockChanged => InputMessage::PointerLockChanged {
            locked: read_bool_field(data, "locked").unwrap_or(false),
            released: read_bool_field(data, "released").unwrap_or(false),
        },
        Kind::Pointer => {
            let phase = match read_string_field(data, "phase").as_deref() {
                Some("pointerdown") => PointerPhase::Down,
                Some("pointermove") => PointerPhase::Move,
                Some("pointerup") => PointerPhase::Up,
                Some("pointercancel") => PointerPhase::Cancel,
                _ => return Ok(None),
            };
            InputMessage::Pointer {
                id: read_f64_field(data, "id").unwrap_or(0.0).max(0.0) as u64,
                x: read_f32_field(data, "x").unwrap_or(0.0),
                y: read_f32_field(data, "y").unwrap_or(0.0),
                phase,
                time: read_event_time(data),
            }
        }
        Kind::HostMessage => {
            let Some(topic) = read_string_field(data, "topic") else {
                return Ok(None);
            };
            InputMessage::Host {
                topic,
                values: read_f32_array_field(data, "values"),
            }
        }
        Kind::Init
        | Kind::Pause
        | Kind::Resume
        | Kind::HostPause
        | Kind::HostResume
        | Kind::Ready
        | Kind::PreviewReady
        | Kind::Presented
        | Kind::Error
        | Kind::CursorLock
        | Kind::HostPost => return Ok(None),
    };

    Ok(Some(msg))
}

fn read_f32_array_field(obj: &JsValue, key: &str) -> Vec<f32> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.dyn_into::<js_sys::Float32Array>().ok())
        .map(|array| array.to_vec())
        .unwrap_or_default()
}

fn read_event_time(obj: &JsValue) -> Duration {
    read_f64_field(obj, "time")
        .and_then(|ms| Duration::try_from_secs_f64(ms / 1000.0).ok())
        .unwrap_or_default()
}

/// A missing, zero, or non-finite ratio falls back to 1.0.
pub fn read_device_pixel_ratio(obj: &JsValue) -> f32 {
    read_f32_field(obj, "dpr")
        .filter(|dpr| dpr.is_finite() && *dpr > 0.0)
        .unwrap_or(1.0)
}

fn read_u32_field(obj: &JsValue, key: &str) -> Option<u32> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_f64())
        .map(|f| f as u32)
}

pub(super) fn read_f64_field(obj: &JsValue, key: &str) -> Option<f64> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_f64())
}

fn read_f32_field(obj: &JsValue, key: &str) -> Option<f32> {
    read_f64_field(obj, key).map(|f| f as f32)
}

fn read_bool_field(obj: &JsValue, key: &str) -> Option<bool> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_bool())
}

fn read_string_field(obj: &JsValue, key: &str) -> Option<String> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_string())
}
