use std::cell::Cell;
use std::rc::Rc;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlCanvasElement, MessageEvent};

use super::{set_msg_bool, Link};
use crate::web::error::{js, listener, WebError};
use crate::web::messages::Kind;

pub(super) fn install_cursor_control(
    worker: &Link,
    canvas: &HtmlCanvasElement,
    background: bool,
) -> Result<(), WebError> {
    let document = web_sys::window()
        .and_then(|window| window.document())
        .ok_or(WebError::NoDocument)?;
    let wanted = Rc::new(Cell::new(false));

    {
        let worker = worker.clone();
        let worker_for_callback = worker.clone();
        let canvas = canvas.clone();
        let document = document.clone();
        let wanted = wanted.clone();
        let callback = Closure::wrap(Box::new(move |event: MessageEvent| {
            let data: JsValue = event.data();
            if Kind::of(&data) != Some(Kind::CursorLock) {
                return;
            }
            let locked = js_sys::Reflect::get(&data, &JsValue::from_str("locked"))
                .ok()
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            wanted.set(locked);
            if locked {
                request_pointer_lock(&canvas);
            } else if document
                .pointer_lock_element()
                .as_ref()
                .is_some_and(|element| element == canvas.as_ref())
            {
                document.exit_pointer_lock();
            } else {
                let message = Kind::PointerLockChanged.message();
                set_msg_bool(&message, "locked", false);
                set_msg_bool(&message, "released", false);
                let _ = worker_for_callback.post_message(&message);
            }
        }) as Box<dyn FnMut(MessageEvent)>);
        worker
            .add_event_listener_with_callback("message", callback.as_ref().unchecked_ref())
            .map_err(js("cursor lock message listener"))?;
        callback.forget();
    }

    for event_name in ["pointerlockchange", "pointerlockerror"] {
        let release_event = event_name == "pointerlockchange";
        let worker = worker.clone();
        let canvas = canvas.clone();
        let document_for_event = document.clone();
        let wanted = wanted.clone();
        let callback = Closure::wrap(Box::new(move || {
            let locked = document_for_event
                .pointer_lock_element()
                .as_ref()
                .is_some_and(|element| element == canvas.as_ref());
            let released = release_event && !locked && wanted.replace(false);
            if locked && !wanted.get() {
                document_for_event.exit_pointer_lock();
            }
            let message = Kind::PointerLockChanged.message();
            set_msg_bool(&message, "locked", locked);
            set_msg_bool(&message, "released", released);
            let _ = worker.post_message(&message);
        }) as Box<dyn FnMut()>);
        document
            .add_event_listener_with_callback(event_name, callback.as_ref().unchecked_ref())
            .map_err(listener(event_name))?;
        callback.forget();
    }

    {
        let canvas = canvas.clone();
        let canvas_for_callback = canvas.clone();
        let document = document.clone();
        let wanted = wanted.clone();
        let callback = Closure::wrap(Box::new(move || {
            let locked = document
                .pointer_lock_element()
                .as_ref()
                .is_some_and(|element| element == canvas_for_callback.as_ref());
            if wanted.get() && !locked {
                request_pointer_lock(&canvas_for_callback);
            }
        }) as Box<dyn FnMut()>);
        canvas
            .add_event_listener_with_callback("mousedown", callback.as_ref().unchecked_ref())
            .map_err(js("cursor lock mousedown listener"))?;
        callback.forget();
    }

    if !background {
        let canvas = canvas.clone();
        let document_for_event = document.clone();
        let wanted = wanted.clone();
        let callback = Closure::wrap(Box::new(move || {
            let locked = document_for_event
                .pointer_lock_element()
                .as_ref()
                .is_some_and(|element| element == canvas.as_ref());
            if wanted.get() && !locked {
                request_pointer_lock(&canvas);
            }
        }) as Box<dyn FnMut()>);
        document
            .add_event_listener_with_callback("keydown", callback.as_ref().unchecked_ref())
            .map_err(js("cursor lock keydown listener"))?;
        callback.forget();
    }

    Ok(())
}

fn request_pointer_lock(canvas: &HtmlCanvasElement) {
    let Ok(method) =
        js_sys::Reflect::get(canvas.as_ref(), &JsValue::from_str("requestPointerLock"))
            .and_then(|value| value.dyn_into::<js_sys::Function>())
    else {
        tracing::warn!("loam_app::web::page::pointer_lock: requestPointerLock is unavailable");
        return;
    };
    let Ok(result) = method.call0(canvas.as_ref()) else {
        tracing::warn!("loam_app::web::page::pointer_lock: requestPointerLock failed");
        return;
    };
    let Ok(promise) = result.dyn_into::<js_sys::Promise>() else {
        return;
    };
    wasm_bindgen_futures::spawn_local(async move {
        if let Err(error) = wasm_bindgen_futures::JsFuture::from(promise).await {
            tracing::warn!(
                "loam_app::web::page::pointer_lock: requestPointerLock rejected: {error:?}"
            );
        }
    });
}
