use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

use super::canvas::{canvas_metrics, CanvasMetrics};
use super::host_api::HostChannel;
use super::{
    is_visible, set_msg_bool, set_msg_f32, set_msg_f64, set_msg_string, set_msg_u32, Link,
};
use crate::session::input::message::wheel_lines;
use crate::web::error::{js, listener, WebError};
use crate::web::messages::Kind;

type Pending<T> = Rc<RefCell<Option<T>>>;

pub(super) fn install_dom_input_forwarders(
    worker: &Link,
    canvas: &HtmlCanvasElement,
    launched: CanvasMetrics,
    max_pixels: Option<u32>,
    background: bool,
    channel: &Rc<RefCell<HostChannel>>,
) -> Result<(), WebError> {
    let window = web_sys::window().ok_or(WebError::NoWindow)?;
    let document = window.document().ok_or(WebError::NoDocument)?;

    const RESIZE_DEBOUNCE_FRAMES: u32 = 6;
    {
        let pending: Pending<(CanvasMetrics, u32)> = Rc::new(RefCell::new(None));
        let pending_for_listener = pending.clone();
        let viewport: Pending<(f32, f32)> = Rc::new(RefCell::new(None));
        let viewport_for_listener = viewport.clone();
        let canvas_for_listener = canvas.clone();
        let window_for_listener = window.clone();
        let cb = Closure::wrap(Box::new(move || {
            let metrics = canvas_metrics(
                &canvas_for_listener,
                window_for_listener.device_pixel_ratio() as f32,
                max_pixels,
            );
            *pending_for_listener.borrow_mut() = Some((metrics, 0));
            *viewport_for_listener.borrow_mut() = Some((
                canvas_for_listener.client_width().max(1) as f32,
                canvas_for_listener.client_height().max(1) as f32,
            ));
        }) as Box<dyn FnMut()>);
        window
            .add_event_listener_with_callback("resize", cb.as_ref().unchecked_ref())
            .map_err(js("window resize listener"))?;
        let observer = web_sys::ResizeObserver::new(cb.as_ref().unchecked_ref())
            .map_err(js("ResizeObserver::new"))?;
        observer.observe(canvas);
        cb.forget();

        let worker_for_raf = worker.clone();
        let pending_for_raf = pending.clone();
        let window_for_raf = window.clone();
        let channel_for_resize = channel.clone();
        let committed = Cell::new(launched);
        let raf_cb: Pending<Closure<dyn FnMut()>> = Rc::new(RefCell::new(None));
        let raf_cb_for_closure = raf_cb.clone();
        *raf_cb.borrow_mut() = Some(Closure::wrap(Box::new(move || {
            if channel_for_resize.borrow().ready() {
                if let Some((width, height)) = viewport.borrow_mut().take() {
                    let msg = Kind::Viewport.message();
                    set_msg_f32(&msg, "width", width);
                    set_msg_f32(&msg, "height", height);
                    let _ = worker_for_raf.post_message(&msg);
                }
            }
            let commit = {
                let mut p = pending_for_raf.borrow_mut();
                match p.as_mut() {
                    Some((_, frames)) => {
                        *frames += 1;
                        if *frames >= RESIZE_DEBOUNCE_FRAMES && channel_for_resize.borrow().ready()
                        {
                            p.take().map(|(metrics, _)| metrics)
                        } else {
                            None
                        }
                    }
                    None => None,
                }
            };
            if let Some(metrics) = commit.filter(|metrics| *metrics != committed.get()) {
                committed.set(metrics);
                let msg = Kind::Resize.message();
                set_msg_u32(&msg, "width", metrics.width);
                set_msg_u32(&msg, "height", metrics.height);
                set_msg_f32(&msg, "dpr", metrics.scale);
                let _ = worker_for_raf.post_message(&msg);
            }
            let cb_ref = raf_cb_for_closure.borrow();
            if let Some(cb) = cb_ref.as_ref() {
                let _ = window_for_raf.request_animation_frame(cb.as_ref().unchecked_ref());
            }
        }) as Box<dyn FnMut()>));
        {
            let first_cb = raf_cb.borrow();
            if let Some(first) = first_cb.as_ref() {
                window
                    .request_animation_frame(first.as_ref().unchecked_ref())
                    .map_err(js("resize rAF init"))?;
            }
        }
    }

    {
        let pending: Pending<(f32, f32, u32, f32, f32, f64)> = Rc::new(RefCell::new(None));
        let pending_for_listener = pending.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::MouseEvent| {
            let mut p = pending_for_listener.borrow_mut();
            let (sum_dx, sum_dy) = match *p {
                Some((_, _, _, dx, dy, _)) => (dx, dy),
                None => (0.0, 0.0),
            };
            *p = Some((
                ev.offset_x() as f32,
                ev.offset_y() as f32,
                ev.buttons() as u32,
                sum_dx + ev.movement_x() as f32,
                sum_dy + ev.movement_y() as f32,
                ev.time_stamp(),
            ));
        }) as Box<dyn FnMut(web_sys::MouseEvent)>);
        canvas
            .add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref())
            .map_err(js("mousemove listener"))?;
        cb.forget();

        let worker_for_raf = worker.clone();
        let pending_for_raf = pending.clone();
        let window_for_raf = window.clone();
        let channel_for_raf = channel.clone();
        let raf_cb: Pending<Closure<dyn FnMut()>> = Rc::new(RefCell::new(None));
        let raf_cb_for_closure = raf_cb.clone();
        *raf_cb.borrow_mut() = Some(Closure::wrap(Box::new(move || {
            channel_for_raf.borrow_mut().flush(&worker_for_raf);
            if let Some((x, y, buttons, dx, dy, time)) = pending_for_raf.borrow_mut().take() {
                let msg = Kind::MouseMove.message();
                set_msg_f32(&msg, "x", x);
                set_msg_f32(&msg, "y", y);
                set_msg_u32(&msg, "buttons", buttons);
                set_msg_f32(&msg, "dx", dx);
                set_msg_f32(&msg, "dy", dy);
                set_msg_f64(&msg, "time", time);
                let _ = worker_for_raf.post_message(&msg);
            }
            let cb_ref = raf_cb_for_closure.borrow();
            if let Some(cb) = cb_ref.as_ref() {
                let _ = window_for_raf.request_animation_frame(cb.as_ref().unchecked_ref());
            }
        }) as Box<dyn FnMut()>));
        {
            let first_cb = raf_cb.borrow();
            if let Some(first) = first_cb.as_ref() {
                window
                    .request_animation_frame(first.as_ref().unchecked_ref())
                    .map_err(js("mousemove rAF init"))?;
            }
        }
    }

    for (event_name, pressed) in [("mousedown", true), ("mouseup", false)] {
        let worker = worker.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::MouseEvent| {
            let prevent = if background {
                pressed && ev.button() == 0
            } else {
                ev.button() != 0
            };
            if prevent {
                ev.prevent_default();
            }
            let msg = Kind::MouseButton.message();
            set_msg_f32(&msg, "x", ev.offset_x() as f32);
            set_msg_f32(&msg, "y", ev.offset_y() as f32);
            set_msg_u32(&msg, "button", ev.button() as u32);
            set_msg_bool(&msg, "pressed", pressed);
            set_msg_f64(&msg, "time", ev.time_stamp());
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut(web_sys::MouseEvent)>);
        canvas
            .add_event_listener_with_callback(event_name, cb.as_ref().unchecked_ref())
            .map_err(listener(event_name))?;
        cb.forget();
    }

    if !background {
        let _ = canvas.style().set_property("touch-action", "none");
    }
    for event_name in ["pointerdown", "pointermove", "pointerup", "pointercancel"] {
        let worker = worker.clone();
        let canvas_for_capture = canvas.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::PointerEvent| {
            if event_name == "pointerdown" {
                let _ = canvas_for_capture.set_pointer_capture(ev.pointer_id());
            }
            if ev.pointer_type() == "mouse" {
                return;
            }
            let msg = Kind::Pointer.message();
            set_msg_f64(&msg, "id", f64::from(ev.pointer_id()));
            set_msg_f32(&msg, "x", ev.offset_x() as f32);
            set_msg_f32(&msg, "y", ev.offset_y() as f32);
            set_msg_string(&msg, "phase", event_name);
            set_msg_f64(&msg, "time", ev.time_stamp());
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut(web_sys::PointerEvent)>);
        canvas
            .add_event_listener_with_callback(event_name, cb.as_ref().unchecked_ref())
            .map_err(listener(event_name))?;
        cb.forget();
    }

    if background {
        install_focus_and_visibility(worker, &window, &document)?;
        return Ok(());
    }

    {
        let cb = Closure::wrap(Box::new(move |ev: web_sys::Event| {
            ev.prevent_default();
        }) as Box<dyn FnMut(web_sys::Event)>);
        canvas
            .add_event_listener_with_callback("contextmenu", cb.as_ref().unchecked_ref())
            .map_err(js("contextmenu listener"))?;
        cb.forget();
    }

    {
        let worker = worker.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::WheelEvent| {
            let delta = [ev.delta_x() as f32, ev.delta_y() as f32];
            // Page-mode deltas are scaled as pixels too.
            let [dx, dy] = match ev.delta_mode() {
                1 => delta,
                _ => wheel_lines(delta),
            };
            ev.prevent_default();
            let msg = Kind::MouseWheel.message();
            set_msg_f32(&msg, "dx", dx);
            set_msg_f32(&msg, "dy", dy);
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut(web_sys::WheelEvent)>);
        canvas
            .add_event_listener_with_callback("wheel", cb.as_ref().unchecked_ref())
            .map_err(js("wheel listener"))?;
        cb.forget();
    }

    for (event_name, pressed) in [("keydown", true), ("keyup", false)] {
        let worker = worker.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::KeyboardEvent| {
            let code = ev.code();
            let no_modifier = !ev.ctrl_key() && !ev.alt_key() && !ev.meta_key();
            let is_alt_self = matches!(code.as_str(), "AltLeft" | "AltRight");
            let suppress_alt = is_alt_self && !ev.ctrl_key() && !ev.meta_key();
            let owned_unmodified = matches!(
                code.as_str(),
                "Tab"
                    | "Space"
                    | "ArrowLeft"
                    | "ArrowRight"
                    | "ArrowUp"
                    | "ArrowDown"
                    | "Slash"
                    | "Quote",
            );
            if suppress_alt || (owned_unmodified && no_modifier) {
                ev.prevent_default();
            }
            let msg = Kind::Key.message();
            set_msg_string(&msg, "code", &code);
            set_msg_string(&msg, "key", &ev.key());
            set_msg_bool(&msg, "pressed", pressed);
            set_msg_bool(&msg, "repeat", ev.repeat());
            set_msg_bool(&msg, "ctrl", ev.ctrl_key());
            set_msg_bool(&msg, "shift", ev.shift_key());
            set_msg_bool(&msg, "alt", ev.alt_key());
            set_msg_bool(&msg, "meta", ev.meta_key());
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut(web_sys::KeyboardEvent)>);
        document
            .add_event_listener_with_callback(event_name, cb.as_ref().unchecked_ref())
            .map_err(listener(event_name))?;
        cb.forget();
    }

    install_focus_and_visibility(worker, &window, &document)
}

fn install_focus_and_visibility(
    worker: &Link,
    window: &web_sys::Window,
    document: &web_sys::Document,
) -> Result<(), WebError> {
    for (event_name, focused) in [("focus", true), ("blur", false)] {
        let worker = worker.clone();
        let cb = Closure::wrap(Box::new(move || {
            let msg = Kind::Focus.message();
            set_msg_bool(&msg, "focused", focused);
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut()>);
        window
            .add_event_listener_with_callback(event_name, cb.as_ref().unchecked_ref())
            .map_err(listener(event_name))?;
        cb.forget();
    }

    {
        let worker = worker.clone();
        let document_for_query = document.clone();
        let cb = Closure::wrap(Box::new(move || {
            let msg = Kind::Visibility.message();
            set_msg_bool(&msg, "visible", is_visible(&document_for_query));
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut()>);
        document
            .add_event_listener_with_callback("visibilitychange", cb.as_ref().unchecked_ref())
            .map_err(js("visibilitychange listener"))?;
        cb.forget();
    }

    Ok(())
}
