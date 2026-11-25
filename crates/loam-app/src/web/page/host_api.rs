use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{Element, MessageEvent};

use super::{set_msg_string, Link};
use crate::session::input::queue::MESSAGE_QUEUE_CAPACITY;
use crate::web::error::{js, listener, WebError};
use crate::web::messages::Kind;

const HOST_CONTROL_EVENT: &str = "loam-control";
const HOST_MESSAGE_EVENT: &str = "loam-message";
const HOST_POST_EVENT: &str = "loam-post";
// A page iterable can be unbounded; this caps the copy per message.
const MAX_HOST_VALUES: usize = 64;

struct HostMessage {
    topic: String,
    values: Vec<f32>,
}

#[derive(Default)]
pub(super) struct HostChannel {
    ready: bool,
    control: Option<Kind>,
    pending: VecDeque<HostMessage>,
}

impl HostChannel {
    pub(super) fn ready(&self) -> bool {
        self.ready
    }

    pub(super) fn open(&mut self) -> Option<Kind> {
        self.ready = true;
        self.control.take()
    }

    fn push(&mut self, topic: &str, values: Vec<f32>) {
        if !values.is_empty() {
            let earlier = self
                .pending
                .iter_mut()
                .rev()
                .take_while(|message| !message.values.is_empty())
                .find(|message| message.topic == topic);
            if let Some(earlier) = earlier {
                earlier.values = values;
                return;
            }
        }
        if self.pending.len() >= MESSAGE_QUEUE_CAPACITY {
            self.pending.pop_front();
        }
        self.pending.push_back(HostMessage {
            topic: topic.to_owned(),
            values,
        });
    }

    pub(super) fn flush(&mut self, worker: &Link) {
        if !self.ready {
            return;
        }
        for message in self.pending.drain(..) {
            let msg = Kind::HostMessage.message();
            set_msg_string(&msg, "topic", &message.topic);
            let _ = js_sys::Reflect::set(
                &msg,
                &JsValue::from_str("values"),
                &js_sys::Float32Array::from(message.values.as_slice()),
            );
            if let Err(error) = worker.post_message(&msg) {
                tracing::error!(
                    "loam_app::web::page::host_api: postMessage host_message failed: {error:?}"
                );
            }
        }
    }
}

fn read_host_values(values: &JsValue) -> Vec<f32> {
    if let Some(number) = values.as_f64() {
        return vec![number as f32];
    }
    match js_sys::try_iter(values) {
        Ok(Some(items)) => items
            .take(MAX_HOST_VALUES)
            .map(|item| item.ok().and_then(|item| item.as_f64()).unwrap_or(f64::NAN) as f32)
            .collect(),
        _ => Vec::new(),
    }
}

pub(super) fn install_host_channel(
    worker: &Link,
    host: &Element,
    channel: &Rc<RefCell<HostChannel>>,
) -> Result<(), WebError> {
    {
        let channel = channel.clone();
        let callback = Closure::wrap(Box::new(move |event: web_sys::CustomEvent| {
            let detail = event.detail();
            let topic = js_sys::Reflect::get(&detail, &JsValue::from_str("topic"))
                .ok()
                .and_then(|value| value.as_string());
            let Some(topic) = topic else {
                tracing::warn!(
                    "loam_app::web::page::host_api: {HOST_MESSAGE_EVENT} without a string topic ignored"
                );
                return;
            };
            let values = js_sys::Reflect::get(&detail, &JsValue::from_str("values"))
                .map(|values| read_host_values(&values))
                .unwrap_or_default();
            channel.borrow_mut().push(&topic, values);
        }) as Box<dyn FnMut(web_sys::CustomEvent)>);
        host.add_event_listener_with_callback(
            HOST_MESSAGE_EVENT,
            callback.as_ref().unchecked_ref(),
        )
        .map_err(listener(HOST_MESSAGE_EVENT))?;
        callback.forget();
    }

    let host = host.clone();
    let callback = Closure::wrap(Box::new(move |event: MessageEvent| {
        let data: JsValue = event.data();
        if Kind::of(&data) != Some(Kind::HostPost) {
            return;
        }
        let detail = js_sys::Object::new();
        for key in ["topic", "values"] {
            let key = JsValue::from_str(key);
            if let Ok(value) = js_sys::Reflect::get(&data, &key) {
                let _ = js_sys::Reflect::set(&detail, &key, &value);
            }
        }
        let init = web_sys::CustomEventInit::new();
        init.set_detail(&detail);
        match web_sys::CustomEvent::new_with_event_init_dict(HOST_POST_EVENT, &init) {
            Ok(event) => {
                let _ = host.dispatch_event(&event);
            }
            Err(error) => {
                tracing::error!(
                    "loam_app::web::page::host_api: create {HOST_POST_EVENT} event: {error:?}"
                )
            }
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    worker
        .add_event_listener_with_callback("message", callback.as_ref().unchecked_ref())
        .map_err(js("worker.addEventListener('message') for posts"))?;
    callback.forget();
    Ok(())
}

pub(super) fn install_page_control(
    worker: &Link,
    host: &Element,
    channel: &Rc<RefCell<HostChannel>>,
) -> Result<(), WebError> {
    let worker = worker.clone();
    let channel = channel.clone();
    let callback = Closure::wrap(Box::new(move |event: web_sys::CustomEvent| {
        let detail = event.detail().as_string();
        let kind = match detail.as_deref() {
            Some("pause") => Kind::HostPause,
            Some("resume") => Kind::HostResume,
            other => {
                tracing::warn!(
                    "loam_app::web::page::host_api: {HOST_CONTROL_EVENT} ignored: {other:?}"
                );
                return;
            }
        };
        let mut channel = channel.borrow_mut();
        if !channel.ready {
            channel.control = Some(kind);
            return;
        }
        if let Err(error) = worker.post_message(&kind.message()) {
            tracing::error!(
                "loam_app::web::page::host_api: postMessage {kind:?} failed: {error:?}"
            );
        }
    }) as Box<dyn FnMut(web_sys::CustomEvent)>);
    host.add_event_listener_with_callback(HOST_CONTROL_EVENT, callback.as_ref().unchecked_ref())
        .map_err(listener(HOST_CONTROL_EVENT))?;
    callback.forget();
    Ok(())
}
