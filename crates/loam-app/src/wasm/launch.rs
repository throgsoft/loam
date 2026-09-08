//! Demos opt into click-to-start by marking the host element in `index.html`.

use anyhow::{anyhow, Result};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlButtonElement, HtmlStyleElement};

const LAUNCH_OVERLAY_CSS: &str = r#"
.loam-demo-launch {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font: inherit;
    font-size: 14px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #d8d8df;
    background: rgba(14, 14, 18, 0.35);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: none;
    transition: background 200ms ease, opacity 200ms ease;
}

.loam-demo-launch::after {
    display: none;
}

.loam-demo-launch.ready {
    cursor: pointer;
}
.loam-demo-launch.ready::after {
    display: inline-block;
    content: 'Click anywhere to launch';
    padding: 14px 28px;
    border: 1px solid rgba(200, 200, 220, 0.5);
    border-radius: 6px;
    background: rgba(20, 20, 28, 0.55);
}
.loam-demo-launch.ready:hover {
    background: rgba(14, 14, 18, 0.25);
}
.loam-demo-launch.ready:hover::after {
    background: rgba(28, 28, 38, 0.65);
    border-color: rgba(220, 220, 240, 0.7);
}
.loam-demo-launch.ready:active {
    background: rgba(14, 14, 18, 0.4);
}

.loam-demo-launch.ready.resume::after {
    content: 'Click to start';
}
"#;

const OVERLAY_STYLE_ID: &str = "loam-launch-overlay-styles";

/// No state class: CSS shows only the blurred backdrop until the worker's
/// `preview_ready` adds `.ready`.
pub fn inject_launch_overlay(host_id: &str, button_id: &str) -> Result<HtmlButtonElement> {
    overlay_button(host_id, button_id, "loam-demo-launch", "Launch demo")
}

/// The paused-state affordance (`.ready.resume`: immediately clickable, a paused
/// demo is warm).
pub fn show_resume_overlay(host_id: &str, button_id: &str) -> Result<HtmlButtonElement> {
    overlay_button(
        host_id,
        button_id,
        "loam-demo-launch ready resume",
        "Resume demo",
    )
}

fn overlay_button(
    host_id: &str,
    button_id: &str,
    class_name: &str,
    aria_label: &str,
) -> Result<HtmlButtonElement> {
    let window = web_sys::window().ok_or_else(|| anyhow!("no global window"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("no document on window"))?;
    let host = document
        .get_element_by_id(host_id)
        .ok_or_else(|| anyhow!("no host element with id '{host_id}'"))?;

    if document.get_element_by_id(OVERLAY_STYLE_ID).is_none() {
        let head = document
            .head()
            .ok_or_else(|| anyhow!("no <head> element"))?;
        let style = document
            .create_element("style")
            .map_err(|e| anyhow!("create <style>: {e:?}"))?
            .dyn_into::<HtmlStyleElement>()
            .map_err(|_| anyhow!("created element is not HtmlStyleElement"))?;
        style.set_id(OVERLAY_STYLE_ID);
        style.set_text_content(Some(LAUNCH_OVERLAY_CSS));
        head.append_child(&style)
            .map_err(|e| anyhow!("append <style>: {e:?}"))?;
    }

    if let Some(existing) = document.get_element_by_id(button_id) {
        let button = existing
            .dyn_into::<HtmlButtonElement>()
            .map_err(|_| anyhow!("element '{button_id}' is not a button"))?;
        button.set_class_name(class_name);
        return Ok(button);
    }

    let button = document
        .create_element("button")
        .map_err(|e| anyhow!("create <button>: {e:?}"))?
        .dyn_into::<HtmlButtonElement>()
        .map_err(|_| anyhow!("created element is not HtmlButtonElement"))?;
    button.set_id(button_id);
    button.set_class_name(class_name);
    button.set_type("button");
    button
        .set_attribute("aria-label", aria_label)
        .map_err(|e| anyhow!("set aria-label: {e:?}"))?;
    host.append_child(&button)
        .map_err(|e| anyhow!("append button to host: {e:?}"))?;
    Ok(button)
}

/// False on a missing element, missing attribute, or any other value, which is
/// the auto-launch default.
pub fn is_manual_mode(host_id: &str) -> bool {
    let Some(window) = web_sys::window() else {
        return false;
    };
    let Some(document) = window.document() else {
        return false;
    };
    let Some(el) = document.get_element_by_id(host_id) else {
        return false;
    };
    el.get_attribute("data-mode")
        .map(|m| m == "manual")
        .unwrap_or(false)
}

/// `None` where `performance.memory.usedJSHeapSize` is absent: Chromium exposes
/// it as a non-standard extension, Firefox and Safari do not, and `web-sys` does
/// not surface it, hence `js_sys::Reflect`.
pub fn js_heap_sampler() -> Option<u64> {
    let window = web_sys::window()?;
    let performance = window.performance()?;
    let perf_val: &JsValue = performance.as_ref();
    let memory = js_sys::Reflect::get(perf_val, &JsValue::from_str("memory")).ok()?;
    if memory.is_undefined() || memory.is_null() {
        return None;
    }
    let used = js_sys::Reflect::get(&memory, &JsValue::from_str("usedJSHeapSize")).ok()?;
    let bytes = used.as_f64()?;
    if bytes.is_finite() && bytes >= 0.0 {
        Some(bytes as u64)
    } else {
        None
    }
}
