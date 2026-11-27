use std::fmt;

use wasm_bindgen::JsValue;

/// Why the browser side of a launch failed; a thrown value is kept as its debug text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WebError {
    /// The text names the browser call that threw.
    Js {
        call: &'static str,
        thrown: String,
    },
    Listener {
        event: &'static str,
        thrown: String,
    },
    NoWindow,
    NoDocument,
    NoHead,
    NoElement(String),
    WrongElement {
        id: String,
        kind: &'static str,
    },
    /// A freshly created element did not have the named type.
    Created(&'static str),
    NotWorker,
    NoBundleUrl,
    NoMessageKind,
    SecondInit,
    WorkerUnavailable,
}

impl fmt::Display for WebError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Js { call, thrown } => write!(f, "{call}: {thrown}"),
            Self::Listener { event, thrown } => write!(f, "{event} listener: {thrown}"),
            Self::NoWindow => f.write_str("no global window"),
            Self::NoDocument => f.write_str("no document on global window"),
            Self::NoHead => f.write_str("no <head> element"),
            Self::NoElement(id) => write!(f, "no element with id '{id}'"),
            Self::WrongElement { id, kind } => write!(f, "element '{id}' is not a {kind}"),
            Self::Created(kind) => write!(f, "created element is not {kind}"),
            Self::NotWorker => f.write_str("not running in a DedicatedWorkerGlobalScope"),
            Self::NoBundleUrl => {
                f.write_str("__loam_wasm_url is not a string; demo's index.html must set it")
            }
            Self::NoMessageKind => f.write_str("postMessage missing 'kind' field"),
            Self::SecondInit => f.write_str("a second init reached the session worker"),
            Self::WorkerUnavailable => f.write_str("the session worker is unavailable"),
        }
    }
}

impl std::error::Error for WebError {}

pub(crate) fn js(call: &'static str) -> impl FnOnce(JsValue) -> WebError {
    move |thrown| WebError::Js {
        call,
        thrown: format!("{thrown:?}"),
    }
}

pub(crate) fn listener(event: &'static str) -> impl FnOnce(JsValue) -> WebError {
    move |thrown| WebError::Listener {
        event,
        thrown: format!("{thrown:?}"),
    }
}
