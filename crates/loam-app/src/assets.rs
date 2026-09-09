//! Results arrive at frame boundaries. A request names its scene; a closed
//! scene's requests start nothing and its late results are dropped.

use std::collections::{HashMap, HashSet};
use std::path::{Component, Path};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use crate::Runtime;

/// Relative to the working directory on native and to the page on the browser.
pub const ASSET_DIR: &str = "assets";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SceneId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AssetRequest {
    pub request: RequestId,
    pub scene: SceneId,
}

#[derive(Clone, Debug)]
pub enum AssetState {
    Loading,
    /// Shared by every request for the same path.
    Ready(Arc<[u8]>),
    Failed(String),
}

#[derive(Clone, Debug)]
pub struct AssetEvent {
    pub request: AssetRequest,
    pub state: AssetState,
}

pub(crate) struct Completion {
    path: String,
    result: Result<Arc<[u8]>, String>,
}

pub(crate) struct Assets {
    next_scene: u64,
    next_request: u64,
    live: HashSet<SceneId>,
    submitted: Vec<(AssetRequest, String)>,
    in_flight: HashMap<String, Vec<AssetRequest>>,
    tx: Sender<Completion>,
    rx: Receiver<Completion>,
    #[cfg(target_arch = "wasm32")]
    base: std::rc::Rc<str>,
}

impl Default for Assets {
    fn default() -> Self {
        let (tx, rx) = channel();
        Self {
            next_scene: 1,
            next_request: 1,
            live: HashSet::new(),
            submitted: Vec::new(),
            in_flight: HashMap::new(),
            tx,
            rx,
            #[cfg(target_arch = "wasm32")]
            base: format!("{ASSET_DIR}/").into(),
        }
    }
}

fn stays_under_base(path: &str) -> bool {
    !path.is_empty()
        && Path::new(path)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

impl Assets {
    fn open_scene(&mut self) -> SceneId {
        let scene = SceneId(self.next_scene);
        self.next_scene += 1;
        self.live.insert(scene);
        scene
    }

    fn close_scene(&mut self, scene: SceneId) {
        self.live.remove(&scene);
    }

    fn request(&mut self, scene: SceneId, path: &str) -> AssetRequest {
        let request = AssetRequest {
            request: RequestId(self.next_request),
            scene,
        };
        self.next_request += 1;
        self.submitted.push((request, path.to_owned()));
        request
    }

    fn pump(
        &mut self,
        events: &mut Vec<AssetEvent>,
        mut spawn: impl FnMut(&str, Sender<Completion>),
    ) {
        for (request, path) in self.submitted.drain(..) {
            if !self.live.contains(&request.scene) {
                continue;
            }
            if !stays_under_base(&path) {
                events.push(AssetEvent {
                    request,
                    state: AssetState::Failed(format!(
                        "asset path `{path}` must be relative and stay under `{ASSET_DIR}`"
                    )),
                });
                continue;
            }
            match self.in_flight.get_mut(&path) {
                Some(interests) => interests.push(request),
                None => {
                    spawn(&path, self.tx.clone());
                    self.in_flight.insert(path, vec![request]);
                }
            }
            events.push(AssetEvent {
                request,
                state: AssetState::Loading,
            });
        }
        while let Ok(done) = self.rx.try_recv() {
            let Some(interests) = self.in_flight.remove(&done.path) else {
                continue;
            };
            for request in interests {
                if !self.live.contains(&request.scene) {
                    continue;
                }
                let state = match &done.result {
                    Ok(bytes) => AssetState::Ready(bytes.clone()),
                    Err(message) => AssetState::Failed(message.clone()),
                };
                events.push(AssetEvent { request, state });
            }
        }
    }
}

impl Runtime {
    pub fn open_scene(&self) -> SceneId {
        self.0.assets.borrow_mut().open_scene()
    }

    /// Drops the scene's interest in every request, shared or not.
    pub fn close_scene(&self, scene: SceneId) {
        self.0.assets.borrow_mut().close_scene(scene)
    }

    /// The load starts at the next frame boundary; `path` is relative to [`ASSET_DIR`].
    pub fn request_asset(&self, scene: SceneId, path: &str) -> AssetRequest {
        self.0.assets.borrow_mut().request(scene, path)
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn set_asset_base(&self, base: &str) {
        self.0.assets.borrow_mut().base = base.into();
    }

    pub(crate) fn pump_assets(&self, events: &mut Vec<AssetEvent>) {
        let mut assets = self.0.assets.borrow_mut();
        #[cfg(not(target_arch = "wasm32"))]
        assets.pump(events, native::spawn_load);
        #[cfg(target_arch = "wasm32")]
        {
            let base = assets.base.clone();
            assets.pump(events, |path, tx| web::spawn_load(&base, path, tx));
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::{Completion, ASSET_DIR};
    use std::path::Path;
    use std::sync::mpsc::Sender;
    use std::sync::Arc;

    pub(super) fn spawn_load(path: &str, tx: Sender<Completion>) {
        let path = path.to_owned();
        std::thread::spawn(move || {
            let file = Path::new(ASSET_DIR).join(&path);
            let result = std::fs::read(&file)
                .map(Arc::from)
                .map_err(|e| format!("{}: {e}", file.display()));
            // A dropped receiver means the runner is gone.
            let _ = tx.send(Completion { path, result });
        });
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::Completion;
    use std::sync::mpsc::Sender;
    use std::sync::Arc;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    async fn fetch_bytes(url: &str) -> Result<Arc<[u8]>, String> {
        let global = js_sys::global();
        let promise = match global.dyn_ref::<web_sys::WorkerGlobalScope>() {
            Some(scope) => scope.fetch_with_str(url),
            None => web_sys::window()
                .ok_or_else(|| "no global window or worker scope".to_owned())?
                .fetch_with_str(url),
        };
        let response: web_sys::Response = JsFuture::from(promise)
            .await
            .map_err(|e| format!("fetch failed: {e:?}"))?
            .dyn_into()
            .map_err(|_| "fetch returned no Response".to_owned())?;
        if !response.ok() {
            return Err(format!(
                "HTTP {} {}",
                response.status(),
                response.status_text()
            ));
        }
        let buffer = JsFuture::from(
            response
                .array_buffer()
                .map_err(|e| format!("array_buffer: {e:?}"))?,
        )
        .await
        .map_err(|e| format!("body read failed: {e:?}"))?;
        Ok(Arc::from(js_sys::Uint8Array::new(&buffer).to_vec()))
    }

    pub(super) fn spawn_load(base: &str, path: &str, tx: Sender<Completion>) {
        let url = format!("{base}{path}");
        let path = path.to_owned();
        wasm_bindgen_futures::spawn_local(async move {
            let result = fetch_bytes(&url)
                .await
                .map_err(|message| format!("{url}: {message}"));
            let _ = tx.send(Completion { path, result });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_late_asset_result_does_not_revive_a_removed_scene() {
        let mut assets = Assets::default();
        let kept = assets.open_scene();
        let removed = assets.open_scene();
        let shared = assets.request(kept, "tex.png");
        let doomed = assets.request(removed, "tex.png");
        let mut events = Vec::new();
        let mut senders = Vec::new();
        assets.pump(&mut events, |_, tx| senders.push(tx));
        assert_eq!(senders.len(), 1, "one load per shared path");
        assert_eq!(
            events.iter().map(|event| event.request).collect::<Vec<_>>(),
            [shared, doomed]
        );
        assert!(events
            .iter()
            .all(|event| matches!(event.state, AssetState::Loading)));

        assets.close_scene(removed);
        assets.request(removed, "late.png");
        senders[0]
            .send(Completion {
                path: "tex.png".into(),
                result: Ok(Arc::from(&b"px"[..])),
            })
            .unwrap();
        events.clear();
        assets.pump(&mut events, |path, _| {
            panic!("`{path}` started loading for a removed scene")
        });
        assert_eq!(events.len(), 1, "{events:?}");
        assert_eq!(events[0].request, shared);
        assert!(matches!(&events[0].state, AssetState::Ready(bytes) if &**bytes == b"px"));
        assert!(!assets.live.contains(&removed));
    }
}
