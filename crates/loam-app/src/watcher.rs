#[cfg(not(target_arch = "wasm32"))]
mod native {
    use anyhow::{Context, Result};
    use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};
    use std::sync::mpsc::{channel, Receiver};

    // Windows canonical paths keep their extended prefix after a watched file is removed.
    fn canonical_event_path(path: PathBuf) -> PathBuf {
        path.ancestors()
            .find_map(|ancestor| {
                let mut resolved = ancestor.canonicalize().ok()?;
                let suffix = path.strip_prefix(ancestor).ok()?;
                if !suffix.as_os_str().is_empty() {
                    resolved.push(suffix);
                }
                Some(resolved)
            })
            .unwrap_or(path)
    }

    /// [`poll`](Self::poll) collapses a save burst to one event per path.
    pub struct FileWatcher {
        watcher: RecommendedWatcher,
        rx: Receiver<notify::Result<notify::Event>>,
    }

    impl FileWatcher {
        pub fn new() -> Result<Self> {
            let (tx, rx) = channel();
            let watcher = notify::recommended_watcher(move |res| {
                // A dropped receiver means the app is shutting down.
                let _ = tx.send(res);
            })
            .context("creating notify watcher")?;
            Ok(Self { watcher, rx })
        }

        /// Watches the directory recursively.
        pub fn watch(&mut self, path: impl AsRef<Path>) -> Result<()> {
            let path = path.as_ref();
            self.watcher
                .watch(path, RecursiveMode::Recursive)
                .with_context(|| format!("watching {}", path.display()))?;
            Ok(())
        }

        pub fn unwatch(&mut self, path: impl AsRef<Path>) -> Result<()> {
            let path = path.as_ref();
            self.watcher
                .unwatch(path)
                .with_context(|| format!("unwatching {}", path.display()))?;
            Ok(())
        }

        pub fn poll(&self) -> Vec<PathBuf> {
            let mut changed = HashSet::new();
            while let Ok(result) = self.rx.try_recv() {
                match result {
                    Ok(event)
                        if matches!(
                            event.kind,
                            EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
                        ) =>
                    {
                        changed.extend(event.paths.into_iter().map(canonical_event_path));
                    }
                    Ok(_) => {}
                    Err(error) => tracing::warn!(%error, "file watcher error"),
                }
            }
            changed.into_iter().collect()
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use anyhow::Result;
    use std::path::Path;
    use std::path::PathBuf;

    /// Browsers cannot watch local files.
    pub struct FileWatcher {
        _private: (),
    }

    impl FileWatcher {
        pub fn new() -> Result<Self> {
            Ok(Self { _private: () })
        }

        pub fn watch(&mut self, _path: impl AsRef<Path>) -> Result<()> {
            Ok(())
        }

        pub fn unwatch(&mut self, _path: impl AsRef<Path>) -> Result<()> {
            Ok(())
        }

        pub fn poll(&self) -> Vec<PathBuf> {
            Vec::new()
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::FileWatcher;
#[cfg(target_arch = "wasm32")]
pub use web::FileWatcher;
