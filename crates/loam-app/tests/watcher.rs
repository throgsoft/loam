#![cfg(not(target_arch = "wasm32"))]

use std::fs;
use std::path::Path;
use std::thread::sleep;
use std::time::{Duration, Instant};

use loam_app::FileWatcher;
use std::path::PathBuf;

fn wait_for<F>(watcher: &FileWatcher, timeout: Duration, mut pred: F) -> PathBuf
where
    F: FnMut(&PathBuf) -> bool,
{
    let deadline = Instant::now() + timeout;
    let mut seen = Vec::new();
    while Instant::now() < deadline {
        for ev in watcher.poll() {
            if pred(&ev) {
                return ev;
            }
            seen.push(ev);
        }
        sleep(Duration::from_millis(25));
    }
    panic!("timeout waiting for event; saw: {seen:?}");
}

fn has_path(target: &Path) -> impl Fn(&PathBuf) -> bool + '_ {
    move |path| {
        path.canonicalize()
            .ok()
            .zip(target.canonicalize().ok())
            .is_some_and(|(actual, expected)| actual == expected)
    }
}

#[test]
fn reports_created_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut watcher = FileWatcher::new().unwrap();
    watcher.watch(dir.path()).unwrap();

    let file = dir.path().join("hello.txt");
    fs::write(&file, b"hi").unwrap();

    wait_for(&watcher, Duration::from_secs(3), has_path(&file));
}

#[test]
fn reports_modified_file() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("shader.wgsl");
    fs::write(&file, b"v1").unwrap();

    let mut watcher = FileWatcher::new().unwrap();
    watcher.watch(dir.path()).unwrap();
    let _ = watcher.poll();

    fs::write(&file, b"v2").unwrap();

    wait_for(&watcher, Duration::from_secs(3), has_path(&file));
}

#[test]
fn removed_file_keeps_its_canonical_path_after_parent_removal() {
    let dir = tempfile::tempdir().unwrap();
    let nested = dir.path().join("shaders");
    fs::create_dir(&nested).unwrap();
    let file = nested.join("doomed.wgsl");
    fs::write(&file, b"shader").unwrap();
    let canonical = file.canonicalize().unwrap();
    let mut watcher = FileWatcher::new().unwrap();
    watcher.watch(dir.path()).unwrap();

    fs::remove_file(&file).unwrap();
    fs::remove_dir(&nested).unwrap();

    wait_for(&watcher, Duration::from_secs(3), |path| path == &canonical);
}
