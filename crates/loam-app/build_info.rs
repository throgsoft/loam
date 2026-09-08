use std::path::{Path, PathBuf};
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output.status.success().then(|| {
        String::from_utf8_lossy(&output.stdout)
            .trim_end()
            .to_owned()
    })
}

fn watch(path: &Path) {
    let mut target = path;
    while !target.exists() {
        let Some(parent) = target.parent() else {
            return;
        };
        target = parent;
    }
    println!("cargo:rerun-if-changed={}", target.display());
}

fn main() {
    let hash = git(&["rev-parse", "--short=8", "HEAD"]).unwrap_or_else(|| "nogit".to_owned());
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
        .is_some_and(|status| !status.is_empty());
    let marker = if dirty { "+dirty" } else { "" };
    println!("cargo:rustc-env=BUILD_HASH={hash}");
    println!("cargo:rustc-env=BUILD_DIRTY={marker}");

    // Worktrees store HEAD and index outside the checkout's .git file.
    for name in ["HEAD", "index", "packed-refs"] {
        if let Some(path) = git(&["rev-parse", "--git-path", name]) {
            watch(Path::new(&path));
        }
    }
    if let Some(branch) = git(&["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git(&["rev-parse", "--git-path", &branch]) {
            watch(Path::new(&path));
        }
    }
    if let Some(root) = git(&["rev-parse", "--show-toplevel"]) {
        if let Some(files) = git(&["ls-files", "-z", "--full-name"]) {
            let root = PathBuf::from(root);
            for file in files.split('\0').filter(|file| !file.is_empty()) {
                watch(&root.join(file));
            }
        }
    } else {
        println!("cargo:rerun-if-changed=src");
    }
}
