use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub struct Options {
    pub bin: String,
    pub package: Option<String>,
    pub release: bool,
    pub public_url: String,
    pub features: Vec<String>,
    pub manifest_path: Option<PathBuf>,
    pub out_dir: Option<PathBuf>,
}

impl Options {
    fn package(&self) -> &str {
        self.package.as_deref().unwrap_or(&self.bin)
    }
}

pub fn parse(args: &[String]) -> Result<Options, String> {
    let mut opts = Options {
        bin: "polytope_playground".to_string(),
        package: None,
        release: false,
        public_url: "/".to_string(),
        features: Vec::new(),
        manifest_path: None,
        out_dir: None,
    };
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--release" => opts.release = true,
            "--bin" => opts.bin = args.next().cloned().ok_or("--bin needs a name")?,
            "--package" => {
                opts.package = Some(args.next().cloned().ok_or("--package needs a name")?);
            }
            "--public-url" => {
                opts.public_url = args.next().cloned().ok_or("--public-url needs a path")?;
            }
            "--features" => {
                let list = args.next().cloned().ok_or("--features needs a list")?;
                opts.features.extend(
                    list.split(',')
                        .filter(|name| !name.is_empty())
                        .map(String::from),
                );
            }
            "--manifest-path" => {
                let path = args.next().ok_or("--manifest-path needs a Cargo.toml")?;
                opts.manifest_path = Some(PathBuf::from(path));
            }
            "--out-dir" => {
                let dir = args.next().ok_or("--out-dir needs a directory")?;
                opts.out_dir = Some(PathBuf::from(dir));
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    if !opts.public_url.ends_with('/') {
        opts.public_url.push('/');
    }
    Ok(opts)
}

pub fn run(opts: &Options) -> Result<(), String> {
    let root = crate::workspace_root()?;
    let manifest = opts
        .manifest_path
        .as_deref()
        .map(std::path::absolute)
        .transpose()
        .map_err(|e| format!("--manifest-path: {e}"))?;
    let manifest_dir = match &manifest {
        Some(path) => path
            .parent()
            .ok_or_else(|| format!("{} has no directory", path.display()))?
            .to_path_buf(),
        None => root.clone(),
    };

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let mut command = Command::new(cargo);
    command
        .current_dir(&manifest_dir)
        .args(["build", "--target", "wasm32-unknown-unknown"])
        .args(["-p", opts.package(), "--bin", &opts.bin])
        .arg("--no-default-features")
        .arg("--message-format=json-render-diagnostics")
        .stdout(Stdio::piped());
    if let Some(manifest) = &manifest {
        command.arg("--manifest-path").arg(manifest);
    }
    if opts.release {
        command.args(["--profile", "wasm-release"]);
        remap_paths(&mut command, &root)?;
    }
    for feature in &opts.features {
        if let Some((package, _)) = feature.split_once('/') {
            command.args(["-p", package]);
        }
        command.args(["--features", feature]);
    }
    let mut child = command.spawn().map_err(|e| format!("run cargo: {e}"))?;
    let stdout = child.stdout.take().ok_or("cargo stdout is not piped")?;
    let wasm = wasm_artifact(std::io::BufReader::new(stdout), &opts.bin);
    let status = child.wait().map_err(|e| format!("wait for cargo: {e}"))?;
    if !status.success() {
        return Err(format!("cargo build failed: {status}"));
    }
    let wasm = wasm?;

    let out_dir = opts
        .out_dir
        .clone()
        .unwrap_or_else(|| root.join("dist").join(&opts.bin));
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("create {}: {e}", out_dir.display()))?;

    wasm_bindgen_cli_support::Bindgen::new()
        .input_path(&wasm)
        .out_name(&opts.bin)
        .web(true)
        .map_err(|e| format!("wasm-bindgen: {e:#}"))?
        .typescript(false)
        .omit_default_module_path(false)
        .remove_name_section(opts.release)
        .generate(&out_dir)
        .map_err(|e| format!("wasm-bindgen: {e:#}"))?;

    optimize(&out_dir.join(format!("{}_bg.wasm", opts.bin)), opts.release)?;

    let template = if manifest.is_some() {
        let own = manifest_dir.join("index.html");
        own.exists().then_some(own)
    } else {
        let package = root.join("crates").join(opts.package());
        let session = root.join("crates/loam-app/static/session_page.html");
        let own = [
            package.join(format!("{}.html", opts.bin)),
            package.join("index.html"),
        ];
        Some(
            own.into_iter()
                .find(|path| path.exists())
                .unwrap_or(session),
        )
    };
    if let Some(template) = template {
        let dest = out_dir.join("index.html");
        let same = std::fs::canonicalize(&dest)
            .is_ok_and(|d| std::fs::canonicalize(&template).is_ok_and(|t| t == d));
        if same {
            return Err(format!(
                "--out-dir would overwrite the template {}",
                template.display()
            ));
        }
        let page = render_page(&root, &manifest_dir, &template, opts)?;
        std::fs::write(dest, page).map_err(|e| format!("write index.html: {e}"))?;
    } else {
        println!(
            "index.html skipped: no {}",
            manifest_dir.join("index.html").display()
        );
    }

    let entries =
        std::fs::read_dir(&out_dir).map_err(|e| format!("read {}: {e}", out_dir.display()))?;
    for entry in entries.flatten() {
        let size = entry.metadata().map_or(0, |meta| meta.len());
        println!("{} ({size} bytes)", entry.path().display());
    }
    Ok(())
}

// Shipped wasm must not carry build-machine paths; rustc applies the last matching remap.
fn remap_paths(command: &mut Command, root: &Path) -> Result<(), String> {
    let home = std::env::home_dir();
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| home.as_ref().map(|home| home.join(".cargo")));
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let sysroot = Command::new(rustc)
        .args(["--print", "sysroot"])
        .output()
        .map_err(|e| format!("rustc --print sysroot: {e}"))?;
    let sysroot = PathBuf::from(String::from_utf8_lossy(&sysroot.stdout).trim());
    let flags: Vec<String> = [
        (home, "~"),
        (cargo_home, "cargo"),
        (Some(sysroot), "rust"),
        (Some(root.to_path_buf()), "loam"),
    ]
    .into_iter()
    .filter_map(|(from, to)| {
        from.map(|from| format!("--remap-path-prefix={}={to}", from.display()))
    })
    .collect();
    // Cargo ignores config rustflags whenever the environment sets any.
    let set = std::env::var("CARGO_ENCODED_RUSTFLAGS").ok().or_else(|| {
        std::env::var("RUSTFLAGS")
            .ok()
            .map(|flags| flags.split_whitespace().collect::<Vec<_>>().join("\u{1f}"))
    });
    match set {
        Some(set) => {
            let all: Vec<String> = std::iter::once(set)
                .filter(|set| !set.is_empty())
                .chain(flags)
                .collect();
            command.env("CARGO_ENCODED_RUSTFLAGS", all.join("\u{1f}"));
        }
        None => {
            let array = serde_json::to_string(&flags).map_err(|e| format!("remap flags: {e}"))?;
            command
                .arg("--config")
                .arg(format!("target.wasm32-unknown-unknown.rustflags={array}"));
        }
    }
    Ok(())
}

fn wasm_artifact(messages: impl BufRead, bin: &str) -> Result<PathBuf, String> {
    let mut found = None;
    for line in messages.lines() {
        let line = line.map_err(|e| format!("read cargo output: {e}"))?;
        let Ok(message) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        if message["reason"] != "compiler-artifact" {
            continue;
        }
        let target = &message["target"];
        let is_bin = target["kind"]
            .as_array()
            .is_some_and(|kinds| kinds.iter().any(|kind| kind == "bin"));
        if !is_bin || target["name"] != bin {
            continue;
        }
        if let Some(path) = message["executable"].as_str() {
            found = Some(PathBuf::from(path));
        }
    }
    found.ok_or_else(|| format!("cargo built no bin named {bin}"))
}

fn optimize(wasm: &Path, release: bool) -> Result<(), String> {
    if !release {
        println!("wasm-opt skipped: not a --release build");
        return Ok(());
    }
    let optimized = wasm.with_extension("opt.wasm");
    let result = Command::new("wasm-opt")
        .args(["-Oz", "-o"])
        .arg(&optimized)
        .arg(wasm)
        .status();
    match result {
        Ok(status) if status.success() => std::fs::rename(&optimized, wasm)
            .map_err(|e| format!("replace {}: {e}", wasm.display())),
        Ok(status) => Err(format!("wasm-opt failed: {status}")),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!("wasm-opt skipped: not on PATH");
            Ok(())
        }
        Err(e) => Err(format!("run wasm-opt: {e}")),
    }
}

fn render_page(
    root: &Path,
    manifest_dir: &Path,
    template: &Path,
    opts: &Options,
) -> Result<String, String> {
    let read = |path: &Path| {
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))
    };
    let template = read(template)?;
    let loader = read(&root.join("crates/loam-app/static/page_loader.html"))?;
    let hash = git_hash(manifest_dir);
    Ok(template
        .replace("{{LOAM_PAGE_LOADER}}", &loader)
        .replace("{{LOAM_PUBLIC_URL}}", &opts.public_url)
        .replace("{{LOAM_BIN}}", &opts.bin)
        .replace("{{LOAM_HASH}}", &hash))
}

fn git_hash(dir: &Path) -> String {
    let git = |args: &[&str]| -> Option<String> {
        let out = Command::new("git")
            .current_dir(dir)
            .args(args)
            .output()
            .ok()
            .filter(|out| out.status.success())?;
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let Some(hash) = git(&["rev-parse", "--short", "HEAD"]) else {
        return "nogit".to_string();
    };
    let dirty = git(&["status", "--porcelain"]).is_some_and(|status| !status.is_empty());
    if dirty {
        format!("{hash}-dirty")
    } else {
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::wasm_artifact;

    #[test]
    fn wasm_artifact_ignores_other_units() {
        let unit = |name: &str, kind: &str, executable: serde_json::Value| {
            serde_json::json!({
                "reason": "compiler-artifact",
                "target": { "name": name, "kind": [kind] },
                "executable": executable,
            })
            .to_string()
        };
        let lines = [
            r#"{"reason":"build-script-executed","package_id":"examples"}"#.to_string(),
            unit(
                "build-script-build",
                "custom-build",
                "t/build/examples/build-script".into(),
            ),
            unit("loam_app", "lib", serde_json::Value::Null),
            unit("hero", "bin", "t/wasm-release/hero.wasm".into()),
            unit("backdrop", "bin", "t/wasm-release/backdrop.wasm".into()),
            r#"{"reason":"build-finished","success":true}"#.to_string(),
        ];
        let wasm = wasm_artifact(lines.join("\n").as_bytes(), "backdrop").unwrap();
        assert_eq!(wasm, std::path::Path::new("t/wasm-release/backdrop.wasm"));
    }
}
