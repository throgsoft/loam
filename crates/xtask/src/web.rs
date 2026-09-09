use std::path::{Path, PathBuf};
use std::process::Command;

pub struct Options {
    pub bin: String,
    pub package: Option<String>,
    pub release: bool,
    pub public_url: String,
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
    let profile = if opts.release { "release" } else { "debug" };
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let status = Command::new(cargo)
        .current_dir(&root)
        .args(["build", "--target", "wasm32-unknown-unknown"])
        .args(["-p", opts.package(), "--bin", &opts.bin])
        .arg("--no-default-features")
        .args(opts.release.then_some("--release"))
        .status()
        .map_err(|e| format!("run cargo: {e}"))?;
    if !status.success() {
        return Err(format!("cargo build failed: {status}"));
    }

    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map_or_else(|| root.join("target"), |dir| root.join(dir));
    let wasm = target_dir
        .join("wasm32-unknown-unknown")
        .join(profile)
        .join(format!("{}.wasm", opts.bin));
    let dist = root.join("dist").join(&opts.bin);
    std::fs::create_dir_all(&dist).map_err(|e| format!("create {}: {e}", dist.display()))?;

    wasm_bindgen_cli_support::Bindgen::new()
        .input_path(&wasm)
        .out_name(&opts.bin)
        .web(true)
        .map_err(|e| format!("wasm-bindgen: {e:#}"))?
        .typescript(false)
        .omit_default_module_path(false)
        .generate(&dist)
        .map_err(|e| format!("wasm-bindgen: {e:#}"))?;

    optimize(&dist.join(format!("{}_bg.wasm", opts.bin)), opts.release)?;

    let page = render_page(&root, opts)?;
    std::fs::write(dist.join("index.html"), page).map_err(|e| format!("write index.html: {e}"))?;

    let entries = std::fs::read_dir(&dist).map_err(|e| format!("read {}: {e}", dist.display()))?;
    for entry in entries.flatten() {
        let size = entry.metadata().map_or(0, |meta| meta.len());
        println!("{} ({size} bytes)", entry.path().display());
    }
    Ok(())
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

fn render_page(root: &Path, opts: &Options) -> Result<String, String> {
    let read = |path: PathBuf| {
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))
    };
    let own = root.join("crates").join(opts.package()).join("index.html");
    let template = read(if own.exists() {
        own
    } else {
        root.join("crates/loam-app/static/session_page.html")
    })?;
    let loader = read(root.join("crates/loam-app/static/page_loader.html"))?;
    let hash = git_hash(root)?;
    Ok(template
        .replace("{{LOAM_PAGE_LOADER}}", &loader)
        .replace("{{LOAM_PUBLIC_URL}}", &opts.public_url)
        .replace("{{LOAM_BIN}}", &opts.bin)
        .replace("{{LOAM_HASH}}", &hash))
}

fn git_hash(root: &Path) -> Result<String, String> {
    let git = |args: &[&str]| -> Result<String, String> {
        let out = Command::new("git")
            .current_dir(root)
            .args(args)
            .output()
            .map_err(|e| format!("run git: {e}"))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(format!("git {} failed: {}", args.join(" "), stderr.trim()));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let hash = git(&["rev-parse", "--short", "HEAD"])?;
    let dirty = !git(&["status", "--porcelain"])?.is_empty();
    Ok(if dirty { format!("{hash}-dirty") } else { hash })
}
