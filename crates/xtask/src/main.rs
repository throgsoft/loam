mod serve;
mod web;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage:
  cargo xtask web [--bin <name>] [--package <name>] [--release] [--public-url <path>] [--features <list>]
                  [--manifest-path <Cargo.toml>] [--out-dir <dir>]
    with --manifest-path, --release needs that workspace to declare [profile.wasm-release]
    (copy it from loam's root Cargo.toml)
  cargo xtask serve [--port <port>]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("web") => web::parse(&args[1..]).and_then(|opts| web::run(&opts)),
        Some("serve") => parse_port(&args[1..]).and_then(|port| {
            let root = workspace_root()?;
            serve::run(&root.join("dist"), port)
        }),
        _ => Err(USAGE.to_string()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .ok_or_else(|| "crates/xtask has no workspace root".to_string())
}

fn parse_port(args: &[String]) -> Result<u16, String> {
    match args {
        [] => Ok(8081),
        [flag, port] if flag == "--port" => port.parse().map_err(|e| format!("--port: {e}")),
        _ => Err(USAGE.to_string()),
    }
}
