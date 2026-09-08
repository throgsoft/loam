use std::path::PathBuf;

use anyhow::{bail, Result};
use loam_app::args::Args;
use loam_app::shell::{SceneEntry, SceneRegistry};
use loam_app::RunConfig;
use winit::window::WindowAttributes;

mod scene;

use scene::RecordRequest;

fn record_request(args: &Args) -> Result<Option<RecordRequest>> {
    let request = if args.has_bare_flag("record") {
        Some(RecordRequest { dir: None })
    } else {
        args.get("record").map(|dir| RecordRequest {
            dir: Some(PathBuf::from(dir)),
        })
    };
    if request.is_some() && !cfg!(all(feature = "capture", not(target_arch = "wasm32"))) {
        bail!("recording requires a native build with the `capture` feature");
    }
    Ok(request)
}

struct Hero;

impl SceneRegistry for Hero {
    const SCENES: &'static [SceneEntry] = &[SceneEntry {
        slug: "hero",
        label: "LOAM",
        build: |ctx, control| {
            Ok(Box::new(scene::HeroScene::new(
                ctx,
                control,
                record_request(&Args::current())?,
            )?))
        },
    }];
}

fn main() -> Result<()> {
    loam_app::run::<loam_app::shell::SceneShell<Hero>>(RunConfig {
        window: WindowAttributes::default()
            .with_title("loam")
            .with_visible(false),
        ..RunConfig::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(not(feature = "capture"), target_arch = "wasm32"))]
    #[test]
    fn recording_requires_capture_support() {
        assert!(record_request(&Args::from_argv(["--record"])).is_err());
        assert!(record_request(&Args::default()).unwrap().is_none());
    }

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    #[test]
    fn record_takes_a_directory_or_the_shell_default_and_is_off_otherwise() {
        let ask = |argv: [&str; 2]| record_request(&Args::from_argv(argv)).expect("capture is on");

        assert!(ask(["--seed=7", "--fps=60"]).is_none());
        assert_eq!(
            ask(["--record", "--seed=7"]).expect("bare records").dir,
            None
        );
        assert_eq!(
            ask(["--record=out/hero", "--seed=7"])
                .expect("named records")
                .dir,
            Some(PathBuf::from("out/hero"))
        );

        assert_eq!(
            ask(["--record", "out/hero"]).expect("detached records").dir,
            None
        );
    }
}
