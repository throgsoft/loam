use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureUnavailable;

impl std::fmt::Display for CaptureUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("capture is unavailable on this host")
    }
}

impl std::error::Error for CaptureUnavailable {}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CaptureStage {
    Pre,
    Post,
    /// Two files per frame; PNG only.
    Both,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CaptureFormat {
    Png,
    Gif,
    /// Buffered in memory until stop.
    Apng,
}

/// NeuQuant (Dekker, 1994) picks 256 colors; the mode controls what it trains on.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum PaletteMode {
    /// Per-frame: consecutive palettes differ, so gradients shimmer.
    #[default]
    Local,
    /// One palette trained on the first `GIF_WARMUP_FRAMES` captures.
    Global,
}

#[derive(Debug)]
pub enum CaptureRequest {
    OneShot {
        stage: CaptureStage,
        dir: Option<PathBuf>,
        name: Option<String>,
    },
    StartSequence {
        format: CaptureFormat,
        stage: CaptureStage,
        dir: Option<PathBuf>,
        name: Option<String>,
        /// `None` = every render frame; GIF and APNG fall back to 30 fps.
        fps: Option<u16>,
        /// Output width, aspect preserved; `None` = native. PNG ignores it.
        scale: Option<u32>,
        /// GIF only.
        palette: PaletteMode,
    },
    Stop,
    Toggle {
        format: CaptureFormat,
        stage: CaptureStage,
        dir: Option<PathBuf>,
        name: Option<String>,
        fps: Option<u16>,
        scale: Option<u32>,
        palette: PaletteMode,
    },
}
