use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use wgpu::{CommandEncoder, Device};

use super::apng::{ApngFrame, ApngWorker};
use super::error::{create, CaptureError};
use super::gif::{GifFrame, GifWorker};
use super::readback::{record_texture_rgba, TextureReadback};
use super::types::{CaptureFormat, CaptureRequest, CaptureStage, PaletteMode};
use crate::session::Target;

impl CaptureStage {
    fn wants_pre(self) -> bool {
        matches!(self, CaptureStage::Pre | CaptureStage::Both)
    }
    fn wants_post(self) -> bool {
        matches!(self, CaptureStage::Post | CaptureStage::Both)
    }
}

impl CaptureFormat {
    fn default_fps(self) -> Option<u16> {
        match self {
            CaptureFormat::Png => None,
            CaptureFormat::Gif | CaptureFormat::Apng => Some(30),
        }
    }

    fn supports_both_stages(self) -> bool {
        matches!(self, CaptureFormat::Png)
    }
}

const FPS_SLACK_DIVISOR: u32 = 8;

fn interval_elapsed(since_last: Duration, interval: Duration) -> bool {
    since_last + interval / FPS_SLACK_DIVISOR >= interval
}

pub(crate) struct Capture {
    default_dir: PathBuf,
    state: CaptureState,
    pending: Vec<JoinHandle<()>>,
}

enum CaptureState {
    Idle,
    OneShot {
        path_pre: Option<PathBuf>,
        path_post: Option<PathBuf>,
    },
    Sequence {
        stage: CaptureStage,
        writer: SequenceWriter,
        fps_interval: Option<Duration>,
        last_capture_time: Option<Instant>,
        frame_count: u32,
    },
}

enum SequenceWriter {
    Png {
        dir: PathBuf,
    },
    Gif {
        worker: GifWorker,
        path: PathBuf,
        default_delay_cs: u16,
        scale: Option<u32>,
        palette_mode: PaletteMode,
        warming: Option<WarmingState>,
        global_palette: Option<Arc<color_quant::NeuQuant>>,
    },
    Apng {
        worker: ApngWorker,
        path: PathBuf,
        scale: Option<u32>,
    },
}

struct WarmingState {
    buffer: Vec<WarmupFrame>,
}

struct WarmupFrame {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    captured_at: Instant,
}

// ~1 s at 30 fps; ~57 MB at 800x600, released after training.
const GIF_WARMUP_FRAMES: u32 = 30;

impl Capture {
    pub(crate) fn new() -> Self {
        Self {
            default_dir: PathBuf::from("captures"),
            state: CaptureState::Idle,
            pending: Vec::new(),
        }
    }

    fn reap_finished(&mut self) {
        let mut still_running = Vec::with_capacity(self.pending.len());
        for h in self.pending.drain(..) {
            if h.is_finished() {
                let _ = h.join();
            } else {
                still_running.push(h);
            }
        }
        self.pending = still_running;
    }

    fn apply_requests(&mut self, requests: Vec<CaptureRequest>) -> Vec<String> {
        let mut log = Vec::new();
        for req in requests {
            match req {
                CaptureRequest::OneShot { stage, dir, name } => {
                    let dir = dir.unwrap_or_else(|| self.default_dir.clone());
                    let name = name.unwrap_or_else(default_name);
                    let path_pre = stage
                        .wants_pre()
                        .then(|| dir.join(format!("{name}_pre.png")));
                    let path_post = stage
                        .wants_post()
                        .then(|| dir.join(format!("{name}_post.png")));
                    self.state = CaptureState::OneShot {
                        path_pre,
                        path_post,
                    };
                    log.push(format!("capture: one-shot queued ({stage:?})"));
                }
                CaptureRequest::StartSequence {
                    format,
                    stage,
                    dir,
                    name,
                    fps,
                    scale,
                    palette,
                } => match self.start_sequence(format, stage, dir, name, fps, scale, palette) {
                    Ok(msg) => log.push(msg),
                    Err(e) => log.push(format!("capture: failed to start sequence: {e}")),
                },
                CaptureRequest::Stop => self.stop(&mut log),
                CaptureRequest::Toggle {
                    format,
                    stage,
                    dir,
                    name,
                    fps,
                    scale,
                    palette,
                } => {
                    if matches!(self.state, CaptureState::Sequence { .. }) {
                        self.stop(&mut log);
                    } else {
                        match self.start_sequence(format, stage, dir, name, fps, scale, palette) {
                            Ok(msg) => log.push(msg),
                            Err(e) => log.push(format!("capture: failed to start sequence: {e}")),
                        }
                    }
                }
            }
        }
        log
    }

    #[allow(clippy::too_many_arguments)]
    fn start_sequence(
        &mut self,
        format: CaptureFormat,
        mut stage: CaptureStage,
        dir: Option<PathBuf>,
        name: Option<String>,
        fps: Option<u16>,
        scale: Option<u32>,
        palette: PaletteMode,
    ) -> Result<String, CaptureError> {
        let dir = dir.unwrap_or_else(|| self.default_dir.clone());
        let name = name.unwrap_or_else(default_name);
        let fps = fps.or_else(|| format.default_fps());

        if !format.supports_both_stages() && stage == CaptureStage::Both {
            stage = CaptureStage::Post;
        }

        let writer = match format {
            CaptureFormat::Png => {
                let dir = dir.join(&name);
                std::fs::create_dir_all(&dir).map_err(create("png sequence dir", dir.clone()))?;
                SequenceWriter::Png { dir }
            }
            CaptureFormat::Gif => {
                let path = dir.join(format!("{name}.gif"));
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(create("gif parent dir", parent.to_path_buf()))?;
                }
                let default_delay_cs = fps_to_centiseconds(fps.unwrap_or(30));
                let worker = GifWorker::spawn(path.clone())?;
                let warming = match palette {
                    PaletteMode::Local => None,
                    PaletteMode::Global => Some(WarmingState {
                        buffer: Vec::with_capacity(GIF_WARMUP_FRAMES as usize),
                    }),
                };
                SequenceWriter::Gif {
                    worker,
                    path,
                    default_delay_cs,
                    scale,
                    palette_mode: palette,
                    warming,
                    global_palette: None,
                }
            }
            CaptureFormat::Apng => {
                let path = dir.join(format!("{name}.apng"));
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(create("apng parent dir", parent.to_path_buf()))?;
                }
                let worker = ApngWorker::spawn(path.clone())?;
                SequenceWriter::Apng {
                    worker,
                    path,
                    scale,
                }
            }
        };

        let fps_interval = fps.map(|f| Duration::from_secs_f64(1.0 / f.max(1) as f64));
        self.state = CaptureState::Sequence {
            stage,
            writer,
            fps_interval,
            last_capture_time: None,
            frame_count: 0,
        };
        Ok(format!(
            "capture: sequence started ({format:?}, {stage:?}, fps={}, palette={palette:?})",
            fps.map(|f| f.to_string())
                .unwrap_or_else(|| "unlimited".into())
        ))
    }

    fn stop(&mut self, log: &mut Vec<String>) {
        self.reap_finished();
        let state = std::mem::replace(&mut self.state, CaptureState::Idle);
        match state {
            CaptureState::Sequence {
                writer,
                frame_count,
                ..
            } => match writer {
                SequenceWriter::Png { dir } => {
                    log.push(format!(
                        "capture: PNG sequence stopped, {frame_count} frame(s) at {}",
                        dir.display()
                    ));
                }
                SequenceWriter::Gif {
                    worker,
                    path,
                    default_delay_cs,
                    scale,
                    warming,
                    ..
                } => {
                    if let Some(mut w) = warming {
                        if !w.buffer.is_empty() {
                            tracing::info!(
                                "capture: stopped during warmup ({} frame(s)); \
                                 training palette on partial buffer",
                                w.buffer.len()
                            );
                            let nq = Arc::new(train_global_palette(&w.buffer));
                            for f in w.buffer.drain(..) {
                                worker.try_send(GifFrame {
                                    rgba: f.rgba,
                                    src_width: f.width,
                                    src_height: f.height,
                                    captured_at: f.captured_at,
                                    default_delay_cs,
                                    scale,
                                    global_palette: Some(nq.clone()),
                                });
                            }
                        }
                    }
                    let dropped = worker.dropped();
                    let handle = worker.detach();
                    self.pending.push(handle);
                    let drop_note = if dropped > 0 {
                        format!(", {dropped} dropped under backpressure")
                    } else {
                        String::new()
                    };
                    log.push(format!(
                        "capture: GIF stream stopped, {frame_count} frame(s){drop_note} \
                         encoding in background -> {}",
                        path.display()
                    ));
                }
                SequenceWriter::Apng { worker, path, .. } => {
                    let buffered = worker.frame_count();
                    let handle = worker.detach();
                    self.pending.push(handle);
                    log.push(format!(
                        "capture: APNG stream stopped, {buffered} frame(s) buffered; \
                         assembling and writing in background -> {}",
                        path.display()
                    ));
                }
            },
            CaptureState::OneShot { .. } => {
                log.push("capture: pending one-shot cancelled".into());
            }
            CaptureState::Idle => {
                log.push("capture: stop with no active session (no-op)".into());
            }
        }
    }

    fn wants_pre(&self) -> bool {
        match &self.state {
            CaptureState::Idle => false,
            CaptureState::OneShot { path_pre, .. } => path_pre.is_some(),
            CaptureState::Sequence { stage, .. } => stage.wants_pre(),
        }
    }

    fn wants_post(&self) -> bool {
        match &self.state {
            CaptureState::Idle => false,
            CaptureState::OneShot { path_post, .. } => path_post.is_some(),
            CaptureState::Sequence { stage, .. } => stage.wants_post(),
        }
    }

    fn should_capture(&self, now: Instant) -> bool {
        match &self.state {
            CaptureState::Idle => false,
            CaptureState::OneShot { .. } => true,
            CaptureState::Sequence {
                fps_interval,
                last_capture_time,
                ..
            } => match (fps_interval, last_capture_time) {
                (None, _) => true,
                (Some(_), None) => true,
                (Some(interval), Some(last)) => {
                    interval_elapsed(now.duration_since(*last), *interval)
                }
            },
        }
    }

    fn consume_frame(
        &mut self,
        is_pre: bool,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        captured_at: Instant,
    ) -> Result<(), CaptureError> {
        match &mut self.state {
            CaptureState::Idle => Ok(()),
            CaptureState::OneShot {
                path_pre,
                path_post,
            } => {
                let path = if is_pre {
                    path_pre.take()
                } else {
                    path_post.take()
                };
                if let Some(path) = path {
                    write_png_bytes(&path, &rgba, width, height)?;
                    tracing::info!("capture: wrote {}", path.display());
                }
                Ok(())
            }
            CaptureState::Sequence {
                writer,
                frame_count,
                ..
            } => writer.write_frame(is_pre, *frame_count, rgba, width, height, captured_at),
        }
    }

    fn join_pending(&mut self) {
        for h in self.pending.drain(..) {
            let _ = h.join();
        }
    }

    fn advance_frame(&mut self, now: Instant) {
        match &mut self.state {
            CaptureState::OneShot {
                path_pre,
                path_post,
            } => {
                if path_pre.is_none() && path_post.is_none() {
                    self.state = CaptureState::Idle;
                }
            }
            CaptureState::Sequence {
                last_capture_time,
                frame_count,
                ..
            } => {
                *last_capture_time = Some(now);
                *frame_count = frame_count.saturating_add(1);
            }
            CaptureState::Idle => {}
        }
    }
}

/// The copies one drawn frame owes the capture.
pub(crate) struct Shot {
    pre: Slot,
    post: Slot,
}

enum Slot {
    Skipped,
    Wanted,
    Recorded(TextureReadback),
}

impl Slot {
    fn new(wanted: bool) -> Self {
        if wanted {
            Slot::Wanted
        } else {
            Slot::Skipped
        }
    }
}

impl Shot {
    pub(crate) fn wants(&self, pre: bool) -> bool {
        let copy = if pre { &self.pre } else { &self.post };
        matches!(copy, Slot::Wanted)
    }
}

impl Capture {
    /// True when this frame copies anything, counting requests not yet applied.
    pub(crate) fn pending(&self, requests: &[CaptureRequest], now: Instant) -> bool {
        !requests.is_empty() || self.should_capture(now)
    }

    /// Drains `requests` into the capture state.
    pub(crate) fn begin(&mut self, requests: &mut Vec<CaptureRequest>, now: Instant) -> Shot {
        if !requests.is_empty() {
            for line in self.apply_requests(std::mem::take(requests)) {
                tracing::info!("{line}");
            }
        }
        let taken = self.should_capture(now);
        Shot {
            pre: Slot::new(taken && self.wants_pre()),
            post: Slot::new(taken && self.wants_post()),
        }
    }

    pub(crate) fn record(
        &self,
        shot: &mut Shot,
        pre: bool,
        device: &Device,
        encoder: &mut CommandEncoder,
        target: &Target<'_>,
    ) {
        let copy = if pre { &mut shot.pre } else { &mut shot.post };
        if !matches!(copy, Slot::Wanted) {
            return;
        }
        *copy = match record_texture_rgba(
            device,
            encoder,
            target.texture,
            target.size.0,
            target.size.1,
            target.format,
        ) {
            Ok(readback) => Slot::Recorded(readback),
            Err(error) => {
                tracing::error!("capture: copy failed: {error}");
                Slot::Skipped
            }
        };
    }

    /// Reads back the recorded copies, so it runs after the frame's submit.
    pub(crate) fn finish(&mut self, shot: Shot, device: &Device, now: Instant) {
        for (is_pre, copy) in [(true, shot.pre), (false, shot.post)] {
            let Slot::Recorded(readback) = copy else {
                continue;
            };
            match readback.read(device) {
                Ok(image) => {
                    if let Err(error) =
                        self.consume_frame(is_pre, image.rgba, image.width, image.height, now)
                    {
                        tracing::error!("capture: write failed: {error}");
                    }
                }
                Err(error) => tracing::error!("capture: readback failed: {error}"),
            }
        }
        self.advance_frame(now);
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stop(&mut Vec::new());
        self.join_pending();
    }
}

impl SequenceWriter {
    fn write_frame(
        &mut self,
        is_pre: bool,
        frame_idx: u32,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        captured_at: Instant,
    ) -> Result<(), CaptureError> {
        match self {
            SequenceWriter::Png { dir } => {
                let label = if is_pre { "pre" } else { "post" };
                let path = dir.join(format!("{label}_{frame_idx:06}.png"));
                write_png_bytes(&path, &rgba, width, height)?;
                Ok(())
            }
            SequenceWriter::Gif {
                worker,
                default_delay_cs,
                scale,
                palette_mode,
                warming,
                global_palette,
                ..
            } => {
                match *palette_mode {
                    PaletteMode::Local => {
                        worker.try_send(GifFrame {
                            rgba,
                            src_width: width,
                            src_height: height,
                            captured_at,
                            default_delay_cs: *default_delay_cs,
                            scale: *scale,
                            global_palette: None,
                        });
                    }
                    PaletteMode::Global => {
                        if let Some(w) = warming.as_mut() {
                            w.buffer.push(WarmupFrame {
                                rgba,
                                width,
                                height,
                                captured_at,
                            });
                            if w.buffer.len() as u32 >= GIF_WARMUP_FRAMES {
                                let nq = Arc::new(train_global_palette(&w.buffer));
                                tracing::info!(
                                    "capture: gif global palette trained from {} frames",
                                    w.buffer.len()
                                );
                                for f in w.buffer.drain(..) {
                                    worker.try_send(GifFrame {
                                        rgba: f.rgba,
                                        src_width: f.width,
                                        src_height: f.height,
                                        captured_at: f.captured_at,
                                        default_delay_cs: *default_delay_cs,
                                        scale: *scale,
                                        global_palette: Some(nq.clone()),
                                    });
                                }
                                *global_palette = Some(nq);
                                *warming = None;
                            }
                        } else if let Some(nq) = global_palette.as_ref() {
                            worker.try_send(GifFrame {
                                rgba,
                                src_width: width,
                                src_height: height,
                                captured_at,
                                default_delay_cs: *default_delay_cs,
                                scale: *scale,
                                global_palette: Some(nq.clone()),
                            });
                        } else {
                            tracing::error!(
                                "capture: gif global mode lost palette state (frame dropped)"
                            );
                        }
                    }
                }
                Ok(())
            }
            SequenceWriter::Apng { worker, scale, .. } => {
                worker.try_send(ApngFrame {
                    rgba,
                    src_width: width,
                    src_height: height,
                    captured_at,
                    scale: *scale,
                });
                Ok(())
            }
        }
    }
}

fn train_global_palette(buffer: &[WarmupFrame]) -> color_quant::NeuQuant {
    // Stride 16 feeds ~110 K samples at 800x600 over 30 frames.
    const STRIDE: usize = 16;
    let mut samples: Vec<u8> = Vec::new();
    for frame in buffer {
        for px in frame.rgba.chunks_exact(4).step_by(STRIDE) {
            // Opaque alpha is normalized so NeuQuant's 4D metric is not biased.
            samples.push(px[0]);
            samples.push(px[1]);
            samples.push(px[2]);
            samples.push(if px[3] != 0 { 0xFF } else { 0 });
        }
    }
    color_quant::NeuQuant::new(10, 256, &samples)
}

fn fps_to_centiseconds(fps: u16) -> u16 {
    // A zero-delay GIF frame is rejected.
    let cs = (100.0_f32 / fps.max(1) as f32).round() as u16;
    cs.max(1)
}

fn default_name() -> String {
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("capture_{unix}")
}

fn write_png_bytes(path: &Path, rgba: &[u8], width: u32, height: u32) -> Result<(), CaptureError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(create("capture dir", parent.to_path_buf()))?;
    }
    ::image::save_buffer_with_format(
        path,
        rgba,
        width,
        height,
        ::image::ColorType::Rgba8,
        ::image::ImageFormat::Png,
    )
    .map_err(|error| CaptureError::Image {
        path: path.to_path_buf(),
        error,
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn early_frame() {
        let interval = Duration::from_micros(33_333);
        assert!(interval_elapsed(interval, interval));
        assert!(interval_elapsed(
            interval - Duration::from_micros(200),
            interval
        ));
        assert!(interval_elapsed(interval * 2, interval));

        assert!(!interval_elapsed(interval / 2, interval));
        assert!(!interval_elapsed(Duration::ZERO, interval));
    }
}
