use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

use anyhow::{anyhow, Context as _, Result};

use super::scale::scaled_dims;

pub(super) struct ApngWorker {
    tx: Option<SyncSender<ApngFrame>>,
    handle: Option<JoinHandle<()>>,
    frame_count: Arc<AtomicU32>,
}

pub(super) struct ApngFrame {
    pub(super) rgba: Vec<u8>,
    pub(super) src_width: u32,
    pub(super) src_height: u32,
    pub(super) captured_at: Instant,
    pub(super) scale: Option<u32>,
}

// In-flight capacity only; the memory ceiling is the worker's internal Vec.
const APNG_CHANNEL_CAPACITY: usize = 16;

impl ApngWorker {
    pub(super) fn spawn(path: PathBuf) -> Result<Self> {
        let (tx, rx) = sync_channel::<ApngFrame>(APNG_CHANNEL_CAPACITY);
        let frame_count = Arc::new(AtomicU32::new(0));
        let frame_count_for_worker = frame_count.clone();
        let handle = thread::Builder::new()
            .name("loam-app::apng-encoder".into())
            .spawn(move || apng_encoder_loop(path, rx, frame_count_for_worker))
            .context("spawn apng encoder thread")?;
        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
            frame_count,
        })
    }

    pub(super) fn try_send(&self, frame: ApngFrame) {
        let Some(tx) = self.tx.as_ref() else { return };
        if let Err(TrySendError::Disconnected(_)) = tx.try_send(frame) {
            tracing::error!("capture: apng encoder thread exited unexpectedly");
        }
    }

    pub(super) fn frame_count(&self) -> u32 {
        self.frame_count.load(Ordering::Relaxed)
    }

    pub(super) fn detach(mut self) -> JoinHandle<()> {
        self.tx.take();
        self.handle.take().expect("worker handle present")
    }
}

impl Drop for ApngWorker {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn apng_encoder_loop(path: PathBuf, rx: Receiver<ApngFrame>, frame_count: Arc<AtomicU32>) {
    let mut frames: Vec<ApngFrame> = Vec::new();
    for frame in rx {
        frames.push(frame);
        frame_count.store(frames.len() as u32, Ordering::Relaxed);
    }
    if frames.is_empty() {
        tracing::info!("capture: apng stopped before any frames captured; no file written");
        return;
    }
    if let Err(e) = write_apng(&path, frames) {
        tracing::error!("capture: apng write failed: {e:#}");
        return;
    }
    tracing::info!("capture: apng file finalised at {}", path.display());
}

fn write_apng(path: &Path, frames: Vec<ApngFrame>) -> Result<()> {
    let (out_w, out_h) = scaled_dims(frames[0].src_width, frames[0].src_height, frames[0].scale)?;
    let file =
        File::create(path).with_context(|| format!("create apng output {}", path.display()))?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), out_w, out_h);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    // num_plays=0 loops infinitely.
    encoder
        .set_animated(frames.len() as u32, 0)
        .context("apng set_animated")?;
    let mut writer = encoder.write_header().context("apng write_header")?;

    let mut last_captured_at: Option<Instant> = None;
    for frame in frames {
        let (fw, fh) = scaled_dims(frame.src_width, frame.src_height, frame.scale)?;
        if fw != out_w || fh != out_h {
            return Err(anyhow!(
                "apng frame dims {fw}x{fh} != first frame {out_w}x{out_h}"
            ));
        }
        let rgba = if frame.scale.is_some() {
            let src: ::image::RgbaImage =
                ::image::ImageBuffer::from_raw(frame.src_width, frame.src_height, frame.rgba)
                    .ok_or_else(|| {
                        anyhow!(
                            "RGBA size mismatch at {}x{}",
                            frame.src_width,
                            frame.src_height
                        )
                    })?;
            ::image::imageops::resize(&src, out_w, out_h, ::image::imageops::FilterType::Lanczos3)
                .into_raw()
        } else {
            frame.rgba
        };

        // The first frame has no gap; 33 ms is ~30 fps.
        let delay_ms = match last_captured_at {
            None => 33u16,
            Some(prev) => {
                let ms = frame.captured_at.duration_since(prev).as_millis();
                ms.min(u16::MAX as u128).max(1) as u16
            }
        };
        last_captured_at = Some(frame.captured_at);

        writer
            .set_frame_delay(delay_ms, 1000)
            .context("apng set_frame_delay")?;
        writer
            .write_image_data(&rgba)
            .context("apng write_image_data")?;
    }
    writer.finish().context("apng finish")?;
    Ok(())
}
