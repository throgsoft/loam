use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

use super::error::{create, CaptureError};
use super::scale::scaled_dims;

pub(super) struct GifWorker {
    tx: Option<SyncSender<GifFrame>>,
    handle: Option<JoinHandle<()>>,
    dropped: Arc<AtomicU32>,
}

pub(super) struct GifFrame {
    pub(super) rgba: Vec<u8>,
    pub(super) src_width: u32,
    pub(super) src_height: u32,
    pub(super) captured_at: Instant,
    pub(super) default_delay_cs: u16,
    pub(super) scale: Option<u32>,
    pub(super) global_palette: Option<Arc<color_quant::NeuQuant>>,
}

// Bounds worker lag to ~270 ms at 30 fps.
const GIF_CHANNEL_CAPACITY: usize = 8;

impl GifWorker {
    pub(super) fn spawn(path: PathBuf) -> Result<Self, CaptureError> {
        let (tx, rx) = sync_channel::<GifFrame>(GIF_CHANNEL_CAPACITY);
        let dropped = Arc::new(AtomicU32::new(0));
        let handle = thread::Builder::new()
            .name("loam-app::gif-encoder".into())
            .spawn(move || gif_encoder_loop(path, rx))
            .map_err(CaptureError::Thread)?;
        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
            dropped,
        })
    }

    pub(super) fn detach(mut self) -> JoinHandle<()> {
        self.tx.take();
        self.handle.take().expect("worker handle present")
    }

    pub(super) fn try_send(&self, frame: GifFrame) {
        let Some(tx) = self.tx.as_ref() else { return };
        match tx.try_send(frame) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                let count = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
                if count.is_power_of_two() {
                    tracing::warn!(
                        "capture: GIF encoder queue full; dropped {count} frame(s) so far"
                    );
                }
            }
            Err(TrySendError::Disconnected(_)) => {
                tracing::error!("capture: GIF encoder thread exited unexpectedly");
            }
        }
    }

    pub(super) fn dropped(&self) -> u32 {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl Drop for GifWorker {
    fn drop(&mut self) {
        // Fallback for a worker never detached, such as a panic unwinding through `Capture`.
        self.tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn gif_encoder_loop(path: PathBuf, rx: Receiver<GifFrame>) {
    let mut encoder: Option<gif::Encoder<BufWriter<File>>> = None;
    let mut last_captured_at: Option<Instant> = None;
    for frame in rx {
        if let Err(e) = encode_one_frame(&path, &mut encoder, &mut last_captured_at, frame) {
            tracing::error!("capture: gif encode error: {e}");
            return;
        }
    }
    drop(encoder);
    tracing::info!("capture: gif file finalised at {}", path.display());
}

fn encode_one_frame(
    path: &Path,
    encoder: &mut Option<gif::Encoder<BufWriter<File>>>,
    last_captured_at: &mut Option<Instant>,
    frame: GifFrame,
) -> Result<(), CaptureError> {
    let (out_w, out_h) = scaled_dims(frame.src_width, frame.src_height, frame.scale)?;
    let too_large = |_| CaptureError::GifSize {
        width: out_w,
        height: out_h,
    };
    let w_u16: u16 = out_w.try_into().map_err(too_large)?;
    let h_u16: u16 = out_h.try_into().map_err(too_large)?;

    let enc = match encoder {
        Some(e) => e,
        None => {
            let file = File::create(path).map_err(create("gif output", path.to_path_buf()))?;
            let global_palette_bytes: Vec<u8> = frame
                .global_palette
                .as_ref()
                .map(|nq| nq.color_map_rgb())
                .unwrap_or_default();
            let mut e =
                gif::Encoder::new(BufWriter::new(file), w_u16, h_u16, &global_palette_bytes)?;
            e.set_repeat(gif::Repeat::Infinite)?;
            tracing::info!(
                "capture: gif encoder opened {}x{} target {}cs/frame ({} fps); \
                 palette={}; actual delays computed from wall-clock per-frame",
                w_u16,
                h_u16,
                frame.default_delay_cs,
                if frame.default_delay_cs == 0 {
                    0
                } else {
                    100 / frame.default_delay_cs as u32
                },
                if frame.global_palette.is_some() {
                    "global (shared NeuQuant)"
                } else {
                    "per-frame"
                }
            );
            encoder.insert(e)
        }
    };

    let delay_cs = match *last_captured_at {
        None => frame.default_delay_cs,
        Some(prev) => {
            let ms = frame.captured_at.duration_since(prev).as_millis() as u64;
            let cs = (ms + 5) / 10;
            cs.clamp(1, u16::MAX as u64) as u16
        }
    };
    *last_captured_at = Some(frame.captured_at);

    let mut buf: Vec<u8> = if frame.scale.is_some() {
        let src: ::image::RgbaImage =
            ::image::ImageBuffer::from_raw(frame.src_width, frame.src_height, frame.rgba).ok_or(
                CaptureError::RgbaSize {
                    width: frame.src_width,
                    height: frame.src_height,
                },
            )?;
        let dst =
            ::image::imageops::resize(&src, out_w, out_h, ::image::imageops::FilterType::Lanczos3);
        dst.into_raw()
    } else {
        frame.rgba
    };

    let mut gif_frame = if let Some(nq) = &frame.global_palette {
        for px in buf.chunks_exact_mut(4) {
            if px[3] != 0 {
                px[3] = 0xFF;
            }
        }
        let mut indices = Vec::with_capacity((out_w as usize) * (out_h as usize));
        for px in buf.chunks_exact(4) {
            indices.push(nq.index_of(px) as u8);
        }
        gif::Frame {
            width: w_u16,
            height: h_u16,
            buffer: std::borrow::Cow::Owned(indices),
            palette: None,
            ..gif::Frame::default()
        }
    } else {
        // NeuQuant, Dekker (1994).
        gif::Frame::from_rgba_speed(w_u16, h_u16, &mut buf, 10)
    };
    gif_frame.delay = delay_cs;
    // `Background` disposal flashes between frames.
    gif_frame.dispose = gif::DisposalMethod::Any;
    enc.write_frame(&gif_frame)?;
    Ok(())
}
