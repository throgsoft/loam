use std::fmt;
use std::io;
use std::path::PathBuf;

#[derive(Debug)]
pub(super) enum CaptureError {
    /// The text names what the path is for.
    Create {
        what: &'static str,
        path: PathBuf,
        error: io::Error,
    },
    Thread(io::Error),
    Png(::png::EncodingError),
    Gif(::gif::EncodingError),
    Image {
        path: PathBuf,
        error: ::image::ImageError,
    },
    Scale {
        target: u32,
        width: u32,
        height: u32,
    },
    GifSize {
        width: u32,
        height: u32,
    },
    /// An APNG frame whose size differs from the first frame's.
    FrameSize {
        size: (u32, u32),
        first: (u32, u32),
    },
    /// The pixel buffer does not hold this many RGBA pixels.
    RgbaSize {
        width: u32,
        height: u32,
    },
    RowBytes(u32),
    Poll(wgpu::PollError),
    MapDropped,
    Map(wgpu::BufferAsyncError),
}

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Create { what, path, error } => {
                write!(f, "create {what} {}: {error}", path.display())
            }
            Self::Thread(error) => write!(f, "spawn the encoder thread: {error}"),
            Self::Png(error) => write!(f, "apng encode: {error}"),
            Self::Gif(error) => write!(f, "gif encode: {error}"),
            Self::Image { path, error } => write!(f, "write png {}: {error}", path.display()),
            Self::Scale {
                target,
                width,
                height,
            } => write!(f, "invalid scale: target_w={target}, src={width}x{height}"),
            Self::GifSize { width, height } => {
                write!(f, "gif size {width}x{height} exceeds 65535")
            }
            Self::FrameSize { size, first } => write!(
                f,
                "apng frame dims {}x{} != first frame {}x{}",
                size.0, size.1, first.0, first.1
            ),
            Self::RgbaSize { width, height } => write!(f, "RGBA size mismatch at {width}x{height}"),
            Self::RowBytes(width) => write!(f, "width {width} * 4 overflows u32"),
            Self::Poll(error) => write!(f, "device.poll on capture readback failed: {error}"),
            Self::MapDropped => f.write_str("capture map callback dropped"),
            Self::Map(error) => write!(f, "capture buffer mapping failed: {error}"),
        }
    }
}

impl From<::png::EncodingError> for CaptureError {
    fn from(error: ::png::EncodingError) -> Self {
        Self::Png(error)
    }
}

impl From<::gif::EncodingError> for CaptureError {
    fn from(error: ::gif::EncodingError) -> Self {
        Self::Gif(error)
    }
}

pub(super) fn create(what: &'static str, path: PathBuf) -> impl FnOnce(io::Error) -> CaptureError {
    move |error| CaptureError::Create { what, path, error }
}
