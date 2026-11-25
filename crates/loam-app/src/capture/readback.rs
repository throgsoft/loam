use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Extent3d, MapMode, Origin3d,
    PollType, TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture,
    TextureAspect, TextureFormat,
};

use super::error::CaptureError;

// Row-major RGBA8, already R/B-swapped from BGRA sources.
pub(super) struct RawImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

pub(super) struct TextureReadback {
    buffer: Buffer,
    width: u32,
    height: u32,
    unpadded_bpr: u32,
    padded_bpr: u32,
    format: TextureFormat,
}

pub(super) fn record_texture_rgba(
    device: &Device,
    encoder: &mut CommandEncoder,
    texture: &Texture,
    width: u32,
    height: u32,
    format: TextureFormat,
) -> Result<TextureReadback, CaptureError> {
    let unpadded_bpr = width.checked_mul(4).ok_or(CaptureError::RowBytes(width))?;
    let padded_bpr = unpadded_bpr.next_multiple_of(256);
    let buffer_size = (padded_bpr as u64) * (height as u64);

    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("loam-app::capture-staging"),
        size: buffer_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: None,
            },
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    Ok(TextureReadback {
        buffer,
        width,
        height,
        unpadded_bpr,
        padded_bpr,
        format,
    })
}

impl TextureReadback {
    pub(super) fn read(self, device: &Device) -> Result<RawImage, CaptureError> {
        let Self {
            buffer,
            width,
            height,
            unpadded_bpr,
            padded_bpr,
            format,
        } = self;

        let slice = buffer.slice(..);
        let (mapped_tx, mapped_rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            let _ = mapped_tx.send(result);
        });
        device
            .poll(PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(CaptureError::Poll)?;

        mapped_rx
            .recv()
            .map_err(|_| CaptureError::MapDropped)?
            .map_err(CaptureError::Map)?;
        let data = slice.get_mapped_range();
        let mut rgba = Vec::with_capacity((unpadded_bpr * height) as usize);
        for row in 0..height as usize {
            let start = row * padded_bpr as usize;
            let end = start + unpadded_bpr as usize;
            rgba.extend_from_slice(&data[start..end]);
        }
        drop(data);
        buffer.unmap();

        if format_is_bgra(format) {
            for px in rgba.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }

        Ok(RawImage {
            width,
            height,
            rgba,
        })
    }
}

fn format_is_bgra(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb
    )
}
