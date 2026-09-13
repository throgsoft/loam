use anyhow::{anyhow, Result};
use loam_render::device::{FeatureRequest, GpuContext, RenderDevice};
use wgpu::{
    CompositeAlphaMode, Device, DownlevelFlags, Instance, PresentMode, Surface,
    SurfaceConfiguration, SurfaceTexture, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
};

pub(crate) struct SurfaceHost {
    surface: Surface<'static>,
    config: SurfaceConfiguration,
    size: (u32, u32),
    present_modes: Vec<PresentMode>,
}

impl SurfaceHost {
    pub(crate) async fn new(
        instance: Instance,
        surface: Surface<'static>,
        size: (u32, u32),
        request: FeatureRequest,
        requested_msaa_samples: u32,
    ) -> Result<(Self, RenderDevice)> {
        let context = GpuContext::new(instance, request, Some(&surface)).await?;
        let caps = surface.get_capabilities(&context.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(TextureFormat::is_srgb)
            .or(caps.formats.first().copied())
            .ok_or_else(|| anyhow!("the surface advertises no texture format"))?;
        tracing::info!(
            "surface picked format={format:?} (advertised={:?})",
            caps.formats
        );
        let alpha_mode = caps
            .alpha_modes
            .iter()
            .copied()
            .find(|mode| *mode == CompositeAlphaMode::Opaque)
            .or(caps.alpha_modes.first().copied())
            .ok_or_else(|| anyhow!("the surface advertises no alpha mode"))?;
        let downlevel = context.adapter.get_downlevel_capabilities().flags;
        let view_format = surface_view_format(format, downlevel);
        let configured_size = (size.0.max(1), size.1.max(1));
        let config = surface_configuration(format, configured_size, alpha_mode, view_format);
        surface.configure(&context.device, &config);
        let present_modes = caps.present_modes;
        tracing::info!("surface present modes advertised: {present_modes:?}");
        let renderer = RenderDevice::new(context, format, requested_msaa_samples, configured_size);
        Ok((
            Self {
                surface,
                config,
                size,
                present_modes,
            },
            renderer,
        ))
    }

    pub(crate) fn size(&self) -> (u32, u32) {
        self.size
    }

    pub(crate) fn format(&self) -> TextureFormat {
        self.config.format
    }

    pub(crate) fn reconfigure(&self, device: &Device) {
        self.surface.configure(device, &self.config);
    }

    pub(crate) fn resize(&mut self, device: &Device, size: (u32, u32)) {
        self.size = size;
        if size.0 == 0 || size.1 == 0 {
            return;
        }
        self.config.width = size.0;
        self.config.height = size.1;
        self.reconfigure(device);
    }

    pub(crate) fn begin_frame(
        &self,
    ) -> std::result::Result<(SurfaceTexture, TextureView), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&TextureViewDescriptor::default());
        Ok((frame, view))
    }

    pub(crate) fn set_vsync(&mut self, device: &Device, enabled: bool) {
        let target = if enabled {
            PresentMode::Fifo
        } else {
            [PresentMode::Mailbox, PresentMode::Immediate]
                .into_iter()
                .find(|mode| self.present_modes.contains(mode))
                .unwrap_or(self.config.present_mode)
        };
        if self.config.present_mode == target {
            return;
        }
        self.config.present_mode = target;
        self.reconfigure(device);
        tracing::info!("surface present_mode={target:?}");
    }
}

fn surface_view_format(format: TextureFormat, downlevel: DownlevelFlags) -> Option<TextureFormat> {
    (format.is_srgb() && downlevel.contains(DownlevelFlags::SURFACE_VIEW_FORMATS))
        .then(|| format.remove_srgb_suffix())
}

fn surface_configuration(
    format: TextureFormat,
    size: (u32, u32),
    alpha_mode: CompositeAlphaMode,
    view_format: Option<TextureFormat>,
) -> SurfaceConfiguration {
    SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        format,
        width: size.0,
        height: size.1,
        present_mode: PresentMode::Fifo,
        alpha_mode,
        view_formats: view_format.into_iter().collect(),
        desired_maximum_frame_latency: 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_configuration_registers_only_a_supported_gamma_view() {
        let srgb = TextureFormat::Bgra8UnormSrgb;
        let gamma = TextureFormat::Bgra8Unorm;
        for (format, flags, expected) in [
            (srgb, DownlevelFlags::empty(), None),
            (srgb, DownlevelFlags::VIEW_FORMATS, None),
            (srgb, DownlevelFlags::SURFACE_VIEW_FORMATS, Some(gamma)),
            (gamma, DownlevelFlags::all(), None),
        ] {
            let view_format = surface_view_format(format, flags);
            let config =
                surface_configuration(format, (800, 600), CompositeAlphaMode::Opaque, view_format);
            assert_eq!(view_format, expected);
            assert_eq!(
                config.view_formats,
                expected.into_iter().collect::<Vec<_>>()
            );
        }
    }
}
