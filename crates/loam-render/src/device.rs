//! sRGB surface: the scene draws and resolves through sRGB views; the UI
//! paints through the swapchain's non-sRGB twin. Non-sRGB surface: MSAA off,
//! scene and UI draw into [`OffscreenTarget`] and the composite encodes them.

use anyhow::Result;
use std::fmt;
use std::ops::Deref;
use std::sync::{Arc, Mutex};
use wgpu::*;
use winit::window::Window;

use crate::gpu_timer::GpuTimer;

pub const GPU_TIMER_FEATURES: Features =
    Features::TIMESTAMP_QUERY.union(Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

#[derive(Clone, Debug)]
pub struct FeatureRequest {
    pub required_features: Features,
    /// Requested where the adapter has them; the rest are logged and skipped.
    pub optional_features: Features,
    pub required_limits: Limits,
}

impl Default for FeatureRequest {
    fn default() -> Self {
        Self {
            required_features: Features::empty(),
            optional_features: GPU_TIMER_FEATURES,
            required_limits: Limits::default(),
        }
    }
}

impl FeatureRequest {
    /// The set to request: `required_features` plus the adapter's share of `optional_features`.
    pub fn resolve(
        &self,
        adapter_features: Features,
        adapter_limits: &Limits,
    ) -> std::result::Result<Features, MissingGpuCapability> {
        let missing = self.required_features - adapter_features;
        if !missing.is_empty() {
            return Err(MissingGpuCapability::Feature(missing));
        }
        let mut missing_limit = None;
        self.required_limits.check_limits_with_fail_fn(
            adapter_limits,
            true,
            |name, required, available| {
                missing_limit = Some(MissingGpuCapability::Limit {
                    name,
                    required,
                    available,
                });
            },
        );
        match missing_limit {
            Some(missing) => Err(missing),
            None => Ok(self.required_features | (self.optional_features & adapter_features)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MissingGpuCapability {
    Feature(Features),
    Limit {
        name: &'static str,
        required: u64,
        available: u64,
    },
}

impl fmt::Display for MissingGpuCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Feature(features) => write!(
                f,
                "this adapter does not provide the required GPU feature `{features:?}`"
            ),
            Self::Limit {
                name,
                required,
                available,
            } => write!(
                f,
                "this adapter limit `{name}` is {available}; the app requires {required}"
            ),
        }
    }
}

impl std::error::Error for MissingGpuCapability {}

#[derive(Debug)]
pub struct DeviceLoss {
    pub reason: DeviceLostReason,
    pub message: String,
}

#[derive(Default)]
struct LossSignal(Mutex<Option<DeviceLoss>>);

impl LossSignal {
    fn take(&self) -> Option<DeviceLoss> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).take()
    }
}

pub struct GpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub gpu_timer: Option<GpuTimer>,
    request: FeatureRequest,
    loss: Arc<LossSignal>,
}

impl GpuContext {
    /// `compatible_surface` is `None` for headless work.
    pub async fn new(
        instance: Instance,
        request: FeatureRequest,
        compatible_surface: Option<&Surface<'_>>,
    ) -> Result<Self> {
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                compatible_surface,
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
            })
            .await?;
        let (device, queue, loss) = request_device(&adapter, &request).await?;
        let gpu_timer = GpuTimer::new(&device, &queue);
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            gpu_timer,
            request,
            loss,
        })
    }

    /// Poll at a frame boundary; a loss is reported once.
    pub fn take_device_loss(&self) -> Option<DeviceLoss> {
        self.loss.take()
    }

    /// A new device from the same adapter and request; the old device's work is cancelled and its late callbacks are ignored.
    pub async fn recover(&mut self) -> Result<()> {
        let (device, queue, loss) = request_device(&self.adapter, &self.request).await?;
        self.device.destroy();
        self.gpu_timer = GpuTimer::new(&device, &queue);
        self.device = device;
        self.queue = queue;
        self.loss = loss;
        Ok(())
    }
}

async fn request_device(
    adapter: &Adapter,
    request: &FeatureRequest,
) -> Result<(Device, Queue, Arc<LossSignal>)> {
    let resolved = request.resolve(adapter.features(), &adapter.limits())?;
    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            label: Some("Loam Device"),
            required_features: resolved,
            required_limits: request.required_limits.clone(),
            memory_hints: MemoryHints::default(),
            trace: Trace::Off,
            experimental_features: Default::default(),
        })
        .await?;
    tracing::info!(
        "GPU features enabled: {resolved:?}; optional features absent: {:?}",
        request.optional_features - resolved
    );
    let loss = Arc::new(LossSignal::default());
    let signal = loss.clone();
    device.set_device_lost_callback(move |reason, message| {
        *signal.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(DeviceLoss { reason, message });
    });
    Ok((device, queue, loss))
}

pub struct SurfaceBundle {
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
}

pub struct MsaaTarget {
    #[allow(dead_code)]
    texture: Texture,
    pub view: TextureView,
}

pub struct OffscreenTarget {
    #[allow(dead_code)]
    texture: Texture,
    pub view: TextureView,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UiTargetFormats {
    pub ui_format: TextureFormat,
    /// `None`: the UI paints through the swapchain's own format.
    pub swap_view_format: Option<TextureFormat>,
}

fn ui_target_formats(surface_format: TextureFormat, downlevel: DownlevelFlags) -> UiTargetFormats {
    if !surface_format.is_srgb() {
        return UiTargetFormats {
            ui_format: surface_format.add_srgb_suffix(),
            swap_view_format: None,
        };
    }
    if !downlevel.contains(DownlevelFlags::SURFACE_VIEW_FORMATS) {
        return UiTargetFormats {
            ui_format: surface_format,
            swap_view_format: None,
        };
    }
    let gamma = surface_format.remove_srgb_suffix();
    UiTargetFormats {
        ui_format: gamma,
        swap_view_format: Some(gamma),
    }
}

fn surface_configuration(
    format: TextureFormat,
    size: winit::dpi::PhysicalSize<u32>,
    alpha_mode: CompositeAlphaMode,
    ui_targets: UiTargetFormats,
) -> SurfaceConfiguration {
    SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        format,
        width: size.width,
        height: size.height,
        present_mode: PresentMode::Fifo,
        alpha_mode,
        view_formats: ui_targets.swap_view_format.into_iter().collect(),
        desired_maximum_frame_latency: 2,
    }
}

fn ui_view_descriptor(ui_view_format: Option<TextureFormat>) -> TextureViewDescriptor<'static> {
    TextureViewDescriptor {
        format: ui_view_format,
        ..Default::default()
    }
}

// Both ends of the MSAA resolve take the target's own format; a non-sRGB view would average encoded bytes.
fn scene_view_descriptor() -> TextureViewDescriptor<'static> {
    TextureViewDescriptor::default()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PresentationSpec {
    format: TextureFormat,
    sample_count: u32,
    scene_format: Option<TextureFormat>,
}

struct Presentation {
    spec: PresentationSpec,
    msaa_target: Option<MsaaTarget>,
    scene_target: Option<OffscreenTarget>,
    composite: Option<crate::composite::CompositeNode>,
}

impl Presentation {
    fn build(device: &Device, spec: PresentationSpec, size: winit::dpi::PhysicalSize<u32>) -> Self {
        let composite = spec
            .scene_format
            .map(|_| crate::composite::CompositeNode::new(device, spec.format));
        let mut presentation = Self {
            spec,
            msaa_target: None,
            scene_target: None,
            composite,
        };
        presentation.resize(device, size);
        presentation
    }

    fn resize(&mut self, device: &Device, size: winit::dpi::PhysicalSize<u32>) {
        let spec = self.spec;
        self.msaa_target = (spec.sample_count > 1).then(|| {
            create_msaa_target(
                device,
                spec.format,
                size.width,
                size.height,
                spec.sample_count,
            )
        });
        if let (Some(scene_fmt), Some(composite)) = (spec.scene_format, self.composite.as_mut()) {
            let scene = create_scene_target(device, scene_fmt, size.width, size.height);
            composite.set_scene_view(device, &scene.view);
            self.scene_target = Some(scene);
        }
    }

    fn rebuild(&mut self, device: &Device, size: winit::dpi::PhysicalSize<u32>) {
        *self = Self::build(device, self.spec, size);
    }
}

pub struct RenderDevice {
    pub context: GpuContext,
    pub surface_bundle: SurfaceBundle,
    presentation: Presentation,
    present_modes: Vec<PresentMode>,
    ui_targets: UiTargetFormats,
}

impl Deref for RenderDevice {
    type Target = GpuContext;

    fn deref(&self) -> &GpuContext {
        &self.context
    }
}

impl RenderDevice {
    pub async fn new(
        window: Arc<Window>,
        request: FeatureRequest,
        requested_msaa_samples: u32,
    ) -> Result<Self> {
        let instance = Instance::default();
        let surface = instance.create_surface(window.clone())?;
        let size = window.inner_size();
        let context = GpuContext::new(instance, request, Some(&surface)).await?;
        Self::attach(context, surface, size, requested_msaa_samples)
    }

    /// Configures `surface` on the context's device and builds the presentation resources for it.
    pub fn attach(
        context: GpuContext,
        surface: Surface<'static>,
        size: winit::dpi::PhysicalSize<u32>,
        requested_msaa_samples: u32,
    ) -> Result<Self> {
        let caps = surface.get_capabilities(&context.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .or(caps.formats.first().copied())
            .ok_or_else(|| anyhow::anyhow!("the surface advertises no texture format"))?;
        tracing::info!(
            "surface picked format={format:?} (advertised={:?})",
            caps.formats
        );

        let alpha_mode = caps
            .alpha_modes
            .iter()
            .copied()
            .find(|m| *m == CompositeAlphaMode::Opaque)
            .or(caps.alpha_modes.first().copied())
            .ok_or_else(|| anyhow::anyhow!("the surface advertises no alpha mode"))?;

        let ui_targets =
            ui_target_formats(format, context.adapter.get_downlevel_capabilities().flags);
        if format.is_srgb() && ui_targets.swap_view_format.is_none() {
            tracing::warn!(
                "adapter lacks SURFACE_VIEW_FORMATS; UI blends in linear space and \
                 egui feathering will look thin on hairlines"
            );
        }
        let config = surface_configuration(format, size, alpha_mode, ui_targets);

        surface.configure(&context.device, &config);

        let needs_composite = !format.is_srgb();
        let effective_msaa = surface_msaa_request(format, requested_msaa_samples);
        if effective_msaa < requested_msaa_samples {
            tracing::warn!(
                "MSAA={requested_msaa_samples}x ignored: composite pass for sRGB \
                 gamma encoding (browser-WebGPU linear surface) is incompatible \
                 with MSAA in v1; falling back to sample_count=1",
            );
        }

        let sample_count = negotiate_sample_count(&context.adapter, format, effective_msaa);
        let scene_format = needs_composite.then(|| format.add_srgb_suffix());
        if let Some(scene_fmt) = scene_format {
            tracing::info!(
                "non-sRGB surface; rendering through offscreen scene target {scene_fmt:?} \
                 with composite pass to {format:?} swapchain"
            );
        }
        let presentation = Presentation::build(
            &context.device,
            PresentationSpec {
                format,
                sample_count,
                scene_format,
            },
            size,
        );

        let present_modes = caps.present_modes.clone();
        tracing::info!("surface present modes advertised: {present_modes:?}");

        Ok(Self {
            context,
            surface_bundle: SurfaceBundle {
                surface,
                config,
                size,
            },
            presentation,
            present_modes,
            ui_targets,
        })
    }

    /// Regenerable: the swapchain configuration, the MSAA and scene targets, the composite pass, and the GPU timer.
    pub async fn recover(&mut self) -> Result<()> {
        self.context.recover().await?;
        self.surface_bundle
            .surface
            .configure(&self.context.device, &self.surface_bundle.config);
        self.presentation
            .rebuild(&self.context.device, self.surface_bundle.size);
        Ok(())
    }

    /// No-op on a zero dimension, which wgpu rejects.
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.surface_bundle.size = new_size;
        self.surface_bundle.config.width = new_size.width;
        self.surface_bundle.config.height = new_size.height;
        self.surface_bundle
            .surface
            .configure(&self.context.device, &self.surface_bundle.config);
        self.presentation.resize(&self.context.device, new_size);
    }

    pub fn begin_frame(
        &self,
    ) -> std::result::Result<(SurfaceTexture, TextureView), wgpu::SurfaceError> {
        let frame = self.surface_bundle.surface.get_current_texture()?;
        let view = frame.texture.create_view(&scene_view_descriptor());
        Ok((frame, view))
    }

    pub fn sample_count(&self) -> u32 {
        self.presentation.spec.sample_count
    }

    pub fn present_mode(&self) -> PresentMode {
        self.surface_bundle.config.present_mode
    }

    pub fn supported_present_modes(&self) -> &[PresentMode] {
        &self.present_modes
    }

    /// `Fifo` is the only browser-WebGPU mode.
    pub fn set_present_mode(&mut self, mode: PresentMode) -> std::result::Result<(), PresentMode> {
        if !self.present_modes.contains(&mode) {
            return Err(mode);
        }
        if self.surface_bundle.config.present_mode == mode {
            return Ok(());
        }
        self.surface_bundle.config.present_mode = mode;
        self.surface_bundle
            .surface
            .configure(&self.context.device, &self.surface_bundle.config);
        tracing::info!("surface present_mode -> {mode:?}");
        Ok(())
    }

    pub fn msaa_view(&self) -> Option<&TextureView> {
        self.presentation.msaa_target.as_ref().map(|t| &t.view)
    }

    /// Scene-pass target priority: `msaa_view`, then this, then the swapchain view.
    pub fn scene_view(&self) -> Option<&TextureView> {
        self.presentation.scene_target.as_ref().map(|t| &t.view)
    }

    /// Pipeline constructors take this, not the surface format.
    pub fn target_format(&self) -> TextureFormat {
        self.presentation
            .spec
            .scene_format
            .unwrap_or(self.surface_bundle.config.format)
    }

    /// The format of every view the UI pass renders into.
    pub fn ui_format(&self) -> TextureFormat {
        self.ui_targets.ui_format
    }

    /// Never a `resolve_target`; the UI pass is single-sampled on every path.
    pub fn create_ui_swap_view(&self, frame: &SurfaceTexture) -> TextureView {
        frame
            .texture
            .create_view(&ui_view_descriptor(self.ui_targets.swap_view_format))
    }

    /// No-op with MSAA off; both ends take the target's own sRGB format so the resolve averages linear samples.
    pub fn resolve_scene_to_swap(&self, encoder: &mut CommandEncoder, swap_view: &TextureView) {
        let Some(msaa) = self.presentation.msaa_target.as_ref() else {
            return;
        };
        let _resolve_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("loam-render::scene-msaa-resolve"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &msaa.view,
                depth_slice: None,
                resolve_target: Some(swap_view),
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    }

    pub fn composite_to_swap(&self, encoder: &mut wgpu::CommandEncoder, swap_view: &TextureView) {
        if let Some(composite) = self.presentation.composite.as_ref() {
            composite.run(encoder, swap_view);
        }
    }

    /// Compiles the composite PSO at setup rather than on the first frame.
    pub fn warm_composite(&self) {
        if self.presentation.composite.is_none() {
            return;
        }
        let format = self.surface_bundle.config.format;
        let dummy = self
            .context
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("loam-render::composite::warm dummy"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
        let dummy_view = dummy.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("loam-render::composite::warm encoder"),
                });
        self.composite_to_swap(&mut encoder, &dummy_view);
        self.context.queue.submit(Some(encoder.finish()));
    }
}

fn create_scene_target(
    device: &Device,
    format: TextureFormat,
    width: u32,
    height: u32,
) -> OffscreenTarget {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("loam-render::scene_target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    OffscreenTarget { texture, view }
}

fn surface_msaa_request(surface_format: TextureFormat, requested: u32) -> u32 {
    if surface_format.is_srgb() {
        requested
    } else {
        1
    }
}

fn negotiate_sample_count(adapter: &Adapter, format: TextureFormat, requested: u32) -> u32 {
    if requested <= 1 {
        return 1;
    }
    let features = adapter.get_texture_format_features(format);
    let flags = features.flags;
    for count in [16u32, 8, 4, 2] {
        if count > requested {
            continue;
        }
        let supported = match count {
            2 => flags.contains(TextureFormatFeatureFlags::MULTISAMPLE_X2),
            4 => flags.contains(TextureFormatFeatureFlags::MULTISAMPLE_X4),
            8 => flags.contains(TextureFormatFeatureFlags::MULTISAMPLE_X8),
            16 => flags.contains(TextureFormatFeatureFlags::MULTISAMPLE_X16),
            _ => false,
        };
        if supported {
            if count != requested {
                tracing::warn!(
                    "requested MSAA {requested}x not supported on {format:?}; falling back to {count}x"
                );
            }
            return count;
        }
    }
    tracing::warn!("no multisampled count supported on {format:?}; MSAA disabled");
    1
}

fn msaa_texture_descriptor(
    format: TextureFormat,
    width: u32,
    height: u32,
    sample_count: u32,
) -> TextureDescriptor<'static> {
    TextureDescriptor {
        label: Some("loam-render::msaa-color"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    }
}

fn create_msaa_target(
    device: &Device,
    format: TextureFormat,
    width: u32,
    height: u32,
    sample_count: u32,
) -> MsaaTarget {
    let texture = device.create_texture(&msaa_texture_descriptor(
        format,
        width,
        height,
        sample_count,
    ));
    let view = texture.create_view(&scene_view_descriptor());
    MsaaTarget { texture, view }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOTH: DownlevelFlags =
        DownlevelFlags::SURFACE_VIEW_FORMATS.union(DownlevelFlags::VIEW_FORMATS);

    const SURFACES: [TextureFormat; 4] = [
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba16Float,
    ];

    const DOWNLEVELS: [DownlevelFlags; 5] = [
        DownlevelFlags::empty(),
        DownlevelFlags::SURFACE_VIEW_FORMATS,
        DownlevelFlags::VIEW_FORMATS,
        BOTH,
        DownlevelFlags::all(),
    ];

    const SIZE: winit::dpi::PhysicalSize<u32> = winit::dpi::PhysicalSize {
        width: 800,
        height: 600,
    };

    #[test]
    fn srgb_surface_registers_the_gamma_twin_only_with_surface_view_formats() {
        let srgb = TextureFormat::Bgra8UnormSrgb;
        let gamma = TextureFormat::Bgra8Unorm;
        let table = [
            (DownlevelFlags::empty(), None),
            (DownlevelFlags::SURFACE_VIEW_FORMATS, Some(gamma)),
            (DownlevelFlags::VIEW_FORMATS, None),
            (BOTH, Some(gamma)),
            (DownlevelFlags::all(), Some(gamma)),
        ];
        for (downlevel, expected) in table {
            let targets = ui_target_formats(srgb, downlevel);
            assert_eq!(targets.swap_view_format, expected, "{downlevel:?}");
            assert_eq!(targets.ui_format, expected.unwrap_or(srgb), "{downlevel:?}");
        }
    }

    #[test]
    fn composite_path_registers_no_view_formats_and_targets_the_scene_format() {
        let table = [
            (TextureFormat::Bgra8Unorm, TextureFormat::Bgra8UnormSrgb),
            (TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb),
            (TextureFormat::Rgba16Float, TextureFormat::Rgba16Float),
        ];
        for (surface, scene) in table {
            for downlevel in [DownlevelFlags::empty(), DownlevelFlags::all()] {
                let targets = ui_target_formats(surface, downlevel);
                assert_eq!(targets.swap_view_format, None, "{surface:?} {downlevel:?}");
                assert_eq!(targets.ui_format, scene, "{surface:?} {downlevel:?}");
            }
        }
    }

    #[test]
    fn ui_view_requests_match_their_target_registration_in_both_arms() {
        for surface in SURFACES {
            for downlevel in DOWNLEVELS {
                let case = format!("{surface:?} {downlevel:?}");
                let expected = (surface.is_srgb()
                    && downlevel.contains(DownlevelFlags::SURFACE_VIEW_FORMATS))
                .then(|| surface.remove_srgb_suffix());
                let targets = ui_target_formats(surface, downlevel);

                let swap_request = ui_view_descriptor(targets.swap_view_format).format;
                assert_eq!(swap_request, expected, "swapchain request: {case}");
                let config =
                    surface_configuration(surface, SIZE, CompositeAlphaMode::Opaque, targets);
                assert_eq!(
                    config.view_formats,
                    swap_request.into_iter().collect::<Vec<_>>(),
                    "swapchain registration: {case}"
                );
            }
        }
    }

    #[test]
    fn launch_names_the_absent_required_feature_or_limit() {
        let adapter_features = Features::TIMESTAMP_QUERY | Features::SHADER_F16;
        let adapter_limits = Limits::default();
        let request = FeatureRequest {
            required_features: Features::SHADER_F16 | Features::MULTIVIEW,
            optional_features: Features::TIMESTAMP_QUERY | Features::POLYGON_MODE_LINE,
            required_limits: Limits::default(),
        };
        let error = request
            .resolve(adapter_features, &adapter_limits)
            .unwrap_err();
        assert_eq!(error, MissingGpuCapability::Feature(Features::MULTIVIEW));
        assert!(error.to_string().contains("MULTIVIEW"), "{error}");

        let request = FeatureRequest {
            required_features: Features::SHADER_F16,
            required_limits: Limits {
                max_texture_dimension_2d: adapter_limits.max_texture_dimension_2d * 2,
                ..Limits::default()
            },
            ..request
        };
        let error = request
            .resolve(adapter_features, &adapter_limits)
            .unwrap_err();
        assert_eq!(
            error,
            MissingGpuCapability::Limit {
                name: "max_texture_dimension_2d",
                required: u64::from(adapter_limits.max_texture_dimension_2d) * 2,
                available: u64::from(adapter_limits.max_texture_dimension_2d),
            }
        );
        assert!(
            error.to_string().contains("max_texture_dimension_2d"),
            "{error}"
        );

        let request = FeatureRequest {
            required_limits: Limits::default(),
            ..request
        };
        assert_eq!(
            request.resolve(adapter_features, &adapter_limits),
            Ok(Features::SHADER_F16 | Features::TIMESTAMP_QUERY)
        );
    }

    fn noop_context() -> GpuContext {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::NOOP,
            backend_options: BackendOptions {
                noop: NoopBackendOptions { enable: true },
                ..Default::default()
            },
            ..Default::default()
        });
        pollster::block_on(GpuContext::new(instance, FeatureRequest::default(), None)).unwrap()
    }

    fn render_target(device: &Device, format: TextureFormat) -> TextureView {
        device
            .create_texture(&TextureDescriptor {
                label: Some("recovery target"),
                size: Extent3d {
                    width: SIZE.width,
                    height: SIZE.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&TextureViewDescriptor::default())
    }

    #[test]
    fn regenerable_presentation_resources_rebuild_on_the_new_device_after_device_loss() {
        let mut context = noop_context();
        let spec = PresentationSpec {
            format: TextureFormat::Bgra8Unorm,
            sample_count: 1,
            scene_format: Some(TextureFormat::Bgra8UnormSrgb),
        };
        let mut presentation = Presentation::build(&context.device, spec, SIZE);
        let old_device = context.device.clone();
        old_device.destroy();

        pollster::block_on(context.recover()).unwrap();
        presentation.rebuild(&context.device, SIZE);
        let _ = old_device.poll(PollType::Poll);
        assert!(
            context.take_device_loss().is_none(),
            "the old device's late loss callback reached the new context"
        );

        let device = &context.device;
        device.push_error_scope(ErrorFilter::Validation);
        let target = render_target(device, spec.format);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        if let Some(timer) = context.gpu_timer.as_ref() {
            timer.write_start(&mut encoder);
        }
        if let Some(composite) = presentation.composite.as_ref() {
            composite.run(&mut encoder, &target);
        }
        context.queue.submit(Some(encoder.finish()));
        let error = pollster::block_on(device.pop_error_scope());
        assert!(
            error.is_none(),
            "a resource stayed on the lost device: {error:?}"
        );

        context.device.destroy();
        let _ = context.device.poll(PollType::Poll);
        let loss = context.take_device_loss().unwrap();
        assert_eq!(loss.reason, DeviceLostReason::Destroyed);
    }
}
