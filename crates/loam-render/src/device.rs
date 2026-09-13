//! Non-sRGB presentation uses an sRGB offscreen target and a final composite.

use anyhow::Result;
use std::fmt;
use std::ops::Deref;
use std::sync::{Arc, Mutex};
use wgpu::*;

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

#[cfg(test)]
pub(crate) fn noop_context() -> GpuContext {
    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::NOOP,
        backend_options: BackendOptions {
            noop: NoopBackendOptions { enable: true },
            ..Default::default()
        },
        ..Default::default()
    });
    pollster::block_on(GpuContext::new(instance, FeatureRequest::default(), None))
        .expect("the noop backend always yields a context")
}

#[derive(Debug)]
pub struct DeviceLoss {
    pub reason: DeviceLostReason,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuErrorKind {
    OutOfMemory,
    Validation,
    Internal,
}

impl fmt::Display for GpuErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory => f.write_str("out of memory"),
            Self::Validation => f.write_str("validation"),
            Self::Internal => f.write_str("internal"),
        }
    }
}

#[derive(Debug)]
pub struct UncapturedGpuError {
    backend: Backend,
    kind: GpuErrorKind,
    cause: String,
}

impl fmt::Display for UncapturedGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} backend {} error: {}",
            self.backend, self.kind, self.cause
        )
    }
}

impl std::error::Error for UncapturedGpuError {}

#[derive(Default)]
struct DeviceSignals {
    loss: Option<DeviceLoss>,
    error: Option<UncapturedGpuError>,
}

#[derive(Default)]
pub(crate) struct LossSignal(Mutex<DeviceSignals>);

impl LossSignal {
    fn take_loss(&self) -> Option<DeviceLoss> {
        self.0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .loss
            .take()
    }

    pub(crate) fn take_error(&self) -> Option<UncapturedGpuError> {
        self.0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .error
            .take()
    }
}

pub struct GpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
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
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            request,
            loss,
        })
    }

    /// Poll at a frame boundary; a loss is reported once.
    pub fn take_device_loss(&self) -> Option<DeviceLoss> {
        self.loss.take_loss()
    }

    pub fn take_uncaptured_error(&self) -> Option<UncapturedGpuError> {
        self.loss.take_error()
    }

    pub(crate) fn loss_signal(&self) -> Arc<LossSignal> {
        self.loss.clone()
    }

    /// A new device from the same adapter and request; the old device's work is cancelled and its late callbacks are ignored.
    pub async fn recover(&mut self) -> Result<()> {
        let (device, queue, loss) = request_device(&self.adapter, &self.request).await?;
        self.device.destroy();
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
    let backend = adapter.get_info().backend;
    let loss = Arc::new(LossSignal::default());
    let signal = loss.clone();
    device.set_device_lost_callback(move |reason, message| {
        let mut signal = signal.0.lock().unwrap_or_else(|error| error.into_inner());
        signal.loss.get_or_insert(DeviceLoss { reason, message });
    });
    let signal = loss.clone();
    device.on_uncaptured_error(Arc::new(move |error| {
        let kind = match &error {
            Error::OutOfMemory { .. } => GpuErrorKind::OutOfMemory,
            Error::Validation { .. } => GpuErrorKind::Validation,
            Error::Internal { .. } => GpuErrorKind::Internal,
        };
        let mut cause = error.to_string();
        let mut source = std::error::Error::source(&error);
        while let Some(next) = source {
            cause.push_str(": ");
            cause.push_str(&next.to_string());
            source = next.source();
        }
        let mut signal = signal.0.lock().unwrap_or_else(|error| error.into_inner());
        signal.error.get_or_insert(UncapturedGpuError {
            backend,
            kind,
            cause,
        });
    }));
    Ok((device, queue, loss))
}

pub struct OffscreenTarget {
    #[allow(dead_code)]
    texture: Texture,
    pub view: TextureView,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PresentationSpec {
    format: TextureFormat,
    scene_format: Option<TextureFormat>,
}

struct Presentation {
    spec: PresentationSpec,
    scene_target: Option<OffscreenTarget>,
    composite: Option<crate::composite::CompositeNode>,
}

impl Presentation {
    fn build(device: &Device, spec: PresentationSpec, size: (u32, u32)) -> Self {
        let composite = spec
            .scene_format
            .map(|_| crate::composite::CompositeNode::new(device, spec.format));
        let mut presentation = Self {
            spec,
            scene_target: None,
            composite,
        };
        presentation.resize(device, size);
        presentation
    }

    fn resize(&mut self, device: &Device, size: (u32, u32)) {
        let spec = self.spec;
        if let (Some(scene_fmt), Some(composite)) = (spec.scene_format, self.composite.as_mut()) {
            let scene = create_scene_target(device, scene_fmt, size.0, size.1);
            composite.set_scene_view(device, &scene.view);
            self.scene_target = Some(scene);
        }
    }

    fn rebuild(&mut self, device: &Device, size: (u32, u32)) {
        *self = Self::build(device, self.spec, size);
    }
}

pub struct RenderDevice {
    pub context: GpuContext,
    presentation: Presentation,
    size: (u32, u32),
}

impl Deref for RenderDevice {
    type Target = GpuContext;

    fn deref(&self) -> &GpuContext {
        &self.context
    }
}

impl RenderDevice {
    pub fn new(context: GpuContext, surface_format: TextureFormat, size: (u32, u32)) -> Self {
        let needs_composite = !surface_format.is_srgb();
        let scene_format = needs_composite.then(|| surface_format.add_srgb_suffix());
        if let Some(scene_fmt) = scene_format {
            tracing::info!(
                "non-sRGB surface; rendering through offscreen scene target {scene_fmt:?} \
                 with composite pass to {surface_format:?} swapchain"
            );
        }
        let presentation = Presentation::build(
            &context.device,
            PresentationSpec {
                format: surface_format,
                scene_format,
            },
            size,
        );

        Self {
            context,
            presentation,
            size,
        }
    }

    /// Rebuilds the device and the presentation targets.
    pub async fn recover(&mut self) -> Result<()> {
        self.context.recover().await?;
        self.presentation.rebuild(&self.context.device, self.size);
        Ok(())
    }

    pub fn resize(&mut self, new_size: (u32, u32)) {
        if new_size.0 == 0 || new_size.1 == 0 {
            return;
        }
        self.size = new_size;
        self.presentation.resize(&self.context.device, new_size);
    }

    /// Scene-pass target priority: this, then the swapchain view.
    pub fn scene_view(&self) -> Option<&TextureView> {
        self.presentation.scene_target.as_ref().map(|t| &t.view)
    }

    /// Pipeline constructors take this, not the surface format.
    pub fn target_format(&self) -> TextureFormat {
        self.presentation
            .spec
            .scene_format
            .unwrap_or(self.presentation.spec.format)
    }

    pub fn composite_to_swap(&self, encoder: &mut wgpu::CommandEncoder, swap_view: &TextureView) {
        if let Some(composite) = self.presentation.composite.as_ref() {
            composite.run(encoder, swap_view);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: (u32, u32) = (800, 600);

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

    fn render_target(device: &Device, format: TextureFormat) -> TextureView {
        device
            .create_texture(&TextureDescriptor {
                label: Some("recovery target"),
                size: Extent3d {
                    width: SIZE.0,
                    height: SIZE.1,
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
