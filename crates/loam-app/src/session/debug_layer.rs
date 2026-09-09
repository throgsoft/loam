use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use loam_egui::egui;
use loam_render::device::{GpuContext, MissingGpuCapability};
use loam_render::pass::{FramePass, FrameTarget, PassOrder, ResourceId, SCENE_COLOR};
use wgpu::{CommandBuffer, CommandEncoder, Device, Queue, TextureFormat, TextureView};
use winit::window::Window;

pub const DEBUG_LAYER: ResourceId = "debug-layer";

const READS: [ResourceId; 1] = [SCENE_COLOR];
const WRITES: [ResourceId; 1] = [DEBUG_LAYER];

enum Feed {
    Winit(Box<loam_egui::WinitInput>),
    Raw {
        events: Vec<egui::Event>,
        modifiers: egui::Modifiers,
    },
}

struct Layer {
    ctx: egui::Context,
    renderer: Renderer,
    device: Device,
    queue: Queue,
    format: TextureFormat,
    sample_count: u32,
    feed: Feed,
    primitives: Vec<egui::ClippedPrimitive>,
    textures: egui::TexturesDelta,
    screen: ScreenDescriptor,
    callbacks: Vec<CommandBuffer>,
}

fn renderer(device: &Device, format: TextureFormat, sample_count: u32) -> Renderer {
    Renderer::new(
        device,
        format,
        RendererOptions {
            msaa_samples: sample_count,
            ..Default::default()
        },
    )
}

impl Layer {
    fn new(
        gpu: &GpuContext,
        format: TextureFormat,
        sample_count: u32,
        ctx: egui::Context,
        feed: Feed,
        size: (u32, u32),
        scale: f32,
    ) -> Self {
        Self {
            ctx,
            renderer: renderer(&gpu.device, format, sample_count),
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
            format,
            sample_count,
            feed,
            primitives: Vec::new(),
            textures: egui::TexturesDelta::default(),
            screen: ScreenDescriptor {
                size_in_pixels: [size.0.max(1), size.1.max(1)],
                pixels_per_point: scale,
            },
            callbacks: Vec::new(),
        }
    }

    fn raw_input(&mut self, window: Option<&Window>) -> egui::RawInput {
        let points = self.screen.pixels_per_point * self.ctx.zoom_factor();
        let size = self.screen.size_in_pixels;
        match (&mut self.feed, window) {
            (Feed::Winit(input), Some(window)) => input.take(window),
            (Feed::Winit(_), None) => egui::RawInput::default(),
            (Feed::Raw { events, modifiers }, _) => {
                let mut raw = egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(size[0] as f32 / points, size[1] as f32 / points),
                    )),
                    events: std::mem::take(events),
                    modifiers: *modifiers,
                    viewport_id: egui::ViewportId::ROOT,
                    ..egui::RawInput::default()
                };
                raw.viewports
                    .entry(egui::ViewportId::ROOT)
                    .or_default()
                    .native_pixels_per_point = Some(self.screen.pixels_per_point);
                raw
            }
        }
    }

    fn finish(&mut self, window: Option<&Window>) {
        let output = self.ctx.end_pass();
        if let (Feed::Winit(input), Some(window)) = (&mut self.feed, window) {
            input.handle_output(window, output.platform_output);
        }
        self.primitives.clear();
        self.primitives
            .extend(self.ctx.tessellate(output.shapes, output.pixels_per_point));
        self.textures = output.textures_delta;
        self.screen.pixels_per_point = output.pixels_per_point;
    }

    fn paint(&mut self, encoder: &mut CommandEncoder, target: &TextureView) {
        for (id, delta) in &self.textures.set {
            self.renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        self.callbacks.extend(self.renderer.update_buffers(
            &self.device,
            &self.queue,
            encoder,
            &self.primitives,
            &self.screen,
        ));
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("loam-app::session::debug-layer"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.renderer
                .render(&mut pass.forget_lifetime(), &self.primitives, &self.screen);
        }
        for id in &self.textures.free {
            self.renderer.free_texture(id);
        }
        self.textures.set.clear();
        self.textures.free.clear();
    }
}

/// An egui pass after the scene that reads scene colour and writes its own `DEBUG_LAYER` resource, so it records after every pass that writes the scene; it lives here because loam-egui does not depend on loam-render.
#[derive(Clone)]
pub struct DebugLayer {
    shared: Rc<RefCell<Layer>>,
    window: Option<Arc<Window>>,
}

impl DebugLayer {
    pub fn on_window(
        gpu: &GpuContext,
        format: TextureFormat,
        sample_count: u32,
        window: Arc<Window>,
        size: (u32, u32),
    ) -> Self {
        let ctx = egui::Context::default();
        let feed = Feed::Winit(Box::new(loam_egui::WinitInput::new(&ctx, window.as_ref())));
        let scale = window.scale_factor() as f32;
        let layer = Layer::new(gpu, format, sample_count, ctx, feed, size, scale);
        Self {
            shared: Rc::new(RefCell::new(layer)),
            window: Some(window),
        }
    }

    pub fn offscreen(
        gpu: &GpuContext,
        format: TextureFormat,
        sample_count: u32,
        size: (u32, u32),
        scale: f32,
    ) -> Self {
        let feed = Feed::Raw {
            events: Vec::new(),
            modifiers: egui::Modifiers::default(),
        };
        let layer = Layer::new(
            gpu,
            format,
            sample_count,
            egui::Context::default(),
            feed,
            size,
            scale,
        );
        Self {
            shared: Rc::new(RefCell::new(layer)),
            window: None,
        }
    }

    pub fn pass(&self) -> Box<dyn FramePass> {
        Box::new(LayerPass {
            shared: self.shared.clone(),
        })
    }

    pub fn context(&self) -> egui::Context {
        self.shared.borrow().ctx.clone()
    }

    pub fn resize(&self, width: u32, height: u32, scale: f32) {
        let mut layer = self.shared.borrow_mut();
        layer.screen.size_in_pixels = [width.max(1), height.max(1)];
        if scale.is_finite() && scale > 0.0 {
            layer.screen.pixels_per_point = scale;
        }
    }

    pub fn on_window_event(&self, event: &winit::event::WindowEvent) -> bool {
        let Some(window) = self.window.clone() else {
            return false;
        };
        let mut layer = self.shared.borrow_mut();
        let Feed::Winit(input) = &mut layer.feed else {
            return false;
        };
        input.on_event(window.as_ref(), event).consumed
    }

    pub fn push(&self, event: egui::Event) {
        if let Feed::Raw { events, .. } = &mut self.shared.borrow_mut().feed {
            events.push(event);
        }
    }

    pub fn modifiers(&self) -> egui::Modifiers {
        match &self.shared.borrow().feed {
            Feed::Raw { modifiers, .. } => *modifiers,
            Feed::Winit(_) => egui::Modifiers::default(),
        }
    }

    pub fn set_modifiers(&self, next: egui::Modifiers) {
        if let Feed::Raw { modifiers, .. } = &mut self.shared.borrow_mut().feed {
            *modifiers = next;
        }
    }

    pub fn begin(&self) -> egui::Context {
        let mut layer = self.shared.borrow_mut();
        let raw = layer.raw_input(self.window.as_deref());
        layer.ctx.begin_pass(raw);
        layer.ctx.clone()
    }

    pub fn finish(&self) {
        let window = self.window.clone();
        self.shared.borrow_mut().finish(window.as_deref());
    }

    pub fn take_callbacks(&self, into: &mut Vec<CommandBuffer>) {
        into.append(&mut self.shared.borrow_mut().callbacks);
    }
}

struct LayerPass {
    shared: Rc<RefCell<Layer>>,
}

impl FramePass for LayerPass {
    fn name(&self) -> &'static str {
        DEBUG_LAYER
    }

    fn reads(&self) -> &[ResourceId] {
        &READS
    }

    fn writes(&self) -> &[ResourceId] {
        &WRITES
    }

    fn order(&self) -> PassOrder {
        PassOrder::AfterScene
    }

    fn record(&self, encoder: &mut CommandEncoder, target: &FrameTarget<'_>) {
        self.shared.borrow_mut().paint(encoder, target.color);
    }

    fn rebuild(&mut self, gpu: &GpuContext) -> Result<(), MissingGpuCapability> {
        let mut layer = self.shared.borrow_mut();
        layer.device = gpu.device.clone();
        layer.queue = gpu.queue.clone();
        layer.renderer = renderer(&gpu.device, layer.format, layer.sample_count);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use loam_render::device::FeatureRequest;
    use loam_render::pass::{PassSchedule, SCENE_COLOR};
    use loam_render::DepthConvention;
    use wgpu::{BackendOptions, Backends, Instance, InstanceDescriptor, NoopBackendOptions};

    use super::*;

    const SCENE: [ResourceId; 1] = [SCENE_COLOR];

    struct Overlay;

    impl FramePass for Overlay {
        fn name(&self) -> &'static str {
            "overlay"
        }

        fn reads(&self) -> &[ResourceId] {
            &SCENE
        }

        fn writes(&self) -> &[ResourceId] {
            &SCENE
        }

        fn order(&self) -> PassOrder {
            PassOrder::AfterScene
        }

        fn record(&self, _encoder: &mut CommandEncoder, _target: &FrameTarget<'_>) {}

        fn rebuild(&mut self, _gpu: &GpuContext) -> Result<(), MissingGpuCapability> {
            Ok(())
        }
    }

    fn noop_gpu() -> GpuContext {
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

    #[test]
    fn an_application_pass_over_the_scene_records_before_the_debug_layer() {
        let gpu = noop_gpu();
        let layer = DebugLayer::offscreen(&gpu, TextureFormat::Rgba8UnormSrgb, 1, (16, 16), 1.0);
        for layer_first in [true, false] {
            let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
            let passes: [Box<dyn FramePass>; 2] = if layer_first {
                [layer.pass(), Box::new(Overlay)]
            } else {
                [Box::new(Overlay), layer.pass()]
            };
            for pass in passes {
                schedule.register(pass).expect("registered");
            }
            assert_eq!(
                schedule.names().collect::<Vec<_>>(),
                ["overlay", DEBUG_LAYER],
                "an application pass over the scene paints over the debug layer"
            );
        }
    }
}
