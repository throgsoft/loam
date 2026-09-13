use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use loam_egui::egui;
use loam_render::device::GpuContext;
use loam_render::pass::{FrameFormat, FramePass, FrameTarget, PassStage, ResourceId, SCENE_COLOR};
use wgpu::{CommandBuffer, CommandEncoder, Device, Queue, TextureFormat, TextureView};
use winit::window::Window;

#[cfg(any(target_arch = "wasm32", test))]
use super::input::TouchCapture;
use super::winit_input::WinitInput;

pub const DEBUG_LAYER: ResourceId = "debug-layer";

const READS: [ResourceId; 1] = [SCENE_COLOR];
const WRITES: [ResourceId; 1] = [DEBUG_LAYER];

enum Feed {
    Winit(Box<WinitInput>),
    Raw(RawFeed),
}

struct RawFeed {
    events: Vec<egui::Event>,
    deferred: Vec<egui::Event>,
    modifiers: egui::Modifiers,
    cancellation_pass: bool,
    #[cfg(any(target_arch = "wasm32", test))]
    reset_after_pass: bool,
}

impl RawFeed {
    fn push(&mut self, event: egui::Event) {
        let delayed = matches!(
            event,
            egui::Event::PointerMoved(_)
                | egui::Event::PointerButton { .. }
                | egui::Event::PointerGone
                | egui::Event::Touch { .. }
        );
        if self.cancellation_pass && delayed {
            self.deferred.push(event);
        } else {
            self.events.push(event);
        }
    }

    #[cfg(any(target_arch = "wasm32", test))]
    fn cancel_touch(&mut self, id: u64, pos: egui::Pos2, pointer: bool) -> bool {
        let id = egui::TouchId::from(id);
        let deferred_start = self.deferred.iter().any(|event| {
            matches!(
                event,
                egui::Event::Touch {
                    id: event_id,
                    phase: egui::TouchPhase::Start,
                    ..
                } if *event_id == id
            )
        });
        let completed_prefix = if pointer {
            if deferred_start {
                remove_pointer_gesture(&mut self.deferred, id)
            } else {
                let completed = remove_pointer_gesture(&mut self.events, id);
                self.deferred.retain(|event| !is_pointer_event(event));
                completed
            }
        } else {
            false
        };
        self.cancellation_pass |= pointer;
        self.deferred.retain(
            |event| !matches!(event, egui::Event::Touch { id: event_id, .. } if *event_id == id),
        );
        if !deferred_start {
            self.events.push(egui::Event::Touch {
                device_id: egui::TouchDeviceId(0),
                id,
                phase: egui::TouchPhase::Cancel,
                pos,
                force: None,
            });
        }
        self.reset_before_pass(pointer, completed_prefix)
    }

    #[cfg(any(target_arch = "wasm32", test))]
    fn cancel_pointer(&mut self) -> bool {
        let completed_prefix = retain_completed_pointer_prefix(&mut self.events);
        retain_completed_pointer_prefix(&mut self.deferred);
        self.cancellation_pass = true;
        self.reset_before_pass(true, completed_prefix)
    }

    #[cfg(any(target_arch = "wasm32", test))]
    fn reset_before_pass(&mut self, pointer: bool, completed_prefix: bool) -> bool {
        if !pointer {
            return false;
        }
        if completed_prefix {
            self.reset_after_pass = true;
            return false;
        }
        !self.reset_after_pass
    }

    #[cfg(any(target_arch = "wasm32", test))]
    fn take_reset_after_pass(&mut self) -> bool {
        std::mem::take(&mut self.reset_after_pass)
    }

    fn take(&mut self) -> Vec<egui::Event> {
        let current = std::mem::take(&mut self.events);
        if self.cancellation_pass {
            // egui 0.33.3 interaction.rs clears click ownership only in a pass with no pointer down.
            self.cancellation_pass = false;
            self.events.append(&mut self.deferred);
        }
        current
    }
}

#[cfg(any(target_arch = "wasm32", test))]
fn is_pointer_event(event: &egui::Event) -> bool {
    matches!(
        event,
        egui::Event::PointerMoved(_) | egui::Event::PointerButton { .. } | egui::Event::PointerGone
    )
}

#[cfg(any(target_arch = "wasm32", test))]
fn remove_pointer_gesture(events: &mut Vec<egui::Event>, id: egui::TouchId) -> bool {
    let start = events.iter().rposition(|event| {
        matches!(
            event,
            egui::Event::Touch {
                id: event_id,
                phase: egui::TouchPhase::Start,
                ..
            } if *event_id == id
        )
    });
    let completed_prefix = start.is_some_and(|start| events[..start].iter().any(primary_release));
    let mut index = 0;
    events.retain(|event| {
        let keep = !is_pointer_event(event) || start.is_some_and(|start| index < start);
        index += 1;
        keep
    });
    completed_prefix
}

#[cfg(any(target_arch = "wasm32", test))]
fn retain_completed_pointer_prefix(events: &mut Vec<egui::Event>) -> bool {
    let release = events.iter().rposition(primary_release);
    let mut index = 0;
    events.retain(|event| {
        let keep = !is_pointer_event(event) || release.is_some_and(|release| index <= release);
        index += 1;
        keep
    });
    release.is_some()
}

#[cfg(any(target_arch = "wasm32", test))]
fn primary_release(event: &egui::Event) -> bool {
    matches!(
        event,
        egui::Event::PointerButton {
            button: egui::PointerButton::Primary,
            pressed: false,
            ..
        }
    )
}

#[cfg(any(target_arch = "wasm32", test))]
fn reset_pointer(context: &egui::Context) {
    context.input_mut(|input| input.pointer = egui::PointerState::default());
    context.stop_dragging();
    egui::DragAndDrop::clear_payload(context);
}

struct ManagedTexture {
    image: egui::ImageData,
    options: egui::TextureOptions,
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
    native_pixels_per_point: f32,
    callbacks: Vec<CommandBuffer>,
    managed_textures: HashMap<egui::TextureId, ManagedTexture>,
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
            native_pixels_per_point: scale,
            callbacks: Vec::new(),
            managed_textures: HashMap::new(),
        }
    }

    fn raw_input(&mut self, window: Option<&Window>) -> egui::RawInput {
        let points = self.native_pixels_per_point * self.ctx.zoom_factor();
        let size = self.screen.size_in_pixels;
        match (&mut self.feed, window) {
            (Feed::Winit(input), Some(window)) => input.take(window),
            (Feed::Winit(_), None) => egui::RawInput::default(),
            (Feed::Raw(feed), _) => {
                let mut raw = egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(size[0] as f32 / points, size[1] as f32 / points),
                    )),
                    events: feed.take(),
                    modifiers: feed.modifiers,
                    viewport_id: egui::ViewportId::ROOT,
                    ..egui::RawInput::default()
                };
                raw.viewports
                    .entry(egui::ViewportId::ROOT)
                    .or_default()
                    .native_pixels_per_point = Some(self.native_pixels_per_point);
                raw
            }
        }
    }

    fn finish(&mut self, window: Option<&Window>) {
        let output = self.ctx.end_pass();
        #[cfg(any(target_arch = "wasm32", test))]
        if let Feed::Raw(feed) = &mut self.feed {
            if feed.take_reset_after_pass() {
                reset_pointer(&self.ctx);
            }
        }
        if let (Feed::Winit(input), Some(window)) = (&mut self.feed, window) {
            input.handle_output(window, output.platform_output);
        }
        self.primitives.clear();
        self.primitives
            .extend(self.ctx.tessellate(output.shapes, output.pixels_per_point));
        self.textures = output.textures_delta;
        self.screen.pixels_per_point = output.pixels_per_point;
    }

    fn paint(&mut self, encoder: &mut CommandEncoder, target: &TextureView) -> anyhow::Result<()> {
        for (id, delta) in &self.textures.set {
            retain_managed_texture(&mut self.managed_textures, *id, delta)?;
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
            self.managed_textures.remove(id);
        }
        self.textures.set.clear();
        self.textures.free.clear();
        Ok(())
    }
}

fn retain_managed_texture(
    textures: &mut HashMap<egui::TextureId, ManagedTexture>,
    id: egui::TextureId,
    delta: &egui::epaint::ImageDelta,
) -> anyhow::Result<()> {
    if !matches!(id, egui::TextureId::Managed(_)) {
        return Ok(());
    }
    let Some(pos) = delta.pos else {
        textures.insert(
            id,
            ManagedTexture {
                image: delta.image.clone(),
                options: delta.options,
            },
        );
        return Ok(());
    };
    let texture = textures
        .get_mut(&id)
        .ok_or_else(|| anyhow::anyhow!("managed texture {id:?} has no full image"))?;
    let egui::ImageData::Color(image) = &mut texture.image;
    let egui::ImageData::Color(patch) = &delta.image;
    let end_x = pos[0]
        .checked_add(patch.size[0])
        .filter(|end| *end <= image.size[0]);
    let end_y = pos[1]
        .checked_add(patch.size[1])
        .filter(|end| *end <= image.size[1]);
    if end_x.is_none() || end_y.is_none() {
        anyhow::bail!("managed texture {id:?} partial update is out of bounds");
    }
    let image = Arc::make_mut(image);
    for row in 0..patch.size[1] {
        let source = row * patch.size[0];
        let target = (pos[1] + row) * image.size[0] + pos[0];
        image.pixels[target..target + patch.size[0]]
            .copy_from_slice(&patch.pixels[source..source + patch.size[0]]);
    }
    texture.options = delta.options;
    Ok(())
}

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
        let feed = Feed::Winit(Box::new(WinitInput::new(&ctx, window.as_ref())));
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
        let feed = Feed::Raw(RawFeed {
            events: Vec::new(),
            deferred: Vec::new(),
            modifiers: egui::Modifiers::default(),
            cancellation_pass: false,
            #[cfg(any(target_arch = "wasm32", test))]
            reset_after_pass: false,
        });
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
            layer.native_pixels_per_point = scale;
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
        if let Feed::Raw(feed) = &mut self.shared.borrow_mut().feed {
            feed.push(event);
        }
    }

    #[cfg(any(target_arch = "wasm32", test))]
    pub(super) fn cancel_touch(&self, id: u64, pos: egui::Pos2, pointer: bool) {
        let (context, reset_before_pass) = {
            let mut layer = self.shared.borrow_mut();
            let Feed::Raw(feed) = &mut layer.feed else {
                return;
            };
            let reset_before_pass = feed.cancel_touch(id, pos, pointer);
            (layer.ctx.clone(), reset_before_pass)
        };
        if reset_before_pass {
            reset_pointer(&context);
        }
    }

    #[cfg(any(target_arch = "wasm32", test))]
    pub(super) fn cancel_touches(&self, touches: &mut TouchCapture) {
        for (id, pos) in touches.cancel_all() {
            self.cancel_touch(id, egui::pos2(pos[0], pos[1]), false);
        }
        self.cancel_pointer();
    }

    #[cfg(any(target_arch = "wasm32", test))]
    fn cancel_pointer(&self) {
        let (context, reset_before_pass) = {
            let mut layer = self.shared.borrow_mut();
            let Feed::Raw(feed) = &mut layer.feed else {
                return;
            };
            let reset_before_pass = feed.cancel_pointer();
            (layer.ctx.clone(), reset_before_pass)
        };
        if reset_before_pass {
            reset_pointer(&context);
        }
    }

    pub fn modifiers(&self) -> egui::Modifiers {
        match &self.shared.borrow().feed {
            Feed::Raw(feed) => feed.modifiers,
            Feed::Winit(_) => egui::Modifiers::default(),
        }
    }

    pub fn set_modifiers(&self, next: egui::Modifiers) {
        if let Feed::Raw(feed) = &mut self.shared.borrow_mut().feed {
            feed.modifiers = next;
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

    fn stage(&self) -> PassStage {
        PassStage::Overlay
    }

    fn record(&self, encoder: &mut CommandEncoder, target: &FrameTarget<'_>) -> anyhow::Result<()> {
        self.shared.borrow_mut().paint(encoder, target.color)
    }

    fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> anyhow::Result<()> {
        let mut layer = self.shared.borrow_mut();
        layer.device = gpu.device.clone();
        layer.queue = gpu.queue.clone();
        layer.format = frame.color;
        layer.sample_count = frame.sample_count;
        layer.renderer = renderer(&gpu.device, frame.color, frame.sample_count);
        let Layer {
            renderer,
            managed_textures,
            ..
        } = &mut *layer;
        for (id, texture) in managed_textures {
            renderer.update_texture(
                &gpu.device,
                &gpu.queue,
                *id,
                &egui::epaint::ImageDelta::full(texture.image.clone(), texture.options),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::wasm::input_queue::PointerPhase;
    use loam_render::device::FeatureRequest;
    use loam_render::pass::PassSchedule;
    use loam_render::{
        DepthConvention, FragmentShading, Ground, HyperslicePass, LinePass, SkyGroundPass,
        TriangleFeed,
    };
    use wgpu::{
        BackendOptions, Backends, Extent3d, Instance, InstanceDescriptor, NoopBackendOptions,
        TextureDescriptor, TextureDimension, TextureUsages, TextureViewDescriptor,
    };

    use super::*;

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
    fn scene_and_overlay_passes_keep_capture_and_compositing_order() {
        let gpu = noop_gpu();
        let layer = DebugLayer::offscreen(&gpu, TextureFormat::Rgba8UnormSrgb, 1, (16, 16), 1.0);
        let text = loam_text::TextPass::new("text", Vec::new(), 16.0);
        let triangle = TriangleFeed::default().pass(FragmentShading::FaceNormalLambert);
        let ground = Ground {
            y: 0.0,
            dark: [0.1, 0.1, 0.1],
            light: [0.2, 0.2, 0.2],
            fog_per_unit: 0.0,
            visible: true,
        };
        let passes: [Box<dyn FramePass>; 6] = [
            layer.pass(),
            Box::new(text),
            Box::new(LinePass::new("lines")),
            triangle,
            Box::new(HyperslicePass::new(String::new())),
            Box::new(SkyGroundPass::new(ground)),
        ];
        let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
        for pass in passes {
            schedule.register(pass).expect("registered");
        }
        assert_eq!(
            schedule.names().collect::<Vec<_>>(),
            [
                "sky-ground",
                "triangles",
                "hyperslice",
                "lines",
                "text",
                DEBUG_LAYER,
            ]
        );
    }

    #[test]
    fn browser_zoom_does_not_feed_output_scale_back_as_native_dpr() {
        let gpu = noop_gpu();
        let layer = DebugLayer::offscreen(&gpu, TextureFormat::Rgba8UnormSrgb, 1, (600, 300), 2.0);
        let mut layer = layer.shared.borrow_mut();
        layer.ctx.set_zoom_factor(1.5);

        let first = layer.raw_input(None);
        layer.ctx.begin_pass(first);
        layer.finish(None);
        let second = layer.raw_input(None);

        assert_eq!(layer.native_pixels_per_point, 2.0);
        assert_eq!(layer.screen.pixels_per_point, 3.0);
        assert_eq!(
            second.screen_rect.map(|rect| rect.size()),
            Some(egui::vec2(200.0, 100.0))
        );
    }

    #[test]
    fn canceled_browser_touch_clears_drag_without_releasing_and_keeps_later_click() {
        struct Observed {
            a: egui::Rect,
            b: egui::Rect,
            c: egui::Rect,
            clicked_a: bool,
            clicked_b: bool,
            clicked_c: bool,
            pressed: usize,
            released: usize,
            any_released: bool,
            down: bool,
            touches: bool,
            payload: bool,
        }

        fn observe(layer: &DebugLayer) -> Observed {
            let context = layer.begin();
            let mut a = egui::Rect::NOTHING;
            let mut b = egui::Rect::NOTHING;
            let mut c = egui::Rect::NOTHING;
            let mut clicked_a = false;
            let mut clicked_b = false;
            let mut clicked_c = false;
            egui::CentralPanel::default().show(&context, |ui| {
                let (_, response) =
                    ui.allocate_exact_size(egui::vec2(120.0, 48.0), egui::Sense::click_and_drag());
                a = response.rect;
                clicked_a = response.clicked();
                if response.is_pointer_button_down_on() {
                    context.set_dragged_id(response.id);
                    egui::DragAndDrop::set_payload(&context, 7_u32);
                }
                let (_, response) =
                    ui.allocate_exact_size(egui::vec2(120.0, 48.0), egui::Sense::click());
                b = response.rect;
                clicked_b = response.clicked();
                let (_, response) =
                    ui.allocate_exact_size(egui::vec2(120.0, 48.0), egui::Sense::click());
                c = response.rect;
                clicked_c = response.clicked();
            });
            let (pressed, released, any_released, down, touches) = context.input(|input| {
                let mut pressed = 0;
                let mut released = 0;
                for event in &input.raw.events {
                    if let egui::Event::PointerButton { pressed: state, .. } = event {
                        if *state {
                            pressed += 1;
                        } else {
                            released += 1;
                        }
                    }
                }
                (
                    pressed,
                    released,
                    input.pointer.any_released(),
                    input.pointer.any_down(),
                    input.any_touches(),
                )
            });
            let payload = egui::DragAndDrop::has_any_payload(&context);
            layer.finish();
            Observed {
                a,
                b,
                c,
                clicked_a,
                clicked_b,
                clicked_c,
                pressed,
                released,
                any_released,
                down,
                touches,
                payload,
            }
        }

        fn down(layer: &DebugLayer, touches: &mut TouchCapture, id: u64, pos: egui::Pos2) {
            assert!(touches.route(id, PointerPhase::Down, true, [pos.x, pos.y]));
            layer.push(egui::Event::Touch {
                device_id: egui::TouchDeviceId(0),
                id: egui::TouchId::from(id),
                phase: egui::TouchPhase::Start,
                pos,
                force: None,
            });
            layer.push(egui::Event::PointerMoved(pos));
            layer.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::default(),
            });
        }

        fn cancel(layer: &DebugLayer, touches: &mut TouchCapture, id: u64, pos: egui::Pos2) {
            let pointer = touches.is_pointer(id);
            assert!(touches.route(id, PointerPhase::Cancel, false, [pos.x, pos.y]));
            layer.cancel_touch(id, pos, pointer);
        }

        fn up(layer: &DebugLayer, touches: &mut TouchCapture, id: u64, pos: egui::Pos2) {
            let pointer = touches.is_pointer(id);
            assert!(touches.route(id, PointerPhase::Up, false, [pos.x, pos.y]));
            layer.push(egui::Event::Touch {
                device_id: egui::TouchDeviceId(0),
                id: egui::TouchId::from(id),
                phase: egui::TouchPhase::End,
                pos,
                force: None,
            });
            if pointer {
                layer.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::default(),
                });
                layer.push(egui::Event::PointerGone);
            }
        }

        let gpu = noop_gpu();
        let layer = DebugLayer::offscreen(&gpu, TextureFormat::Rgba8UnormSrgb, 1, (256, 192), 1.0);
        let mut touches = TouchCapture::default();
        let initial = observe(&layer);

        down(&layer, &mut touches, 1, initial.a.center());
        let held_a = observe(&layer);
        assert!(!held_a.clicked_a && !held_a.clicked_b && !held_a.clicked_c);
        assert!(held_a.down && held_a.touches && held_a.payload);

        layer.cancel_touches(&mut touches);
        down(&layer, &mut touches, 2, initial.c.center());
        cancel(&layer, &mut touches, 2, initial.c.center());
        down(&layer, &mut touches, 3, initial.b.center());
        up(&layer, &mut touches, 3, initial.b.center());

        let canceled = observe(&layer);
        assert!(!canceled.clicked_a && !canceled.clicked_b && !canceled.clicked_c);
        assert_eq!((canceled.pressed, canceled.released), (0, 0));
        assert!(!canceled.any_released);
        assert!(!canceled.down && !canceled.touches && !canceled.payload);

        let valid_b = observe(&layer);
        assert!(!valid_b.clicked_a && valid_b.clicked_b && !valid_b.clicked_c);
        assert_eq!((valid_b.pressed, valid_b.released), (1, 1));
        assert!(valid_b.any_released);
        assert!(!valid_b.down && !valid_b.touches && !valid_b.payload);
        down(&layer, &mut touches, 4, initial.a.center());
        let held_a = observe(&layer);
        assert!(!held_a.clicked_a && !held_a.clicked_b && !held_a.clicked_c);
        assert!(held_a.down && held_a.touches && held_a.payload);

        up(&layer, &mut touches, 4, initial.a.center());
        down(&layer, &mut touches, 5, initial.c.center());
        cancel(&layer, &mut touches, 5, initial.c.center());
        down(&layer, &mut touches, 6, initial.b.center());
        up(&layer, &mut touches, 6, initial.b.center());

        let valid_a = observe(&layer);
        assert!(valid_a.clicked_a && !valid_a.clicked_b && !valid_a.clicked_c);
        assert_eq!((valid_a.pressed, valid_a.released), (0, 1));
        assert!(valid_a.any_released);
        assert!(!valid_a.down && !valid_a.touches);

        let valid_b = observe(&layer);
        assert!(!valid_b.clicked_a && valid_b.clicked_b && !valid_b.clicked_c);
        assert_eq!((valid_b.pressed, valid_b.released), (1, 1));
        assert!(valid_b.any_released);
        assert!(!valid_b.down && !valid_b.touches && !valid_b.payload);
        assert!(!touches.is_pointer(6));
        assert_eq!(touches.cancel_all().count(), 0);
    }

    #[test]
    fn managed_partial_texture_after_recovery_uses_replay_and_free_removes_it() {
        let gpu = noop_gpu();
        let format = TextureFormat::Rgba8UnormSrgb;
        let layer = DebugLayer::offscreen(&gpu, format, 1, (16, 16), 1.0);
        let target = gpu.device.create_texture(&TextureDescriptor {
            label: Some("debug layer texture recovery probe"),
            size: Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target = target.create_view(&TextureViewDescriptor::default());
        let paint = || {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            layer
                .shared
                .borrow_mut()
                .paint(&mut encoder, &target)
                .expect("texture deltas applied");
            gpu.queue.submit(Some(encoder.finish()));
        };
        let red = egui::Color32::from_rgb(255, 0, 0);
        let blue = egui::Color32::from_rgb(0, 0, 255);
        let mut texture = layer.context().load_texture(
            "managed recovery probe",
            egui::ColorImage::filled([2, 2], red),
            egui::TextureOptions::LINEAR,
        );
        let id = texture.id();
        layer.begin();
        layer.finish();
        paint();

        let frame = FrameFormat {
            color: format,
            depth: TextureFormat::Depth32Float,
            sample_count: 1,
        };
        let mut pass = layer.pass();
        pass.attach(&gpu, frame).expect("renderer recreated");
        texture.set_partial(
            [1, 0],
            egui::ColorImage::filled([1, 2], blue),
            egui::TextureOptions::NEAREST,
        );
        layer.begin();
        layer.finish();
        paint();

        {
            let layer = layer.shared.borrow();
            let cached = layer.managed_textures.get(&id).expect("texture retained");
            let egui::ImageData::Color(image) = &cached.image;
            assert_eq!(image.pixels, [red, blue, red, blue]);
            assert_eq!(cached.options, egui::TextureOptions::NEAREST);
        }

        drop(texture);
        layer.begin();
        layer.finish();
        paint();
        pass.attach(&gpu, frame)
            .expect("renderer recreated after free");
        assert!(!layer.shared.borrow().managed_textures.contains_key(&id));
    }
}
