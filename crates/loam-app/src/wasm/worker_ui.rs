//! egui-winit requires a Window; workers receive DOM input through messages.

use loam_egui::egui;

use super::input_queue::InputMessage;

pub struct WorkerUi {
    ctx: egui::Context,
    renderer: egui_wgpu::Renderer,
    raw_events: Vec<egui::Event>,
    modifiers: egui::Modifiers,
    width_px: u32,
    height_px: u32,
    pixels_per_point: f32,
}

impl WorkerUi {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        width_px: u32,
        height_px: u32,
        pixels_per_point: f32,
    ) -> Self {
        let ctx = egui::Context::default();
        let renderer = egui_wgpu::Renderer::new(
            device,
            target_format,
            egui_wgpu::RendererOptions {
                msaa_samples: crate::UI_PASS_SAMPLE_COUNT,
                ..Default::default()
            },
        );
        Self {
            ctx,
            renderer,
            raw_events: Vec::new(),
            modifiers: egui::Modifiers::default(),
            width_px,
            height_px,
            pixels_per_point,
        }
    }

    pub fn record_input(&mut self, msg: &InputMessage) {
        match msg {
            InputMessage::MouseMove { x, y, .. } => {
                let pos = egui::pos2(*x, *y) / self.ctx.zoom_factor();
                self.raw_events.push(egui::Event::PointerMoved(pos));
            }
            InputMessage::MouseButton {
                x,
                y,
                button,
                pressed,
            } => {
                let pos = egui::pos2(*x, *y) / self.ctx.zoom_factor();
                if let Some(b) = crate::keymap::mouse_button_egui(*button) {
                    self.raw_events.push(egui::Event::PointerButton {
                        pos,
                        button: b,
                        pressed: *pressed,
                        modifiers: self.modifiers,
                    });
                }
            }
            InputMessage::MouseWheel { dx, dy } => {
                self.raw_events.push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Line,
                    delta: egui::vec2(-*dx, -*dy), // egui convention: up = +y
                    modifiers: self.modifiers,
                });
            }
            InputMessage::Key {
                code,
                key,
                pressed,
                repeat,
                ctrl,
                shift,
                alt,
                meta,
            } => {
                self.modifiers = egui::Modifiers {
                    alt: *alt,
                    ctrl: *ctrl,
                    shift: *shift,
                    mac_cmd: *meta,
                    command: *ctrl || *meta,
                };
                if let Some(egui_key) = crate::keymap::keycode_egui(code) {
                    self.raw_events.push(egui::Event::Key {
                        key: egui_key,
                        physical_key: Some(egui_key),
                        pressed: *pressed,
                        repeat: *repeat,
                        modifiers: self.modifiers,
                    });
                }
                if *pressed
                    && !*ctrl
                    && !*alt
                    && !*meta
                    && key.chars().count() == 1
                    && !key.starts_with(char::is_control)
                {
                    self.raw_events.push(egui::Event::Text(key.clone()));
                }
            }
            InputMessage::Focus(focused) => {
                if !focused {
                    self.modifiers = egui::Modifiers::default();
                }
                self.raw_events.push(egui::Event::WindowFocused(*focused));
            }
            // Handled outside egui: runner, cursor mirror, frame-loop entry.
            InputMessage::Resize { .. }
            | InputMessage::Visibility(_)
            | InputMessage::Start
            | InputMessage::PointerLockChanged(_) => {}
        }
    }

    pub fn begin_frame(&mut self) -> &egui::Context {
        let mut raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(
                    self.width_px as f32 / (self.pixels_per_point * self.ctx.zoom_factor()),
                    self.height_px as f32 / (self.pixels_per_point * self.ctx.zoom_factor()),
                ),
            )),
            events: std::mem::take(&mut self.raw_events),
            modifiers: self.modifiers,
            viewport_id: egui::ViewportId::ROOT,
            time: None,
            ..egui::RawInput::default()
        };
        raw_input
            .viewports
            .entry(egui::ViewportId::ROOT)
            .or_default()
            .native_pixels_per_point = Some(self.pixels_per_point);
        self.ctx.begin_pass(raw_input);
        &self.ctx
    }

    /// Loads a single-sample view; submit returned callback buffers before the finished encoder.
    pub fn paint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) -> Vec<wgpu::CommandBuffer> {
        let full_output = self.ctx.end_pass();

        let primitives = self
            .ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.width_px, self.height_px],
            pixels_per_point: full_output.pixels_per_point,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }
        let callbacks = self
            .renderer
            .update_buffers(device, queue, encoder, &primitives, &screen);

        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("loam_app::wasm::worker::egui-paint"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            // egui-wgpu requires a static render pass.
            self.renderer
                .render(&mut pass.forget_lifetime(), &primitives, &screen);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
        callbacks
    }

    pub fn resize(&mut self, width: u32, height: u32, dpr: f32) {
        self.width_px = width;
        self.height_px = height;
        self.pixels_per_point = dpr;
    }
}
