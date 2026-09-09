use std::cell::RefCell;
use std::rc::Rc;

use loam_render::device::{GpuContext, MissingGpuCapability};
use loam_render::pass::{FrameFormat, FramePass, FrameTarget, PassOrder, ResourceId, SCENE_COLOR};

use wgpu::{CommandEncoder, Device, Queue, TextureFormat};

use crate::TextRenderer;

// The last consumer of the composited colour; nothing reads what it paints, so it declares no write.
const READS: [ResourceId; 1] = [SCENE_COLOR];

/// One placement of the published text, in physical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextDraw {
    pub origin_px: [f32; 2],
    pub size_px: f32,
    pub color: [f32; 4],
}

struct State {
    font: Vec<u8>,
    bake_size_px: f32,
    format: TextureFormat,
    sample_count: u32,
    text: String,
    draws: Vec<TextDraw>,
    renderer: Option<TextRenderer>,
    device: Option<Device>,
    queue: Option<Queue>,
}

/// A `TextRenderer` after the scene that blends into scene colour with no depth test; clones share one state.
#[derive(Clone)]
pub struct TextPass {
    name: &'static str,
    shared: Rc<RefCell<State>>,
}

impl TextPass {
    pub fn new(name: &'static str, font: Vec<u8>, bake_size_px: f32) -> Self {
        Self {
            name,
            shared: Rc::new(RefCell::new(State {
                font,
                bake_size_px,
                format: TextureFormat::Rgba8UnormSrgb,
                sample_count: 1,
                text: String::new(),
                draws: Vec::new(),
                renderer: None,
                device: None,
                queue: None,
            })),
        }
    }

    /// The record queues `text` once per draw, in order, so a shadow placed first paints under the body.
    pub fn publish(&self, text: &str, draws: &[TextDraw]) {
        let mut state = self.shared.borrow_mut();
        state.text.clear();
        state.text.push_str(text);
        state.draws.clear();
        state.draws.extend_from_slice(draws);
    }

    /// False when the font never parsed, so the pass draws nothing.
    pub fn ready(&self) -> bool {
        self.shared.borrow().renderer.is_some()
    }
}

impl FramePass for TextPass {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reads(&self) -> &[ResourceId] {
        &READS
    }

    fn order(&self) -> PassOrder {
        PassOrder::AfterScene
    }

    fn record(&self, encoder: &mut CommandEncoder, target: &FrameTarget<'_>) {
        let mut state = self.shared.borrow_mut();
        if state.text.is_empty() || state.draws.is_empty() {
            return;
        }
        let State {
            text,
            draws,
            renderer,
            device,
            queue,
            ..
        } = &mut *state;
        let (Some(renderer), Some(device), Some(queue)) =
            (renderer.as_mut(), device.as_ref(), queue.as_ref())
        else {
            return;
        };
        for draw in draws.iter() {
            renderer.queue(text, draw.origin_px, draw.size_px, draw.color);
        }
        renderer.record(
            device,
            queue,
            encoder,
            target.color,
            [target.size.0 as f32, target.size.1 as f32],
        );
    }

    fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> Result<(), MissingGpuCapability> {
        {
            let mut state = self.shared.borrow_mut();
            state.format = frame.color;
            state.sample_count = frame.sample_count;
        }
        self.rebuild(gpu)
    }

    fn rebuild(&mut self, gpu: &GpuContext) -> Result<(), MissingGpuCapability> {
        let mut state = self.shared.borrow_mut();
        state.renderer = TextRenderer::new(
            &gpu.device,
            &gpu.queue,
            state.format,
            &state.font,
            state.bake_size_px,
            state.sample_count,
        )
        .ok();
        state.device = Some(gpu.device.clone());
        state.queue = Some(gpu.queue.clone());
        Ok(())
    }
}
