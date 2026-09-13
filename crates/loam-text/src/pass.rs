use std::cell::RefCell;
use std::rc::Rc;

use loam_render::device::GpuContext;
use loam_render::pass::{FrameFormat, FramePass, FrameTarget, PassStage, ResourceId, SCENE_COLOR};

use wgpu::{CommandEncoder, Device, Queue};

use crate::TextRenderer;

const READS: [ResourceId; 1] = [SCENE_COLOR];
const WRITES: [ResourceId; 1] = [SCENE_COLOR];

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
    text: String,
    draws: Vec<TextDraw>,
    renderer: Option<TextRenderer>,
    device: Option<Device>,
    queue: Option<Queue>,
}

/// A shared text overlay pass without a depth test.
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

    fn writes(&self) -> &[ResourceId] {
        &WRITES
    }

    fn stage(&self) -> PassStage {
        PassStage::Overlay
    }

    fn record(&self, encoder: &mut CommandEncoder, target: &FrameTarget<'_>) -> anyhow::Result<()> {
        let mut state = self.shared.borrow_mut();
        if state.text.is_empty() || state.draws.is_empty() {
            return Ok(());
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
            return Ok(());
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
        Ok(())
    }

    fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> anyhow::Result<()> {
        let mut state = self.shared.borrow_mut();
        state.renderer = Some(TextRenderer::new(
            &gpu.device,
            &gpu.queue,
            frame.color,
            &state.font,
            state.bake_size_px,
        )?);
        state.device = Some(gpu.device.clone());
        state.queue = Some(gpu.queue.clone());
        Ok(())
    }
}
