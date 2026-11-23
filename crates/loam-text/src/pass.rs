use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;

use loam_render::device::GpuContext;
use loam_render::pass::{FrameFormat, FramePass, FrameTarget, PassStage};

use wgpu::{CommandEncoder, Device, Queue};

use crate::TextRenderer;

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
    draws: Vec<(Range<usize>, TextDraw)>,
    renderer: Option<TextRenderer>,
    failure: Option<String>,
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
                failure: None,
                device: None,
                queue: None,
            })),
        }
    }

    /// Draws in order, so a shadow placed before its body paints under it.
    pub fn publish<'a>(&self, draws: impl IntoIterator<Item = (&'a str, TextDraw)>) {
        let mut state = self.shared.borrow_mut();
        let State {
            text, draws: kept, ..
        } = &mut *state;
        text.clear();
        kept.clear();
        for (part, draw) in draws {
            let start = text.len();
            text.push_str(part);
            kept.push((start..text.len(), draw));
        }
    }

    /// False when the font never parsed, so the pass draws nothing.
    pub fn ready(&self) -> bool {
        self.shared.borrow().renderer.is_some()
    }

    pub fn failure(&self) -> Option<String> {
        self.shared.borrow().failure.clone()
    }
}

impl FramePass for TextPass {
    fn name(&self) -> &'static str {
        self.name
    }

    fn stage(&self) -> PassStage {
        PassStage::Overlay
    }

    fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
        let mut state = self.shared.borrow_mut();
        if state.draws.is_empty() {
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
        for (range, draw) in draws.iter() {
            renderer.queue(
                &text[range.clone()],
                draw.origin_px,
                draw.size_px,
                draw.color,
            );
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
        state.renderer = None;
        match TextRenderer::new(
            &gpu.device,
            &gpu.queue,
            frame.color,
            &state.font,
            state.bake_size_px,
        ) {
            Ok(renderer) => {
                state.renderer = Some(renderer);
                state.failure = None;
            }
            Err(error) => {
                let failure = format!("{error:#}");
                tracing::warn!("pass `{}` has no font: {failure}", self.name);
                state.failure = Some(failure);
            }
        }
        state.device = Some(gpu.device.clone());
        state.queue = Some(gpu.queue.clone());
        Ok(())
    }
}
