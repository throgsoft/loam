use std::cell::RefCell;
use std::rc::Rc;

use glam::Vec2;
use loam_runtime::{Eye, SegmentRecord};
use wgpu::{CommandEncoder, Device, Queue, RenderPass};

use crate::device::GpuContext;
use crate::pass::{FrameFormat, FramePass, FrameTarget, PassStage, PublishedFrame};
use crate::{DepthConvention, DepthMode, LineRasterNode};

struct State {
    eye: Eye,
    segments: Vec<SegmentRecord>,
    uploaded: bool,
    built: Option<Built>,
}

struct Built {
    node: LineRasterNode,
    device: Device,
    queue: Queue,
}

#[derive(Clone)]
pub struct LinePass {
    name: &'static str,
    reads_overlay: bool,
    shared: Rc<RefCell<State>>,
}

impl LinePass {
    pub fn new(name: &'static str) -> Self {
        Self::reading(name, false)
    }

    pub(crate) fn overlay(name: &'static str) -> Self {
        Self::reading(name, true)
    }

    fn reading(name: &'static str, reads_overlay: bool) -> Self {
        Self {
            name,
            reads_overlay,
            shared: Rc::new(RefCell::new(State {
                eye: Eye::default(),
                segments: Vec::new(),
                uploaded: false,
                built: None,
            })),
        }
    }

    pub fn publish(&self, eye: &Eye, segments: &[SegmentRecord]) {
        self.shared.borrow_mut().set(eye, segments);
    }

    pub fn segment_count(&self) -> usize {
        self.shared.borrow().segments.len()
    }
}

impl State {
    fn set(&mut self, eye: &Eye, segments: &[SegmentRecord]) {
        self.eye = *eye;
        self.uploaded &= self.segments == segments;
        self.segments.clear();
        self.segments.extend_from_slice(segments);
    }
}

impl FramePass for LinePass {
    fn name(&self) -> &'static str {
        self.name
    }

    fn stage(&self) -> PassStage {
        PassStage::Scene
    }

    fn depth_read(&self) -> Option<DepthConvention> {
        Some(DepthConvention::ReversedZ)
    }

    fn shares_pass(&self) -> bool {
        true
    }

    fn read(&mut self, frame: &PublishedFrame<'_>) {
        if self.reads_overlay {
            self.shared
                .borrow_mut()
                .set(&frame.eye, &frame.overlay.lines);
        }
    }

    fn prepare(
        &mut self,
        _encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
        let mut state = self.shared.borrow_mut();
        let State {
            eye,
            segments,
            uploaded,
            built,
        } = &mut *state;
        let Some(Built {
            node,
            device,
            queue,
        }) = built.as_mut()
        else {
            return Ok(());
        };
        node.set_camera(
            queue,
            crate::depth::root_view_projection(eye),
            Vec2::new(target.size.0 as f32, target.size.1 as f32),
        );
        if !*uploaded {
            node.upload_segments(device, queue, segments);
            *uploaded = true;
        }
        Ok(())
    }

    fn draw(&mut self, pass: &mut RenderPass<'_>, _target: &FrameTarget<'_>) -> anyhow::Result<()> {
        if let Some(built) = self.shared.borrow().built.as_ref() {
            built.node.draw(pass, None);
        }
        Ok(())
    }

    fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
        let Some(depth) = target.depth else {
            return Ok(());
        };
        self.prepare(encoder, target)?;
        if let Some(built) = self.shared.borrow().built.as_ref() {
            built.node.record(encoder, target.color, Some(depth), None);
        }
        Ok(())
    }

    fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> anyhow::Result<()> {
        let mut state = self.shared.borrow_mut();
        state.built = Some(Built {
            node: LineRasterNode::new(
                &gpu.device,
                frame.color,
                DepthMode::ReadOnly {
                    format: frame.depth,
                },
                DepthConvention::ReversedZ,
            ),
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
        });
        state.uploaded = false;
        Ok(())
    }
}
