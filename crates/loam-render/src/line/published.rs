use std::cell::RefCell;
use std::rc::Rc;

use glam::Vec2;
use loam_runtime::{DomainId, SegmentRecord, Stamp, ViewTarget};
use wgpu::{CommandEncoder, Device, Queue, RenderPass, TextureFormat};

use super::LineRasterNode;
use crate::depth::DEPTH_FORMAT;
use crate::device::GpuContext;
use crate::pass::{FrameFormat, FramePass, FrameTarget, PassStage};
use crate::{DepthConvention, DepthMode};

pub(crate) struct ViewLines {
    pub(crate) opaque: LineRasterNode,
    pub(crate) translucent: LineRasterNode,
    scratch: Vec<SegmentRecord>,
    pub(crate) uploaded: Option<(DomainId, ViewTarget, Stamp)>,
    camera: Option<(glam::Mat4, Vec2)>,
}

impl ViewLines {
    pub(crate) fn new(device: &Device, format: TextureFormat) -> Self {
        Self {
            opaque: LineRasterNode::new(
                device,
                format,
                DepthMode::ReadWrite {
                    format: DEPTH_FORMAT,
                },
                DepthConvention::ReversedZ,
            ),
            translucent: LineRasterNode::new(
                device,
                format,
                DepthMode::ReadOnly {
                    format: DEPTH_FORMAT,
                },
                DepthConvention::ReversedZ,
            ),
            scratch: Vec::new(),
            uploaded: None,
            camera: None,
        }
    }

    pub(crate) fn set_camera(&mut self, queue: &Queue, camera: glam::Mat4, viewport: Vec2) {
        if self.camera == Some((camera, viewport)) {
            return;
        }
        self.camera = Some((camera, viewport));
        self.opaque.set_camera(queue, camera, viewport);
        self.translucent.set_camera(queue, camera, viewport);
    }

    pub(crate) fn upload_segments(
        &mut self,
        device: &Device,
        queue: &Queue,
        segments: &[SegmentRecord],
    ) {
        self.scratch.clear();
        self.scratch
            .extend(segments.iter().copied().filter(segment_is_opaque));
        self.opaque.upload_segments(device, queue, &self.scratch);
        self.scratch.clear();
        self.scratch.extend(
            segments
                .iter()
                .copied()
                .filter(|segment| !segment_is_opaque(segment)),
        );
        self.translucent
            .upload_segments(device, queue, &self.scratch);
    }
}

pub(crate) struct PublishedLines {
    pub(crate) views: Rc<RefCell<Vec<ViewLines>>>,
}

impl FramePass for PublishedLines {
    fn name(&self) -> &'static str {
        "present-draw"
    }

    fn stage(&self) -> PassStage {
        PassStage::Scene
    }

    fn depth_convention(&self) -> Option<DepthConvention> {
        Some(DepthConvention::ReversedZ)
    }

    fn depth_read(&self) -> Option<DepthConvention> {
        Some(DepthConvention::ReversedZ)
    }

    fn shares_pass(&self) -> bool {
        true
    }

    fn draw(&mut self, pass: &mut RenderPass<'_>, _target: &FrameTarget<'_>) -> anyhow::Result<()> {
        let views = self.views.borrow();
        for slot in views.iter() {
            slot.opaque.draw(pass, None);
        }
        for slot in views.iter() {
            slot.translucent.draw(pass, None);
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
        let views = self.views.borrow();
        for slot in views.iter() {
            slot.opaque.record(encoder, target.color, Some(depth), None);
        }
        for slot in views.iter() {
            slot.translucent
                .record(encoder, target.color, Some(depth), None);
        }
        Ok(())
    }

    fn attach(&mut self, _gpu: &GpuContext, _frame: FrameFormat) -> anyhow::Result<()> {
        Ok(())
    }
}

fn segment_is_opaque(segment: &SegmentRecord) -> bool {
    segment.start_color[3] >= 1.0 && segment.end_color[3] >= 1.0
}
