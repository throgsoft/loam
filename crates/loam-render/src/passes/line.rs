use std::cell::RefCell;
use std::rc::Rc;

use glam::Vec2;
use loam_runtime::{Eye, SegmentRecord};
use wgpu::{CommandEncoder, Device, Queue};

use crate::device::GpuContext;
use crate::pass::{FrameFormat, FramePass, FrameTarget, PassStage};
use crate::{DepthConvention, DepthMode, LineRasterNode};

struct State {
    eye: Eye,
    segments: Vec<SegmentRecord>,
    uploaded: bool,
    node: Option<LineRasterNode>,
    device: Option<Device>,
    queue: Option<Queue>,
}

/// A shared scene line pass with depth-tested blending.
#[derive(Clone)]
pub struct LinePass {
    name: &'static str,
    shared: Rc<RefCell<State>>,
}

impl LinePass {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            shared: Rc::new(RefCell::new(State {
                eye: Eye::default(),
                segments: Vec::new(),
                uploaded: false,
                node: None,
                device: None,
                queue: None,
            })),
        }
    }

    /// Stores the eye and copies the segments; the record uploads them again only when they differ from the last publish or the device was rebuilt.
    pub fn publish(&self, eye: &Eye, segments: &[SegmentRecord]) {
        let mut state = self.shared.borrow_mut();
        state.eye = *eye;
        state.uploaded &= state.segments == segments;
        state.segments.clear();
        state.segments.extend_from_slice(segments);
    }

    pub fn segment_count(&self) -> usize {
        self.shared.borrow().segments.len()
    }
}

impl FramePass for LinePass {
    fn name(&self) -> &'static str {
        self.name
    }

    fn stage(&self) -> PassStage {
        PassStage::Scene
    }

    fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
        let Some(depth) = target.depth else {
            return Ok(());
        };
        let mut state = self.shared.borrow_mut();
        let State {
            eye,
            segments,
            uploaded,
            node,
            device,
            queue,
            ..
        } = &mut *state;
        let (Some(node), Some(device), Some(queue)) =
            (node.as_mut(), device.as_ref(), queue.as_ref())
        else {
            return Ok(());
        };
        node.set_camera(
            queue,
            crate::view::root_view_projection(eye),
            Vec2::new(target.size.0 as f32, target.size.1 as f32),
        );
        if !*uploaded {
            node.upload_segments(device, queue, segments);
            *uploaded = true;
        }
        node.record(encoder, target.color, Some(depth), None);
        Ok(())
    }

    fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> anyhow::Result<()> {
        let mut state = self.shared.borrow_mut();
        state.node = Some(LineRasterNode::new(
            &gpu.device,
            frame.color,
            DepthMode::ReadOnly {
                format: frame.depth,
            },
            DepthConvention::ReversedZ,
        ));
        state.device = Some(gpu.device.clone());
        state.queue = Some(gpu.queue.clone());
        state.uploaded = false;
        Ok(())
    }
}
