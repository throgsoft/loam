use std::cell::RefCell;
use std::rc::Rc;

use glam::Vec2;
use loam_runtime::{Eye, SegmentRecord};
use wgpu::{CommandEncoder, Device, Queue, TextureFormat};

use crate::device::{GpuContext, MissingGpuCapability};
use crate::pass::{FramePass, FrameTarget, PassOrder, ResourceId, SCENE_COLOR, SCENE_DEPTH};
use crate::view::DEPTH_FORMAT;
use crate::{DepthConvention, DepthMode, LineRasterNode};

const READS: [ResourceId; 1] = [SCENE_DEPTH];
const WRITES: [ResourceId; 1] = [SCENE_COLOR];

struct State {
    format: TextureFormat,
    sample_count: u32,
    eye: Eye,
    segments: Vec<SegmentRecord>,
    uploaded: bool,
    node: Option<LineRasterNode>,
    device: Option<Device>,
    queue: Option<Queue>,
}

/// A [`LineRasterNode`] over the segments an application publishes, drawn after the scene under the root eye and tested against the scene's depth without writing it.
#[derive(Clone)]
pub struct LinePass {
    name: &'static str,
    shared: Rc<RefCell<State>>,
}

impl LinePass {
    pub fn new(name: &'static str, format: TextureFormat, sample_count: u32) -> Self {
        Self {
            name,
            shared: Rc::new(RefCell::new(State {
                format,
                sample_count,
                eye: Eye::default(),
                segments: Vec::new(),
                uploaded: false,
                node: None,
                device: None,
                queue: None,
            })),
        }
    }

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
        let Some(depth) = target.depth else {
            return;
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
            return;
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
    }

    fn rebuild(&mut self, gpu: &GpuContext) -> Result<(), MissingGpuCapability> {
        let mut state = self.shared.borrow_mut();
        state.node = Some(LineRasterNode::new(
            &gpu.device,
            state.format,
            DepthMode::ReadOnly {
                format: DEPTH_FORMAT,
            },
            DepthConvention::ReversedZ,
            state.sample_count,
        ));
        state.device = Some(gpu.device.clone());
        state.queue = Some(gpu.queue.clone());
        state.uploaded = false;
        Ok(())
    }
}
