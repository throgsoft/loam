use std::cell::RefCell;
use std::rc::Rc;

use glam::Vec2;
use loam_runtime::{Eye, SegmentRecord};
use wgpu::{CommandEncoder, Device, Queue, TextureFormat};

use crate::device::{GpuContext, MissingGpuCapability};
use crate::pass::{
    FrameFormat, FramePass, FrameTarget, PassOrder, ResourceId, SCENE_COLOR, SCENE_DEPTH,
};
use crate::{DepthConvention, DepthMode, LineRasterNode};

const READS: [ResourceId; 1] = [SCENE_DEPTH];
const WRITES: [ResourceId; 1] = [SCENE_COLOR];

struct State {
    format: TextureFormat,
    depth: TextureFormat,
    sample_count: u32,
    eye: Eye,
    segments: Vec<SegmentRecord>,
    uploaded: bool,
    node: Option<LineRasterNode>,
    device: Option<Device>,
    queue: Option<Queue>,
}

/// A `LineRasterNode` after the scene that tests scene depth and writes scene colour; clones share one state.
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
                format: TextureFormat::Rgba8UnormSrgb,
                depth: crate::view::DEPTH_FORMAT,
                sample_count: 1,
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

    fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> Result<(), MissingGpuCapability> {
        {
            let mut state = self.shared.borrow_mut();
            state.format = frame.color;
            state.depth = frame.depth;
            state.sample_count = frame.sample_count;
        }
        self.rebuild(gpu)
    }

    fn rebuild(&mut self, gpu: &GpuContext) -> Result<(), MissingGpuCapability> {
        let mut state = self.shared.borrow_mut();
        state.node = Some(LineRasterNode::new(
            &gpu.device,
            state.format,
            DepthMode::ReadOnly {
                format: state.depth,
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
