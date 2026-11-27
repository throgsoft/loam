use web_time::Instant;
use wgpu::{CommandEncoder, Device};

use super::types::CaptureRequest;
use crate::frame::Target;

pub(crate) struct Capture;

pub(crate) struct Shot;

impl Shot {
    pub(crate) fn wants(&self, _pre: bool) -> bool {
        false
    }
}

impl Capture {
    pub(crate) fn new() -> Self {
        Self
    }

    pub(crate) fn pending(&self, _requests: &[CaptureRequest], _now: Instant) -> bool {
        false
    }

    pub(crate) fn begin(&mut self, _requests: &mut Vec<CaptureRequest>, _now: Instant) -> Shot {
        Shot
    }

    pub(crate) fn record(
        &self,
        _shot: &mut Shot,
        _pre: bool,
        _device: &Device,
        _encoder: &mut CommandEncoder,
        _target: &Target<'_>,
    ) {
    }

    pub(crate) fn finish(&mut self, _shot: Shot, _device: &Device, _now: Instant) {}
}
