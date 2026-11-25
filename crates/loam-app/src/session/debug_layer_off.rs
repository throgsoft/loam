#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

use loam_render::device::GpuContext;
use loam_render::pass::FramePass;
use loam_runtime::Stores;
use wgpu::{CommandBuffer, Texture, TextureFormat};
#[cfg(not(target_arch = "wasm32"))]
use winit::window::Window;

use super::console::SessionConsole;

#[derive(Default)]
pub(crate) struct LayerSlot;

impl LayerSlot {
    pub(crate) fn attach(
        &mut self,
        _gpu: &GpuContext,
        _format: TextureFormat,
        _view_format: Option<TextureFormat>,
        #[cfg(not(target_arch = "wasm32"))] _window: Option<Arc<Window>>,
        _size: (u32, u32),
        _scale: f32,
    ) -> Option<Box<dyn FramePass>> {
        None
    }

    pub(crate) fn resize(&self, _width: u32, _height: u32, _scale: f32) {}

    pub(crate) fn aim(&self, _texture: &Texture) {}

    pub(crate) fn begin(&mut self) {}

    pub(crate) fn typing(&self) -> bool {
        false
    }

    pub(crate) fn seat(&self, scale: f32) -> ([f32; 2], f32) {
        ([0.0; 2], scale)
    }

    pub(crate) fn finish<A: Stores>(&mut self, _console: &mut SessionConsole<A>) -> bool {
        false
    }

    pub(crate) fn take_callbacks(&self, _into: &mut Vec<CommandBuffer>) {}
}
