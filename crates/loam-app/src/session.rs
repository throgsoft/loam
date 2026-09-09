use std::sync::Arc;

use glam::Vec2;
use web_time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key as Physical, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

use loam_render::device::{FeatureRequest, RenderDevice};
use loam_render::present::Presenter;
use loam_runtime::host::{HostConfig, HostError};
use loam_runtime::{ActionEvent, Input, Key, Pointer, PointerPhase, Records, Session, Stores};
use loam_time::{frame_trace, FixedTimestep};

const BACKGROUND: wgpu::Color = wgpu::Color {
    r: 0.02,
    g: 0.02,
    b: 0.03,
    a: 1.0,
};

/// Owns the window, device, and loop; the session stays a CPU value.
pub fn run<A: Stores>(session: Session<A>, config: HostConfig) -> Result<(), HostError> {
    let event_loop = EventLoop::new().map_err(failed)?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut host = Host::new(session, config);
    event_loop.run_app(&mut host).map_err(failed)?;
    match host.failure {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn failed(error: impl std::fmt::Display) -> HostError {
    HostError::Host(error.to_string())
}

struct Host<A: Stores> {
    session: Session<A>,
    records: Records<A>,
    config: HostConfig,
    timestep: FixedTimestep,
    started: Instant,
    window: Option<Arc<Window>>,
    device: Option<RenderDevice>,
    presenter: Option<Presenter>,
    input: Input,
    cursor: [f32; 2],
    dragging: bool,
    failure: Option<HostError>,
}

impl<A: Stores> Host<A> {
    fn new(session: Session<A>, config: HostConfig) -> Self {
        let sim = session.config();
        let timestep = FixedTimestep::new(sim.fixed_hz).with_max_catch_up(sim.max_ticks_per_frame);
        Self {
            session,
            records: Records::default(),
            config,
            timestep,
            started: Instant::now(),
            window: None,
            device: None,
            presenter: None,
            input: Input::default(),
            cursor: [0.0; 2],
            dragging: false,
            failure: None,
        }
    }

    fn stop(&mut self, elwt: &ActiveEventLoop, error: HostError) {
        self.failure = Some(error);
        elwt.exit();
    }

    fn ndc(&self, position: PhysicalPosition<f64>) -> [f32; 2] {
        let Some(device) = self.device.as_ref() else {
            return self.cursor;
        };
        let size = device.surface_bundle.size;
        [
            (position.x as f32 / size.width.max(1) as f32) * 2.0 - 1.0,
            1.0 - (position.y as f32 / size.height.max(1) as f32) * 2.0,
        ]
    }

    fn on_key(&mut self, event: &KeyEvent) {
        let key = match &event.logical_key {
            Physical::Named(NamedKey::Space) => Key::Space,
            Physical::Named(NamedKey::Escape) => Key::Escape,
            Physical::Character(text) => match text.chars().next() {
                Some(letter) if letter.is_ascii_digit() => Key::Digit(letter as u8 - b'0'),
                Some(letter) => Key::Letter(letter.to_ascii_lowercase()),
                None => return,
            },
            _ => return,
        };
        self.on_action(key, event.state.is_pressed());
    }

    fn on_action(&mut self, key: Key, pressed: bool) {
        let Some(action) = self.config.bindings.action(key) else {
            return;
        };
        if pressed == self.input.held.contains(&action) {
            return;
        }
        if pressed {
            self.input.held.push(action);
        } else {
            self.input.held.retain(|held| *held != action);
        }
        self.input.actions.push(ActionEvent { action, pressed });
    }

    fn on_pointer(&mut self, ndc: [f32; 2], delta: [f32; 2], phase: PointerPhase) {
        self.input.pointers.push(Pointer {
            id: 0,
            ndc,
            delta,
            phase,
            time: self.started.elapsed().as_secs_f64(),
        });
    }

    fn reclaim_input(&mut self) {
        self.input = self.session.take_input();
        self.input.pointers.clear();
        self.input.actions.clear();
    }

    fn frame(&mut self, elwt: &ActiveEventLoop) {
        frame_trace::begin_frame();
        self.record_frame(elwt);
        frame_trace::end_frame();
    }

    fn record_frame(&mut self, elwt: &ActiveEventLoop) {
        let Some(size) = self
            .device
            .as_ref()
            .map(|device| device.surface_bundle.size)
        else {
            return;
        };
        if size.width == 0 || size.height == 0 {
            return;
        }
        let input = std::mem::take(&mut self.input);
        {
            let _dispatch = frame_trace::scope("dispatch");
            if let Err(error) = self.session.boundary(input) {
                return self.stop(elwt, error.into());
            }
        }
        {
            let _simulation = frame_trace::scope("simulation");
            for _ in self.timestep.advance(Instant::now()) {
                if let Err(error) = self.session.tick() {
                    return self.stop(elwt, error.into());
                }
            }
        }
        self.reclaim_input();
        let (Some(device), Some(presenter)) = (self.device.as_mut(), self.presenter.as_mut())
        else {
            return;
        };
        if let Some(loss) = device.take_device_loss() {
            tracing::warn!("device lost ({:?}): {}", loss.reason, loss.message);
            if let Err(error) = pollster::block_on(device.recover()) {
                return self.stop(elwt, failed(format!("{error:#}")));
            }
            if let Err(error) = presenter.attach(device) {
                return self.stop(elwt, failed(error));
            }
            return;
        }
        let eye = {
            let root = self.session.views().root();
            let image = self.session.views_mut().get_mut(root);
            let Some(image) = image else {
                return;
            };
            image.eye.aspect = size.width as f32 / size.height as f32;
            image.eye
        };
        let published = {
            let _publication = frame_trace::scope("publication");
            if let Err(error) = self.records.publish(&mut self.session) {
                return self.stop(elwt, HostError::Host(format!("{error:?}")));
            }
            self.records.lend()
        };
        let Some(published) = published else {
            return;
        };
        let _presentation = frame_trace::scope("presentation");
        let viewport = Vec2::new(size.width as f32, size.height as f32);
        presenter.upload(
            &device.device,
            &device.queue,
            &eye,
            viewport,
            &published.views,
        );
        self.records.release(published);

        let Ok((frame, swap_view)) = device.begin_frame() else {
            return;
        };
        let target = device
            .msaa_view()
            .or(device.scene_view())
            .unwrap_or(&swap_view);
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("loam-app::session"),
            });
        presenter.record(
            &device.device,
            &mut encoder,
            target,
            (size.width, size.height),
            BACKGROUND,
        );
        if device.sample_count() > 1 {
            device.resolve_scene_to_swap(&mut encoder, &swap_view);
        }
        if device.scene_view().is_some() {
            device.composite_to_swap(&mut encoder, &swap_view);
        }
        device.queue.submit(Some(encoder.finish()));
        presenter.after_submit();
        frame.present();
    }
}

impl<A: Stores> ApplicationHandler for Host<A> {
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = WindowAttributes::default().with_title(self.config.title);
        let window = match elwt.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => return self.stop(elwt, failed(error)),
        };
        let device = pollster::block_on(RenderDevice::new(
            window.clone(),
            FeatureRequest::default(),
            1,
        ));
        let device = match device {
            Ok(device) => device,
            Err(error) => return self.stop(elwt, failed(format!("{error:#}"))),
        };
        let mut presenter = Presenter::new(device.target_format(), device.sample_count());
        if let Err(error) = presenter.attach(&device) {
            return self.stop(elwt, failed(error));
        }
        self.presenter = Some(presenter);
        self.device = Some(device);
        self.window = Some(window);
        self.timestep.reset_clock(Instant::now());
    }

    fn window_event(&mut self, elwt: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(size) => {
                if let Some(device) = self.device.as_mut() {
                    device.resize(size);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => self.on_key(&event),
            WindowEvent::CursorMoved { position, .. } => {
                let ndc = self.ndc(position);
                let delta = [ndc[0] - self.cursor[0], ndc[1] - self.cursor[1]];
                self.cursor = ndc;
                if self.dragging {
                    self.on_pointer(ndc, delta, PointerPhase::Moved);
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.dragging = state.is_pressed();
                let phase = if self.dragging {
                    PointerPhase::Began
                } else {
                    PointerPhase::Ended
                };
                self.on_pointer(self.cursor, [0.0; 2], phase);
            }
            WindowEvent::RedrawRequested => self.frame(elwt),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _elwt: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

#[cfg(test)]
mod tests {
    use loam_runtime::{ActionId, Bindings, SimConfig};

    use super::*;

    mod alloc_probe {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        thread_local! {
            static BYTES: Cell<usize> = const { Cell::new(0) };
        }

        pub struct Counting;

        // SAFETY: Methods preserve System contracts; const TLS and wrapping Cell updates cannot unwind.
        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                let _ = BYTES.try_with(|bytes| bytes.set(bytes.get().wrapping_add(layout.size())));
                // SAFETY: The caller supplies a valid nonzero allocation layout.
                unsafe { System.alloc(layout) }
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                // SAFETY: The caller supplies a live System allocation and its original layout.
                unsafe { System.dealloc(ptr, layout) }
            }

            unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                let _ = BYTES.try_with(|bytes| bytes.set(bytes.get().wrapping_add(new_size)));
                // SAFETY: The caller supplies a live System allocation, its layout, and a valid new size.
                unsafe { System.realloc(ptr, layout, new_size) }
            }
        }

        pub fn bytes_allocated_by(body: impl FnOnce()) -> usize {
            let before = BYTES.with(Cell::get);
            body();
            BYTES.with(Cell::get).wrapping_sub(before)
        }
    }

    #[global_allocator]
    static COUNTING_ALLOCATOR: alloc_probe::Counting = alloc_probe::Counting;

    const WALK: ActionId = ActionId(0);

    loam_runtime::stores! {
        #[derive(Default)]
        pub struct Empty {}
    }

    #[test]
    fn a_warmed_frame_with_input_allocates_in_the_hosts_input_conversion() {
        let bindings = Bindings::new().key(Key::Letter('w'), WALK);
        let session = Session::new(Empty::default(), SimConfig::default());
        let mut host = Host::new(session, HostConfig::new("input", bindings));
        let cycle = |host: &mut Host<Empty>| {
            host.on_action(Key::Letter('w'), true);
            host.on_pointer([0.1, 0.2], [0.01, 0.0], PointerPhase::Began);
            host.on_pointer([0.2, 0.2], [0.1, 0.0], PointerPhase::Moved);
            host.on_action(Key::Letter('w'), false);
            let input = std::mem::take(&mut host.input);
            host.session.boundary(input).unwrap();
            host.session.tick().unwrap();
            host.reclaim_input();
        };
        for _ in 0..8 {
            cycle(&mut host);
        }

        let bytes = alloc_probe::bytes_allocated_by(|| {
            for _ in 0..16 {
                cycle(&mut host);
            }
        });
        assert_eq!(
            bytes, 0,
            "16 warmed frames of input asked the allocator for {bytes} bytes"
        );
    }
}
