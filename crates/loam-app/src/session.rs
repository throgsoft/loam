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
use loam_runtime::{
    ActionEvent, ActionId, Input, Key, Pointer, PointerPhase, Records, Session, Stores,
};
use loam_time::FixedTimestep;

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
    pointers: Vec<Pointer>,
    actions: Vec<ActionEvent>,
    held: Vec<ActionId>,
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
            pointers: Vec::new(),
            actions: Vec::new(),
            held: Vec::new(),
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
        let Some(action) = self.config.bindings.action(key) else {
            return;
        };
        let pressed = event.state.is_pressed();
        if pressed == self.held.contains(&action) {
            return;
        }
        if pressed {
            self.held.push(action);
        } else {
            self.held.retain(|held| *held != action);
        }
        self.actions.push(ActionEvent { action, pressed });
    }

    fn frame(&mut self, elwt: &ActiveEventLoop) {
        let (Some(device), Some(presenter)) = (self.device.as_mut(), self.presenter.as_mut())
        else {
            return;
        };
        let size = device.surface_bundle.size;
        if size.width == 0 || size.height == 0 {
            return;
        }
        let input = Input {
            pointers: std::mem::take(&mut self.pointers),
            actions: std::mem::take(&mut self.actions),
            held: self.held.clone(),
        };
        if let Err(error) = self.session.boundary(input) {
            return self.stop(elwt, error.into());
        }
        for _ in self.timestep.advance(Instant::now()) {
            if let Err(error) = self.session.tick() {
                return self.stop(elwt, error.into());
            }
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
        if let Err(error) = self.records.publish(&mut self.session) {
            return self.stop(elwt, HostError::Host(format!("{error:?}")));
        }
        let Some(published) = self.records.lend() else {
            return;
        };
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
        self.presenter = Some(Presenter::new(
            device.target_format(),
            device.sample_count(),
        ));
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
                    self.pointers.push(Pointer {
                        id: 0,
                        ndc,
                        delta,
                        phase: PointerPhase::Moved,
                        time: self.started.elapsed().as_secs_f64(),
                    });
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.dragging = state.is_pressed();
                self.pointers.push(Pointer {
                    id: 0,
                    ndc: self.cursor,
                    delta: [0.0; 2],
                    phase: if self.dragging {
                        PointerPhase::Began
                    } else {
                        PointerPhase::Ended
                    },
                    time: self.started.elapsed().as_secs_f64(),
                });
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
