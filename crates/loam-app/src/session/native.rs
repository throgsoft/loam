use std::sync::Arc;

use web_time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

use loam_render::device::{FeatureRequest, RenderDevice};
use loam_runtime::host::HostError;
use loam_runtime::{PointerButton, Session, Stores};

use super::app::SessionApp;
use super::frame::{failed, Frame};
use super::input::{winit_alt, winit_key};
use super::pacing::Pace;
use super::surface::SurfaceHost;
use crate::args::Args;
use crate::WasmConfig;

pub fn launch<A: Stores>(
    factory: impl FnOnce(Args) -> Result<(Session<A>, SessionApp<A>), HostError>,
) -> Result<(), HostError> {
    launch_with(WasmConfig::default(), factory)
}

pub fn launch_with<A: Stores>(
    _wasm: WasmConfig,
    factory: impl FnOnce(Args) -> Result<(Session<A>, SessionApp<A>), HostError>,
) -> Result<(), HostError> {
    install_tracing();
    crate::par_native::install();
    run(Args::current(), factory)
}

pub fn launch_or_headless<A: Stores>(
    factory: impl FnOnce(Args) -> Result<(Session<A>, SessionApp<A>), HostError>,
    headless: impl FnOnce(Args) -> Result<(), HostError>,
) -> Result<(), HostError> {
    install_tracing();
    crate::par_native::install();
    let args = Args::current();
    if args.has_bare_flag("headless") {
        return headless(args);
    }
    run(args, factory)
}

fn install_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

fn run<A: Stores>(
    args: Args,
    factory: impl FnOnce(Args) -> Result<(Session<A>, SessionApp<A>), HostError>,
) -> Result<(), HostError> {
    let (session, app) = factory(args)?;
    let event_loop = EventLoop::new().map_err(failed)?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut host = Host::new(session, app)?;
    event_loop.run_app(&mut host).map_err(failed)?;
    match host.failure {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

struct Host<A: Stores> {
    frame: Frame<A>,
    window: Option<Arc<Window>>,
    surface: Option<SurfaceHost>,
    device: Option<RenderDevice>,
    redraw_deadline: Option<Instant>,
    failure: Option<HostError>,
}

impl<A: Stores> Host<A> {
    fn new(session: Session<A>, app: SessionApp<A>) -> Result<Self, HostError> {
        Ok(Self {
            frame: Frame::new(session, app)?,
            window: None,
            surface: None,
            device: None,
            redraw_deadline: None,
            failure: None,
        })
    }

    fn stop(&mut self, elwt: &ActiveEventLoop, error: HostError) {
        self.failure = Some(error);
        elwt.exit();
    }

    fn redraw(&mut self, elwt: &ActiveEventLoop) {
        if let Err(error) = self.present(elwt) {
            self.stop(elwt, error);
        }
    }

    fn present(&mut self, elwt: &ActiveEventLoop) -> Result<(), HostError> {
        let Host {
            frame,
            window,
            surface,
            device,
            redraw_deadline,
            ..
        } = self;
        let Some(device) = device.as_mut() else {
            return Ok(());
        };
        let Some(surface) = surface.as_mut() else {
            return Ok(());
        };
        let size = surface.size();
        if size.0 == 0 || size.1 == 0 {
            return Ok(());
        }

        if let Some(loss) = device.take_device_loss() {
            tracing::warn!("device lost ({:?}): {}", loss.reason, loss.message);
            pollster::block_on(device.recover()).map_err(|error| failed(format!("{error:#}")))?;
            surface.reconfigure(&device.context.device);
            return frame.recover(&device.context);
        }

        if let Some(enabled) = frame.app_mut().vsync.take() {
            surface.set_vsync(&device.context.device, enabled);
        }

        let now = Instant::now();
        if let Pace::Wait(deadline) = frame.app_mut().pacer.decide(now) {
            *redraw_deadline = Some(deadline);
            elwt.set_control_flow(ControlFlow::WaitUntil(deadline));
            return Ok(());
        }
        *redraw_deadline = frame.app_mut().pacer.deadline();
        match *redraw_deadline {
            Some(deadline) => elwt.set_control_flow(ControlFlow::WaitUntil(deadline)),
            None => elwt.set_control_flow(ControlFlow::Poll),
        };

        let window = window.as_deref();
        surface.present(device, frame, now, |frame, stepped| {
            if stepped {
                Self::apply_cursor_request(frame, window);
            }
        })?;
        Ok(())
    }

    fn apply_cursor_request(frame: &mut Frame<A>, window: Option<&Window>) {
        let Some(locked) = frame.take_cursor_request() else {
            return;
        };
        let Some(window) = window else {
            return;
        };
        let mode = if locked {
            CursorGrabMode::Locked
        } else {
            CursorGrabMode::None
        };
        let applied = window.set_cursor_grab(mode).or_else(|locked_error| {
            if !locked {
                return Err(locked_error);
            }
            window
                .set_cursor_grab(CursorGrabMode::Confined)
                .map_err(|confined_error| {
                    tracing::warn!(
                        "locked cursor failed: {locked_error}; confined cursor failed: {confined_error}"
                    );
                    confined_error
                })
        });
        match applied {
            Ok(()) => {
                window.set_cursor_visible(!locked);
                frame.cursor_applied(locked);
            }
            Err(error) => {
                tracing::warn!("cursor lock request failed: {error:?}");
                if locked {
                    frame.cursor_applied(false);
                }
            }
        }
    }

    fn wake_frame(&mut self, elwt: &ActiveEventLoop) {
        self.redraw_deadline = None;
        self.frame.app_mut().pacer.reset();
        elwt.set_control_flow(ControlFlow::Wait);
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

impl<A: Stores> ApplicationHandler for Host<A> {
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = WindowAttributes::default().with_title(self.frame.config().title);
        let window = match elwt.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => return self.stop(elwt, failed(error)),
        };
        let instance = wgpu::Instance::default();
        let surface = match instance.create_surface(window.clone()) {
            Ok(surface) => surface,
            Err(error) => return self.stop(elwt, failed(error)),
        };
        let size = window.inner_size();
        let attached = pollster::block_on(SurfaceHost::new(
            instance,
            surface,
            (size.width, size.height),
            FeatureRequest::default(),
        ));
        let (surface, device) = match attached {
            Ok(attached) => attached,
            Err(error) => return self.stop(elwt, failed(format!("{error:#}"))),
        };
        let attached = self.frame.attach(
            &device.context,
            device.target_format(),
            Some(window.clone()),
            (size.width, size.height),
            window.scale_factor() as f32,
        );
        if let Err(error) = attached {
            return self.stop(elwt, error);
        }
        self.surface = Some(surface);
        self.device = Some(device);
        self.window = Some(window);
        self.frame.reset_clock(Instant::now());
    }

    fn window_event(&mut self, elwt: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let wakes_frame = !matches!(
            &event,
            WindowEvent::RedrawRequested | WindowEvent::CloseRequested
        );
        let hidden_pointer = self.frame.cursor_locked()
            && matches!(
                &event,
                WindowEvent::CursorMoved { .. }
                    | WindowEvent::MouseInput { .. }
                    | WindowEvent::MouseWheel { .. }
            );
        let consumed = hidden_pointer
            || self
                .frame
                .layer()
                .is_some_and(|layer| layer.on_window_event(&event));
        let scale = self
            .window
            .as_ref()
            .map_or(1.0, |window| window.scale_factor() as f32);
        match event {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(size) => {
                if let (Some(surface), Some(device)) = (self.surface.as_mut(), self.device.as_mut())
                {
                    let size = (size.width, size.height);
                    surface.resize(&device.context.device, size);
                    device.resize(size);
                }
                self.frame.resize(size.width, size.height, scale);
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                let (width, height) = self.frame.input_mut().size();
                self.frame.resize(width, height, scale);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(index) = winit_alt(&event) {
                    self.frame.alt(index, event.state.is_pressed());
                }
                if let Some(key) = winit_key(&event) {
                    self.frame.action(key, event.state.is_pressed(), consumed);
                }
            }
            WindowEvent::CursorMoved { position, .. } if !consumed => {
                let input = self.frame.input_mut();
                let ndc = input.ndc(position.x, position.y);
                input.moved(ndc);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let button = match button {
                    MouseButton::Left => Some(PointerButton::Primary),
                    MouseButton::Right => Some(PointerButton::Secondary),
                    MouseButton::Middle => Some(PointerButton::Middle),
                    _ => None,
                };
                if let Some(button) = button {
                    let input = self.frame.input_mut();
                    let cursor = input.cursor();
                    input.host_button(cursor, button, state.is_pressed(), consumed);
                }
            }
            WindowEvent::Focused(focused) => self.frame.focus(focused),
            WindowEvent::MouseWheel { delta, .. } if !consumed => {
                let delta = match delta {
                    MouseScrollDelta::LineDelta(x, y) => [x, y],
                    MouseScrollDelta::PixelDelta(position) => [
                        position.x as f32 / crate::wasm::input_queue::SCROLL_PIXELS_PER_LINE,
                        position.y as f32 / crate::wasm::input_queue::SCROLL_PIXELS_PER_LINE,
                    ],
                };
                self.frame.input_mut().wheel(delta);
            }
            WindowEvent::RedrawRequested => self.redraw(elwt),
            _ => {}
        }
        Self::apply_cursor_request(&mut self.frame, self.window.as_deref());
        if wakes_frame {
            self.wake_frame(elwt);
        }
    }

    fn device_event(&mut self, elwt: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.frame
                .input_mut()
                .raw_motion(delta.0 as f32, delta.1 as f32);
            self.wake_frame(elwt);
        }
    }

    fn about_to_wait(&mut self, elwt: &ActiveEventLoop) {
        if let Some(deadline) = self.redraw_deadline {
            if Instant::now() < deadline {
                elwt.set_control_flow(ControlFlow::WaitUntil(deadline));
                return;
            }
            self.redraw_deadline = None;
        }
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}
