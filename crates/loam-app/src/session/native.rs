use std::sync::Arc;

use web_time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use loam_render::device::{FeatureRequest, RenderDevice};
use loam_runtime::host::{HostConfig, HostError};
use loam_runtime::{Session, Stores};

use super::app::SessionApp;
use super::frame::{failed, Frame, Target};
use super::input::winit_key;
use super::pacing::Pace;
use super::WorkContext;

pub fn run<A: Stores>(session: Session<A>, config: HostConfig) -> Result<(), HostError> {
    launch(session, SessionApp::new(config))
}

pub fn run_with_work<A: Stores>(
    session: Session<A>,
    config: HostConfig,
    record: impl FnMut(WorkContext<'_>) + 'static,
) -> Result<(), HostError> {
    launch(session, SessionApp::new(config).work(record))
}

pub fn launch<A: Stores>(session: Session<A>, app: SessionApp<A>) -> Result<(), HostError> {
    crate::par_native::install();
    let event_loop = EventLoop::new().map_err(failed)?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut host = Host::new(session, app);
    event_loop.run_app(&mut host).map_err(failed)?;
    match host.failure {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

struct Host<A: Stores> {
    frame: Frame<A>,
    window: Option<Arc<Window>>,
    device: Option<RenderDevice>,
    failure: Option<HostError>,
}

impl<A: Stores> Host<A> {
    fn new(session: Session<A>, app: SessionApp<A>) -> Self {
        Self {
            frame: Frame::new(session, app),
            window: None,
            device: None,
            failure: None,
        }
    }

    fn stop(&mut self, elwt: &ActiveEventLoop, error: HostError) {
        self.failure = Some(error);
        elwt.exit();
    }

    fn redraw(&mut self, elwt: &ActiveEventLoop) {
        if let Err(error) = self.present() {
            self.stop(elwt, error);
        }
    }

    fn present(&mut self) -> Result<(), HostError> {
        let Host { frame, device, .. } = self;
        let Some(device) = device.as_mut() else {
            return Ok(());
        };
        let size = device.surface_bundle.size;
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        if let Some(loss) = device.take_device_loss() {
            tracing::warn!("device lost ({:?}): {}", loss.reason, loss.message);
            pollster::block_on(device.recover()).map_err(|error| failed(format!("{error:#}")))?;
            return frame.recover(&device.context);
        }

        if let Some(enabled) = frame.app_mut().vsync.take() {
            crate::frame_pacing::apply_present_mode(device, enabled);
        }

        let now = loop {
            let now = Instant::now();
            match frame.app_mut().pacer.decide(now) {
                Pace::Run => break now,
                Pace::Wait(deadline) => crate::frame_pacing::precise_sleep_until(deadline),
            }
        };

        let Ok((surface, swap_view)) = device.begin_frame() else {
            return Ok(());
        };
        {
            let view = device
                .msaa_view()
                .or(device.scene_view())
                .unwrap_or(&swap_view);
            let target = Target {
                view,
                texture: &surface.texture,
                format: device.surface_bundle.config.format,
                size: (size.width, size.height),
            };
            frame.step(&device.context, &target, now, |encoder| {
                if device.sample_count() > 1 {
                    device.resolve_scene_to_swap(encoder, &swap_view);
                }
                if device.scene_view().is_some() {
                    device.composite_to_swap(encoder, &swap_view);
                }
            })?;
        }
        surface.present();
        Ok(())
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
        let device = pollster::block_on(RenderDevice::new(
            window.clone(),
            FeatureRequest::default(),
            1,
        ));
        let device = match device {
            Ok(device) => device,
            Err(error) => return self.stop(elwt, failed(format!("{error:#}"))),
        };
        let size = device.surface_bundle.size;
        let attached = self.frame.attach(
            &device.context,
            device.target_format(),
            device.sample_count(),
            Some(window.clone()),
            (size.width, size.height),
            window.scale_factor() as f32,
        );
        if let Err(error) = attached {
            return self.stop(elwt, error);
        }
        self.device = Some(device);
        self.window = Some(window);
        self.frame.reset_clock(Instant::now());
    }

    fn window_event(&mut self, elwt: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let consumed = self
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
                if let Some(device) = self.device.as_mut() {
                    device.resize(size);
                }
                self.frame.resize(size.width, size.height, scale);
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                let (width, height) = self.frame.input_mut().size();
                self.frame.resize(width, height, scale);
            }
            WindowEvent::KeyboardInput { event, .. } if !consumed => {
                let Some(key) = winit_key(&event) else {
                    return;
                };
                self.frame.action(key, event.state.is_pressed());
            }
            WindowEvent::CursorMoved { position, .. } if !consumed => {
                let input = self.frame.input_mut();
                let ndc = input.ndc(position.x, position.y);
                input.moved(ndc);
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } if !consumed => {
                let input = self.frame.input_mut();
                let cursor = input.cursor();
                input.button(cursor, state.is_pressed());
            }
            WindowEvent::RedrawRequested => self.redraw(elwt),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _elwt: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}
