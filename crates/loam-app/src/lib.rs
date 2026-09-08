mod runtime;
pub use runtime::Runtime;
use std::borrow::Cow;
use std::marker::PhantomData;
use std::sync::Arc;
// `std::time::Instant::now` panics on wasm32, so the swap is mandatory there.
use web_time::Instant;

#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
pub mod capture;

pub mod args;
#[cfg(any(not(feature = "capture"), target_arch = "wasm32"))]
#[path = "capture_stub.rs"]
pub mod capture;
mod capture_types;

pub mod camera_rig;
pub mod command;
pub mod cursor;
pub mod environment;
pub mod fps;
pub mod frame_pacing;
pub mod freecam;
pub mod keymap;
pub mod log;
pub mod script;
pub mod shell;
pub mod trace;
pub mod version;
pub mod vsync;
#[cfg(target_arch = "wasm32")]
pub mod wasm;
mod watcher;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

use loam_egui::UiIntegration;
use loam_input::{FrameInput, InputState};
use loam_render::device::RenderDevice;
use loam_time::FixedTimestep;

pub use loam_camera::{
    orbit_on_right, Camera, CameraController, CameraView, FirstPersonController, OrbitController,
};
pub use loam_egui::{egui, world_to_screen, UiCapture};
pub use loam_input::FrameInput as Input;
pub use loam_render::shader::{ShaderDb, ShaderOwner};
pub use watcher::FileWatcher;

pub fn reload_shaders(shader_db: &mut ShaderDb, owner: ShaderOwner, paths: &[std::path::PathBuf]) {
    for path in paths {
        if let Err(error) = shader_db.reload_path(owner, path) {
            tracing::warn!(?path, %error, "shader reload failed");
        }
    }
}

pub trait App: Sized + 'static {
    fn setup(ctx: &mut SetupCtx<'_>) -> anyhow::Result<Self>;

    /// Runs 0..N times per frame, N bounded by the runner's catch-up cap.
    fn tick(&mut self, _dt: f32, _ctx: &mut TickCtx) {}

    /// Dispatches frame commands before simulation; games queue simulation effects for `tick`.
    fn apply_command(
        &mut self,
        cmd: &command::CommandLine,
        _ctx: &mut command::CommandCtx<'_>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("no command target for `{}`", cmd.name)
    }

    /// Runs after all the frame's ticks, with the drained input.
    fn update(&mut self, _ctx: &mut FrameCtx<'_>) {}

    /// Winit events only; worker input reaches the portable callbacks.
    fn on_event(&mut self, _ev: &WindowEvent, _ctx: &mut FrameCtx<'_>) {}

    /// Fired for every press and release, after input routing.
    fn on_key(
        &mut self,
        _code: winit::keyboard::KeyCode,
        _state: ElementState,
        _ctx: &mut FrameCtx<'_>,
    ) {
    }

    fn apply_shader_events(&mut self, events: &[std::path::PathBuf], shader_db: &mut ShaderDb) {
        reload_shaders(shader_db, ShaderDb::ROOT_OWNER, events);
    }

    /// Runs after `apply_shader_events`; rebuild any stale consumer pipelines.
    fn on_shader_reload(&mut self, _ctx: &mut SetupCtx<'_>) {}

    /// Record into the runner's encoder; the runner owns submission.
    fn record(&mut self, _ctx: &mut RenderCtx<'_>) -> anyhow::Result<()> {
        Ok(())
    }

    fn ui(&mut self, _ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {}

    /// The runner rate-limits the `set_title` call to ~1 Hz.
    fn title(&self, _fps: f32) -> Cow<'static, str> {
        Cow::Borrowed("loam app")
    }
}

// The catch-up cap lives solely in the `FixedTimestep`.
pub(crate) fn drive_fixed_ticks<A: App>(
    app: &mut A,
    timestep: &mut FixedTimestep,
    now: Instant,
) -> usize {
    let _scope = loam_time::frame_trace::scope("sim-ticks");
    let ticks = timestep.advance(now);
    let dt = timestep.dt_seconds();
    let n_ticks = (ticks.end - ticks.start) as usize;
    for tick in ticks {
        let mut tctx = TickCtx {
            time: tick as f32 * dt,
            tick,
        };
        app.tick(dt, &mut tctx);
    }
    n_ticks
}

pub struct SetupCtx<'a> {
    pub runtime: &'a Runtime,
    pub rd: &'a RenderDevice,
    pub shader_db: &'a mut ShaderDb,
    /// `None` when filesystem watching failed to init.
    pub watcher: Option<&'a mut FileWatcher>,
    /// Wall-clock seconds since the runner started, including scene rebuilds.
    pub time: f32,
}

/// Games choose state ordering, random sources, and parallel reductions for reproducibility.
pub struct TickCtx {
    /// Derived from the tick index, not the clock.
    pub time: f32,
    pub tick: u64,
}

/// The runner submits `encoder` once per frame; `view` is the scene-pass target.
pub struct RenderCtx<'a> {
    pub rd: &'a RenderDevice,
    pub view: &'a wgpu::TextureView,
    pub encoder: &'a mut wgpu::CommandEncoder,
}

pub struct FrameCtx<'a> {
    pub shader_db: &'a mut ShaderDb,
    pub runtime: &'a Runtime,
    pub rd: &'a RenderDevice,
    pub input: FrameInput,
    pub time: f32,
    pub fps: f32,
    pub n_ticks: usize,
    pub tick: u64,
    /// Wall-clock since the previous `update`; the first call reports the tick interval.
    pub dt: f32,
    pub ui_capture: UiCapture,
    _non_exhaustive: PhantomData<()>,
}

pub const DEFAULT_MAX_TICKS_PER_FRAME: u32 = 4;

pub struct RunConfig {
    pub window: WindowAttributes,
    /// Native only; the wasm worker simulates at 60 Hz.
    pub fixed_hz: u32,
    /// Ticks beyond this are dropped; `0` stops the sim. Native only.
    pub max_ticks_per_frame: u32,
    /// `None` keeps the installed subscriber or `RUST_LOG`.
    pub log_filter: Option<String>,
    pub esc_exits: bool,
    /// `0` disables the budget.
    pub render_error_budget: u32,
    /// Larger than `render_error_budget`: a DX12 sleep/resume takes frames to settle.
    pub surface_error_budget: u32,
    /// The UI pass is single-sampled whatever this says. `1` disables MSAA.
    pub msaa_samples: u32,
    /// Ignored on native.
    pub wasm: WasmConfig,
}

#[derive(Clone)]
pub struct WasmConfig {
    /// Must carry `data-mode="manual"`; anything else auto-launches on load.
    pub host_id: String,
    pub button_id: String,
    pub canvas_id: String,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            host_id: "loam-canvas-host".into(),
            button_id: "loam-launch".into(),
            canvas_id: "loam-canvas".into(),
        }
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            window: WindowAttributes::default()
                .with_title("loam app")
                .with_visible(false),
            fixed_hz: 60,
            max_ticks_per_frame: DEFAULT_MAX_TICKS_PER_FRAME,
            log_filter: None,
            esc_exits: true,
            render_error_budget: 8,
            surface_error_budget: 32,
            msaa_samples: 1,
            wasm: WasmConfig::default(),
        }
    }
}

/// Dispatches native, wasm main-thread, and wasm worker mode.
pub fn run<A: App + 'static>(config: RunConfig) -> anyhow::Result<()> {
    #[cfg(target_arch = "wasm32")]
    {
        if wasm::is_worker_context() {
            return wasm::worker::run::<A>();
        }
        if wasm::launch::is_manual_mode(&config.wasm.host_id) {
            return wasm::launch_on_click(
                &config.wasm.host_id,
                &config.wasm.button_id,
                &config.wasm.canvas_id,
            );
        }
    }
    run_with_config::<A>(config)
}

/// On native this blocks until the event loop exits.
pub fn run_with_config<A: App>(config: RunConfig) -> anyhow::Result<()> {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        tracing_wasm::set_as_global_default();
        loam_time::frame_trace::set_heap_sampler(wasm::js_heap_sampler);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        let filter = match &config.log_filter {
            Some(s) => tracing_subscriber::EnvFilter::new(s.clone()),
            None => tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        };
        let _ = tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer())
            .with(log::ConsoleLayer)
            .try_init();
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let runner = Runner::<A>::new(config);

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(runner);
        Ok(())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut runner = runner;
        event_loop.run_app(&mut runner)?;
        runner.finish()
    }
}

struct InitArtifacts<A: App> {
    rd: RenderDevice,
    shader_db: ShaderDb,
    watcher: Option<FileWatcher>,
    ui: UiIntegration,
    app: A,
}

// The MSAA scene attachment resolves into the swapchain before egui paints.
const UI_PASS_SAMPLE_COUNT: u32 = 1;

fn setup_after_device<A: App>(
    win: &Arc<Window>,
    rd: RenderDevice,
    runtime: &Runtime,
) -> anyhow::Result<InitArtifacts<A>> {
    let mut shader_db = ShaderDb::new(rd.device.clone());

    let mut watcher = match FileWatcher::new() {
        Ok(w) => Some(w),
        Err(e) => {
            tracing::warn!("FileWatcher disabled: {e}");
            None
        }
    };

    let mut ctx = SetupCtx {
        runtime,
        rd: &rd,
        shader_db: &mut shader_db,
        watcher: watcher.as_mut(),
        time: 0.0,
    };
    let app = A::setup(&mut ctx).map_err(|e| e.context("App::setup"))?;

    // `ui_format`: the UI pass draws through the swapchain's non-sRGB view.
    let mut ui = UiIntegration::new(&rd.device, win, rd.ui_format(), UI_PASS_SAMPLE_COUNT);

    ui.warm_pipelines(
        &rd.device,
        &rd.queue,
        win,
        rd.ui_format(),
        UI_PASS_SAMPLE_COUNT,
    );
    rd.warm_composite();

    Ok(InitArtifacts {
        rd,
        shader_db,
        watcher,
        ui,
        app,
    })
}

#[cfg(target_arch = "wasm32")]
type PendingInit<A> = std::rc::Rc<std::cell::RefCell<Option<anyhow::Result<InitArtifacts<A>>>>>;

#[cfg(target_arch = "wasm32")]
fn attach_canvas_to_dom(win: &winit::window::Window) -> anyhow::Result<()> {
    use winit::platform::web::WindowExtWebSys;

    let canvas = win
        .canvas()
        .ok_or_else(|| anyhow::anyhow!("winit window has no canvas (wasm32 only)"))?;

    let web_window =
        web_sys::window().ok_or_else(|| anyhow::anyhow!("no global `window` object"))?;
    let document = web_window
        .document()
        .ok_or_else(|| anyhow::anyhow!("no `document` on global window"))?;

    let host: web_sys::Element = match document.get_element_by_id("loam-canvas-host") {
        Some(el) => el,
        None => document.body().map(Into::into).ok_or_else(|| {
            anyhow::anyhow!("no canvas host: page is missing both `#loam-canvas-host` and `<body>`")
        })?,
    };

    // Without these the canvas keeps winit's 1024x768 intrinsic size.
    let style = canvas.style();
    let _ = style.set_property("width", "100%");
    let _ = style.set_property("height", "100%");
    let _ = style.set_property("display", "block");

    host.append_child(&canvas)
        .map_err(|e| anyhow::anyhow!("append canvas to host: {e:?}"))?;

    Ok(())
}

struct Runner<A: App> {
    runtime: Runtime,
    config: RunConfig,

    commands: command::CommandQueue,
    timestep: FixedTimestep,
    input: InputState,
    start: Instant,

    window: Option<Arc<Window>>,
    artifacts: Option<InitArtifacts<A>>,

    /// Read at `begin_frame`; window events between frames see this frame's value.
    ui_capture: UiCapture,

    #[cfg(target_arch = "wasm32")]
    pending_init: Option<PendingInit<A>>,

    minimized: bool,

    last_fps_update: Instant,
    frame_count: u32,
    fps: f32,

    last_update_at: Option<Instant>,

    last_redraw_at: Option<Instant>,
    render_error_streak: u32,
    surface_error_streak: u32,
    last_surface_error_log: Option<Instant>,
    deferred_error: Option<anyhow::Error>,

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    capture: capture::Capture,
}

impl<A: App> Runner<A> {
    fn new(config: RunConfig) -> Self {
        let timestep =
            FixedTimestep::new(config.fixed_hz).with_max_catch_up(config.max_ticks_per_frame);
        Self {
            config,
            runtime: Runtime::default(),
            commands: command::CommandQueue::new(),
            timestep,
            input: InputState::default(),
            start: Instant::now(),
            window: None,
            artifacts: None,
            ui_capture: UiCapture::default(),
            #[cfg(target_arch = "wasm32")]
            pending_init: None,
            minimized: false,
            last_fps_update: Instant::now(),
            frame_count: 0,
            fps: 0.0,
            last_update_at: None,
            last_redraw_at: None,
            render_error_streak: 0,
            surface_error_streak: 0,
            last_surface_error_log: None,
            deferred_error: None,

            #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
            capture: capture::Capture::new(),
        }
    }

    // Setup and render failures call `elwt.exit()`, so the loop returns `Ok`.
    #[cfg(not(target_arch = "wasm32"))]
    fn finish(self) -> anyhow::Result<()> {
        match self.deferred_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    fn time(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }

    fn install_init(&mut self, win: Arc<Window>, artifacts: InitArtifacts<A>) {
        self.window = Some(win.clone());
        self.artifacts = Some(artifacts);
        self.minimized = false;
        self.start = Instant::now();
        self.last_fps_update = Instant::now();

        win.set_visible(true);
        win.request_redraw();
    }

    #[cfg(target_arch = "wasm32")]
    fn poll_pending_init(&mut self, elwt: &ActiveEventLoop) -> bool {
        let Some(cell) = self.pending_init.as_ref() else {
            return false;
        };
        let Some(result) = cell.borrow_mut().take() else {
            return false;
        };
        self.pending_init = None;
        let Some(win) = self.window.clone() else {
            self.deferred_error = Some(anyhow::anyhow!(
                "wasm init future resolved with no window present",
            ));
            elwt.exit();
            return true;
        };
        match result {
            Ok(artifacts) => {
                self.install_init(win, artifacts);
                true
            }
            Err(e) => {
                self.deferred_error = Some(e);
                elwt.exit();
                true
            }
        }
    }
}

#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
fn capture_consume(
    capture: &mut capture::Capture,
    rd: &RenderDevice,
    texture: &wgpu::Texture,
    is_pre: bool,
    captured_at: Instant,
) {
    let img = match capture::read_texture_rgba(
        &rd.device,
        &rd.queue,
        texture,
        rd.surface_bundle.size.width,
        rd.surface_bundle.size.height,
        rd.surface_bundle.config.format,
    ) {
        Ok(i) => i,
        Err(e) => {
            tracing::error!("capture: readback failed: {e:#}");
            return;
        }
    };
    if let Err(e) = capture.consume_frame(is_pre, img.rgba, img.width, img.height, captured_at) {
        tracing::error!("capture: write failed: {e:#}");
    }
}

impl<A: App> ApplicationHandler for Runner<A> {
    #[cfg(target_arch = "wasm32")]
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        // `prevent_default` would swallow Ctrl+R and F12 before browser chrome sees them.
        use winit::platform::web::WindowAttributesExtWebSys;
        let attrs = self.config.window.clone().with_prevent_default(false);
        let win = match elwt.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.deferred_error = Some(anyhow::anyhow!("create_window: {e}"));
                elwt.exit();
                return;
            }
        };

        if let Err(e) = attach_canvas_to_dom(&win) {
            self.deferred_error = Some(e.context("attach canvas to DOM"));
            elwt.exit();
            return;
        }

        let msaa = self.config.msaa_samples;
        let win_for_future = win.clone();
        let runtime = self.runtime.clone();
        let cell: PendingInit<A> = std::rc::Rc::new(std::cell::RefCell::new(None));
        let cell_for_future = cell.clone();

        self.window = Some(win);
        self.pending_init = Some(cell);

        wasm_bindgen_futures::spawn_local(async move {
            let result = async {
                let rd = RenderDevice::new(win_for_future.clone(), msaa)
                    .await
                    .map_err(|e| anyhow::anyhow!("RenderDevice::new: {e:#}"))?;
                setup_after_device::<A>(&win_for_future, rd, &runtime)
            }
            .await;
            *cell_for_future.borrow_mut() = Some(result);
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        let win = match elwt.create_window(self.config.window.clone()) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.deferred_error = Some(anyhow::anyhow!("create_window: {e}"));
                elwt.exit();
                return;
            }
        };

        let rd = match pollster::block_on(RenderDevice::new(win.clone(), self.config.msaa_samples))
        {
            Ok(r) => r,
            Err(e) => {
                self.deferred_error = Some(anyhow::anyhow!("RenderDevice::new: {e:#}"));
                elwt.exit();
                return;
            }
        };
        let artifacts = match setup_after_device::<A>(&win, rd, &self.runtime) {
            Ok(a) => a,
            Err(e) => {
                self.deferred_error = Some(e);
                elwt.exit();
                return;
            }
        };

        self.install_init(win, artifacts);
    }

    fn window_event(
        &mut self,
        elwt: &ActiveEventLoop,
        _id: winit::window::WindowId,
        ev: WindowEvent,
    ) {
        #[cfg(target_arch = "wasm32")]
        let _installed = self.poll_pending_init(elwt);

        let Some(win) = self.window.clone() else {
            return;
        };

        if log::events_enabled() {
            match &ev {
                WindowEvent::CursorMoved { .. }
                | WindowEvent::RedrawRequested
                | WindowEvent::AxisMotion { .. } => {}
                other => {
                    tracing::info!("WindowEvent: {other:?}");
                }
            }
        }

        // egui first, so it claims hover, focus and clicks before Loam's routing.
        if let Some(artifacts) = self.artifacts.as_mut() {
            let _ = artifacts.ui.handle_event(&win, &ev);
        }

        match &ev {
            WindowEvent::CloseRequested => {
                elwt.exit();
                return;
            }
            WindowEvent::KeyboardInput { event, .. }
                if self.config.esc_exits
                    && event.state == ElementState::Pressed
                    && matches!(event.logical_key, Key::Named(NamedKey::Escape))
                    && !self.ui_capture.keyboard =>
            {
                elwt.exit();
                return;
            }
            _ => {}
        }

        // Always route input *first*, before user `on_event` sees it.
        match &ev {
            WindowEvent::KeyboardInput { event, .. } => {
                self.input.key_input(event.physical_key, event.state);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.cursor_moved(position.x, position.y);
            }
            WindowEvent::CursorLeft { .. } => self.input.cursor_invalidated(),
            WindowEvent::Focused(false) => {
                self.runtime.request_release();
                self.input.cursor_invalidated();
                self.input.release_buttons();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.input.mouse_input(*button, *state);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.input.mouse_wheel(*delta);
            }
            WindowEvent::Resized(size) => {
                let was_minimized = self.minimized;
                self.minimized = size.width == 0 || size.height == 0;
                match (was_minimized, self.minimized) {
                    (false, true) => elwt.set_control_flow(ControlFlow::Wait),
                    (true, false) => {
                        let now = Instant::now();
                        self.timestep.reset_clock(now);
                        self.last_update_at = Some(now);
                        self.last_redraw_at = None;
                        elwt.set_control_flow(ControlFlow::Poll);
                        if let Some(artifacts) = &mut self.artifacts {
                            artifacts.rd.resize(*size);
                        }
                        win.request_redraw();
                    }
                    (false, false) => {
                        if let Some(artifacts) = &mut self.artifacts {
                            artifacts.rd.resize(*size);
                        }
                    }
                    (true, true) => {}
                }
            }
            _ => {}
        }

        if let WindowEvent::RedrawRequested = ev {
            self.redraw(elwt, &win);
            return;
        }

        let now = self.time();
        let fps = self.fps;
        let tick = self.timestep.tick();
        let ui_capture = self.ui_capture;
        if let Some(InitArtifacts {
            app, rd, shader_db, ..
        }) = self.artifacts.as_mut()
        {
            let mut ctx = FrameCtx {
                runtime: &self.runtime,
                shader_db,
                rd,
                input: FrameInput::default(),
                time: now,
                fps,
                n_ticks: 0,
                tick,
                dt: 0.0,
                ui_capture,
                _non_exhaustive: PhantomData,
            };
            app.on_event(&ev, &mut ctx);
            // Mirror the wasm worker: keyboard events also reach `on_key`.
            if let WindowEvent::KeyboardInput { event, .. } = &ev {
                if let winit::keyboard::PhysicalKey::Code(code) = event.physical_key {
                    app.on_key(code, event.state, &mut ctx);
                }
            }
        }
    }

    fn about_to_wait(&mut self, _elwt: &ActiveEventLoop) {
        #[cfg(target_arch = "wasm32")]
        {
            if self.pending_init.is_some() {
                self.poll_pending_init(_elwt);
            }
        }
    }

    fn device_event(
        &mut self,
        _elwt: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        ev: winit::event::DeviceEvent,
    ) {
        if let winit::event::DeviceEvent::MouseMotion { delta } = ev {
            self.input.accumulate_raw_motion(delta.0, delta.1);
        }
    }
}

// `crate::trace` subtracts these from `frame` to report `unscoped`.
pub(crate) const FRAME_LOOP_SECTIONS: &[&str] = &[
    "sim-ticks",
    "ui-begin",
    "app-update",
    "app-ui",
    "hot-reload",
    "surface-acquire",
    "app-record",
    "scene-resolve",
    "ui-paint",
    "composite",
    "present",
];

impl<A: App> Runner<A> {
    fn redraw(&mut self, elwt: &ActiveEventLoop, win: &Arc<Window>) {
        if self.minimized {
            return;
        }
        // Read before the frame's work so the last scripted frame presents.
        if self.runtime.exit_requested() {
            elwt.exit();
            return;
        }
        let Some(InitArtifacts {
            app,
            rd,
            shader_db,
            watcher,
            ui,
        }) = self.artifacts.as_mut()
        else {
            return;
        };
        self.runtime.apply_present_mode(rd);

        {
            use winit::window::CursorGrabMode;
            let (pending_grab, pending_vis) = self.runtime.take_cursor_request();
            let mut applied = self.runtime.cursor_state();
            if let Some(mode) = pending_grab {
                match cursor::apply_grab(mode, |mode| {
                    win.set_cursor_grab(match mode {
                        cursor::GrabMode::None => CursorGrabMode::None,
                        cursor::GrabMode::Confined => CursorGrabMode::Confined,
                        cursor::GrabMode::Locked => CursorGrabMode::Locked,
                    })
                }) {
                    Ok(mode) => applied.grab = mode,
                    Err(error) => tracing::warn!(%error, "cursor grab rejected"),
                }
            }
            if let Some(visible) = pending_vis {
                win.set_cursor_visible(visible);
                applied.visible = visible;
            }
            if pending_grab.is_some() || pending_vis.is_some() {
                self.runtime
                    .mark_cursor_applied(applied.grab, applied.visible);
            }
            // After the grab transition: warping a still-Locked cursor is a no-op.
            if self.runtime.take_warp_center() {
                let size = win.inner_size();
                let center = winit::dpi::PhysicalPosition::new(
                    size.width as f64 / 2.0,
                    size.height as f64 / 2.0,
                );
                let _ = win.set_cursor_position(center);
            }
        }

        // Anchor on the previous deadline, not the wake-up, so cadence stays locked.
        let now = Instant::now();
        let frame_anchor = if let (Some(period), Some(last)) =
            (self.runtime.target_period(), self.last_redraw_at)
        {
            let deadline = last + period;
            if now < deadline {
                #[cfg(target_arch = "wasm32")]
                {
                    win.request_redraw();
                    return;
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    frame_pacing::precise_sleep_until(deadline);
                    deadline
                }
            } else {
                now
            }
        } else {
            now
        };
        self.last_redraw_at = Some(frame_anchor);

        loam_time::frame_trace::begin_frame();

        let _frame_scope = loam_time::frame_trace::scope("frame");

        command::apply_drained(
            app,
            &self.runtime,
            shader_db,
            rd,
            self.timestep.tick(),
            self.start.elapsed().as_secs_f32(),
            &mut self.commands,
        );
        let n_ticks = drive_fixed_ticks(app, &mut self.timestep, Instant::now());

        // Opened before `App::update` so input hit-tests the last build's layout.
        let egui_ctx = {
            let _scope = loam_time::frame_trace::scope("ui-begin");
            let ctx = ui.begin_frame(win.as_ref()).clone();
            self.ui_capture = UiCapture::read(&ctx);
            ctx
        };
        let ui_capture = self.ui_capture;
        let input = self.input.take_frame();

        let now_inst = Instant::now();
        let dt = match self.last_update_at {
            Some(prev) => now_inst.saturating_duration_since(prev).as_secs_f32(),
            None => 1.0 / self.config.fixed_hz as f32,
        };
        self.last_update_at = Some(now_inst);

        {
            let mut fctx = FrameCtx {
                shader_db,
                runtime: &self.runtime,
                rd,
                input,
                time: self.start.elapsed().as_secs_f32(),
                fps: self.fps,
                n_ticks,
                tick: self.timestep.tick(),
                dt,
                ui_capture,
                _non_exhaustive: PhantomData,
            };
            {
                let _scope = loam_time::frame_trace::scope("app-update");
                app.update(&mut fctx);
            }

            {
                let _scope = loam_time::frame_trace::scope("app-ui");
                app.ui(&egui_ctx, &mut fctx);
            }
        }

        {
            let _scope = loam_time::frame_trace::scope("hot-reload");
            let reload_events = watcher.as_mut().map(|w| w.poll()).unwrap_or_default();
            if !reload_events.is_empty() {
                app.apply_shader_events(&reload_events, shader_db);
                let mut ctx = SetupCtx {
                    runtime: &self.runtime,
                    rd,
                    shader_db,
                    watcher: watcher.as_mut(),
                    time: self.start.elapsed().as_secs_f32(),
                };
                app.on_shader_reload(&mut ctx);
            }
        }

        self.frame_count += 1;
        let elapsed = self.last_fps_update.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            self.fps = self.frame_count as f32 / elapsed;
            self.frame_count = 0;
            self.last_fps_update = Instant::now();
            let title = app.title(self.fps);
            #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
            let title = match self.capture.status() {
                Some(status) => format!("{title} [{status}]").into(),
                None => title,
            };
            win.set_title(&title);
        }

        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
        {
            let requests = self.runtime.take_captures();
            if !requests.is_empty() {
                let log = self.capture.apply_requests(requests);
                for line in log {
                    tracing::info!("{line}");
                }
            }
        }

        // Each capture tap orders a composite ahead of its readback.
        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
        let capture_now = Instant::now();
        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
        let do_capture = self.capture.should_capture(capture_now);

        // Under `Fifo` the acquire blocks until the next flip, not `present`.
        let begin_result = {
            let _scope = loam_time::frame_trace::scope("surface-acquire");
            rd.begin_frame()
        };
        if begin_result.is_ok() {
            self.surface_error_streak = 0;
            self.last_surface_error_log = None;
        }
        match begin_result {
            Ok((frame, swap_view)) => {
                let mut last_err: Option<anyhow::Error> = None;
                let render_view = rd.msaa_view().or(rd.scene_view()).unwrap_or(&swap_view);

                // Separate submit so the start timestamp lands before the scene passes.
                if let Some(timer) = rd.gpu_timer.as_ref() {
                    let mut t_enc =
                        rd.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("loam-app::gpu-timer-start"),
                            });
                    timer.write_start(&mut t_enc);
                    rd.queue.submit(Some(t_enc.finish()));
                }

                let mut encoder =
                    rd.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("loam-app::frame"),
                        });

                {
                    let _scope = loam_time::frame_trace::scope("app-record");
                    let mut ctx = RenderCtx {
                        rd,
                        view: render_view,
                        encoder: &mut encoder,
                    };
                    if let Err(e) = app.record(&mut ctx) {
                        tracing::error!("App::record error: {e:#}");
                        last_err = Some(e);
                    }
                }

                if rd.sample_count() > 1 {
                    let _scope = loam_time::frame_trace::scope("scene-resolve");
                    rd.resolve_scene_to_swap(&mut encoder, &swap_view);
                }

                // Mid-frame submit so the GPU has drawn the scene before the readback.
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                if do_capture && self.capture.wants_pre() {
                    if rd.scene_view().is_some() {
                        rd.composite_to_swap(&mut encoder, &swap_view);
                    }
                    rd.queue.submit(Some(encoder.finish()));
                    encoder = rd
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("loam-app::frame-post-pre-capture"),
                        });
                    capture_consume(&mut self.capture, rd, &frame.texture, true, capture_now);
                }

                let mut callbacks = {
                    let _scope = loam_time::frame_trace::scope("ui-paint");
                    let viewport = (rd.surface_bundle.size.width, rd.surface_bundle.size.height);
                    let ui_swap_view =
                        (rd.scene_view().is_none()).then(|| rd.create_ui_swap_view(&frame));
                    let ui_view = match &ui_swap_view {
                        Some(swap) => swap,
                        None => render_view,
                    };
                    ui.paint(
                        &rd.device,
                        &rd.queue,
                        &mut encoder,
                        ui_view,
                        None,
                        win.as_ref(),
                        viewport,
                    )
                };

                // Before the post tap: the only pass that writes the swapchain here.
                if rd.scene_view().is_some() {
                    let _scope = loam_time::frame_trace::scope("composite");
                    rd.composite_to_swap(&mut encoder, &swap_view);
                }

                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                if do_capture && self.capture.wants_post() {
                    rd.queue
                        .submit(callbacks.drain(..).chain(Some(encoder.finish())));
                    encoder = rd
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("loam-app::frame-post-post-capture"),
                        });
                    capture_consume(&mut self.capture, rd, &frame.texture, false, capture_now);
                }
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                if do_capture {
                    self.capture.advance_frame(capture_now);
                }
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                self.runtime.publish_capture_status(self.capture.status());

                rd.queue
                    .submit(callbacks.drain(..).chain(Some(encoder.finish())));
                if let Some(timer) = rd.gpu_timer.as_ref() {
                    let mut t_enc =
                        rd.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("loam-app::gpu-timer-end"),
                            });
                    timer.write_end_and_resolve(&mut t_enc);
                    rd.queue.submit(Some(t_enc.finish()));
                }

                {
                    let _scope = loam_time::frame_trace::scope("present");
                    frame.present();
                }

                if let Some(timer) = rd.gpu_timer.as_mut() {
                    timer.tick();
                }
                if let Some(err) = last_err {
                    self.render_error_streak = self.render_error_streak.saturating_add(1);
                    let budget = self.config.render_error_budget;
                    if budget > 0 && self.render_error_streak >= budget {
                        self.deferred_error = Some(err.context(format!(
                            "App::record failed {budget} consecutive frames; aborting"
                        )));
                        elwt.exit();
                        return;
                    }
                } else {
                    self.render_error_streak = 0;
                }
                win.request_redraw();
            }
            Err(err) => {
                if matches!(err, wgpu::SurfaceError::OutOfMemory) {
                    self.deferred_error = Some(anyhow::anyhow!("wgpu surface out of memory"));
                    elwt.exit();
                    return;
                }

                // `Other` is DX12 after sleep/resume; `Timeout` is transient.
                match err {
                    wgpu::SurfaceError::Lost
                    | wgpu::SurfaceError::Outdated
                    | wgpu::SurfaceError::Other => {
                        let size = rd.surface_bundle.size;
                        rd.resize(size);
                    }
                    _ => {}
                }

                if matches!(err, wgpu::SurfaceError::Other) {
                    let now = Instant::now();
                    let should_log = self
                        .last_surface_error_log
                        .map(|t| now.duration_since(t).as_secs_f32() >= 1.0)
                        .unwrap_or(true);
                    if should_log {
                        tracing::error!("surface error: {err:?}");
                        self.last_surface_error_log = Some(now);
                    }
                } else {
                    tracing::debug!("surface error: {err:?}");
                }

                self.surface_error_streak = self.surface_error_streak.saturating_add(1);
                let budget = self.config.surface_error_budget;
                if budget > 0 && self.surface_error_streak >= budget {
                    self.deferred_error = Some(anyhow::anyhow!(
                        "wgpu surface error persisted {budget} consecutive frames: {err:?}"
                    ));
                    elwt.exit();
                    return;
                }
                win.request_redraw();
            }
        }

        drop(_frame_scope);
        loam_time::frame_trace::end_frame();
    }
}

/// Fills [`shell::BuildInfo`] in the calling crate; the build env vars are optional.
#[macro_export]
macro_rules! build_info {
    () => {
        $crate::shell::BuildInfo {
            crate_name: env!("CARGO_PKG_NAME"),
            crate_version: env!("CARGO_PKG_VERSION"),
            build_hash: option_env!("BUILD_HASH").unwrap_or(""),
            build_dirty: option_env!("BUILD_DIRTY").unwrap_or(""),
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    const TICK: Duration = Duration::from_nanos(1_000_000_000 / 60);

    #[derive(Default)]
    struct TickRecorder {
        times: Vec<f32>,
    }

    impl App for TickRecorder {
        fn setup(_ctx: &mut SetupCtx<'_>) -> anyhow::Result<Self> {
            Ok(Self::default())
        }

        fn tick(&mut self, _dt: f32, ctx: &mut TickCtx) {
            self.times.push(ctx.time);
        }
    }
    fn drive(base: Instant, offsets: &[Duration], max_catch_up: u32) -> (Vec<f32>, FixedTimestep) {
        let mut app = TickRecorder::default();
        let mut timestep = FixedTimestep::new(60).with_max_catch_up(max_catch_up);
        for offset in offsets {
            drive_fixed_ticks(&mut app, &mut timestep, base + *offset);
        }
        (app.times, timestep)
    }

    fn tick_times(base: Instant, offsets: &[Duration], max_catch_up: u32) -> Vec<f32> {
        drive(base, offsets, max_catch_up).0
    }

    #[test]
    fn tick_time_sequence_is_independent_of_frame_pacing() {
        let base = Instant::now();
        let one_per_frame: Vec<Duration> = (0..=60).map(|k| TICK * k).collect();
        let ten_per_frame: Vec<Duration> = (0..=6).map(|k| TICK * (k * 10)).collect();

        let smooth = tick_times(base, &one_per_frame, 10);
        let stuttered = tick_times(base, &ten_per_frame, 10);

        let expected: Vec<f32> = (0..60).map(|i| i as f32 * TICK.as_secs_f32()).collect();
        assert_eq!(smooth, expected, "tick time is tick_index * dt from zero");
        assert_eq!(
            stuttered, expected,
            "catching up ten ticks in one frame must yield the same time sequence"
        );
    }

    #[test]
    fn the_runner_caps_catch_up_in_the_accumulator_not_the_tick_loop() {
        let config = RunConfig {
            max_ticks_per_frame: 2,
            ..RunConfig::default()
        };
        let mut runner = Runner::<TickRecorder>::new(config);
        let base = Instant::now();
        runner.timestep.advance(base);
        let ticks = runner.timestep.advance(base + TICK * 10);
        assert_eq!(
            ticks.end - ticks.start,
            2,
            "RunConfig::max_ticks_per_frame must reach the accumulator, \
             which is the only place the cap may be applied"
        );
    }

    #[test]
    fn dropped_wall_time_does_not_skip_simulation_ticks() {
        for cap in [DEFAULT_MAX_TICKS_PER_FRAME, loam_time::DEFAULT_MAX_CATCH_UP] {
            let base = Instant::now();
            let mut app = TickRecorder::default();
            let mut timestep = FixedTimestep::new(60).with_max_catch_up(cap);

            let steps = [
                Duration::ZERO,
                TICK,
                TICK,
                TICK,
                TICK * 30,
                TICK,
                TICK,
                TICK,
            ];
            let mut elapsed = Duration::ZERO;
            for step in steps {
                elapsed += step;
                drive_fixed_ticks(&mut app, &mut timestep, base + elapsed);
                assert_eq!(
                    app.times.len() as u64,
                    timestep.tick(),
                    "cap {cap}: a booked tick was never simulated at {elapsed:?}"
                );
            }

            let expected_ticks = 3 + u64::from(cap) + 3;
            assert_eq!(timestep.tick(), expected_ticks);
            let expected_times: Vec<f32> = (0..expected_ticks)
                .map(|i| i as f32 * TICK.as_secs_f32())
                .collect();
            assert_eq!(app.times, expected_times, "cap {cap}");
        }
    }
}
