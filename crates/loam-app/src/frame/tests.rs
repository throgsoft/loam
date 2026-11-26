use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use glam::Vec4;
use loam_math::EuclideanR4;
use loam_render::device::FeatureRequest;
use loam_render::pass::{FrameFormat, FramePass, FrameTarget, PassFault, PassStage};
use loam_runtime::{
    ActionId, Command, DomainBuilder, FreeCamera, LogCapacity, Material, Phase, PointerPhase, Pose,
    PreparedGeometry, Rig, Section4, SimConfig, SpawnBundle, Tick, ViewSpec,
};
use wgpu::{
    BackendOptions, Backends, Extent3d, Instance, InstanceDescriptor, NoopBackendOptions,
    TextureDescriptor, TextureDimension, TextureUsages, TextureViewDescriptor,
};

use loam_time::par::Sequential;

use super::*;
use crate::args::Args;
use crate::host::Bindings;
use loam_runtime::CursorPolicy;

const FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;
const SIZE: (u32, u32) = (64, 48);
loam_runtime::stores! {
    #[derive(Default)]
    pub struct Bare {}
}

#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
const CAPTURE_FPS: u16 = 60;

struct Probe {
    recorded: Arc<AtomicU32>,
    rebuilt: Arc<AtomicU32>,
}

impl FramePass for Probe {
    fn name(&self) -> &'static str {
        "probe"
    }

    fn stage(&self) -> PassStage {
        PassStage::Scene
    }

    fn record(
        &mut self,
        _encoder: &mut CommandEncoder,
        _target: &FrameTarget<'_>,
    ) -> Result<(), PassFault> {
        self.recorded.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn attach(&mut self, _gpu: &GpuContext, _frame: FrameFormat) -> Result<(), PassFault> {
        self.rebuilt.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

fn noop_gpu() -> GpuContext {
    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::NOOP,
        backend_options: BackendOptions {
            noop: NoopBackendOptions { enable: true },
            ..Default::default()
        },
        ..Default::default()
    });
    pollster::block_on(GpuContext::new(instance, FeatureRequest::default(), None))
        .expect("the noop backend always yields a context")
}

fn offscreen(gpu: &GpuContext) -> Texture {
    gpu.device.create_texture(&TextureDescriptor {
        label: Some("session frame probe"),
        size: Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn bare(app: SessionApp<Bare>) -> Frame<Bare> {
    Frame::new(
        Session::new(Bare::default(), SimConfig::default()),
        app,
        Box::new(Sequential),
    )
    .expect("the frame accepted the simulation config")
}

fn host<A: Stores>(name: &'static str) -> SessionApp<A> {
    quiet(SessionApp::with_args(
        HostConfig::new(name, Bindings::new()),
        Args::default(),
    ))
}

fn quiet<A: Stores>(app: SessionApp<A>) -> SessionApp<A> {
    app.debug_layer(false)
}

fn run_one<A: Stores>(frame: &mut Frame<A>, gpu: &GpuContext, texture: &Texture) {
    run_at(frame, gpu, texture, Instant::now());
}

fn run_at<A: Stores>(frame: &mut Frame<A>, gpu: &GpuContext, texture: &Texture, now: Instant) {
    let view = texture.create_view(&TextureViewDescriptor::default());
    let target = Target {
        view: &view,
        texture,
        format: FORMAT,
        size: SIZE,
    };
    frame
        .step(gpu, &target, now, |_| {})
        .expect("the frame stepped");
}

fn world(moves: bool) -> Session<Bare> {
    let mut session = Session::new(Bare::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let geometry = session.prepare(PreparedGeometry::Lines4 {
        segments: vec![[[0.0; 4], [1.0, 0.0, 0.0, 0.0]]],
    });
    let material = session.add_material(Material::lines([1.0; 4], 1.0));
    let root = session.views().root_id();
    let body = session
        .dispatch(|dispatch| {
            let eye = dispatch.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::Z * 4.0)))?;
            dispatch
                .domains
                .typed(r4)?
                .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))?;
            dispatch.spawn(
                SpawnBundle::new()
                    .at(r4, Pose::at(Vec4::ZERO))
                    .instance(loam_runtime::Instance::new(geometry, material)),
            )
        })
        .expect("the world spawned");
    session.system(
        Phase::Simulation,
        "drift",
        move |ctx: loam_runtime::Ctx<'_, Bare>| -> Result<(), loam_runtime::DomainError> {
            if moves {
                let x = (ctx.step.tick.0 + 1) as f32;
                ctx.domains
                    .typed(r4)?
                    .set_pose(body, Pose::at(Vec4::X * x))?;
            }
            Ok(())
        },
    );
    session
}

fn on_change(
    session: Session<Bare>,
    app: SessionApp<Bare>,
    gpu: &GpuContext,
) -> (Frame<Bare>, Arc<AtomicU32>) {
    let recorded = Arc::new(AtomicU32::new(0));
    let app = app.redraw(Redraw::OnChange).pass(Box::new(Probe {
        recorded: recorded.clone(),
        rebuilt: Arc::new(AtomicU32::new(0)),
    }));
    let mut frame = Frame::new(session, app, Box::new(Sequential))
        .expect("the frame accepted the simulation config");
    frame
        .attach(gpu, FORMAT, None, None, SIZE, 1.0)
        .expect("attached");
    (frame, recorded)
}

#[test]
fn on_change_content_or_scale() {
    let gpu = noop_gpu();
    let texture = offscreen(&gpu);
    let hooked = Arc::new(AtomicU32::new(0));
    let hooks = hooked.clone();
    let app = host("still").on_published(move |_| {
        hooks.fetch_add(1, Ordering::Relaxed);
    });
    let (mut frame, recorded) = on_change(world(false), app, &gpu);
    let start = Instant::now();
    for millis in [0, 20, 25] {
        run_at(
            &mut frame,
            &gpu,
            &texture,
            start + Duration::from_millis(millis),
        );
    }
    assert_eq!(recorded.load(Ordering::Relaxed), 1);
    assert_eq!(hooked.load(Ordering::Relaxed), 3);
    frame.resize(SIZE.0, SIZE.1, 2.0);
    run_at(
        &mut frame,
        &gpu,
        &texture,
        start + Duration::from_millis(30),
    );
    assert_eq!(recorded.load(Ordering::Relaxed), 2);

    let (mut frame, recorded) = on_change(world(true), host("moving"), &gpu);
    for millis in [0, 20, 25] {
        run_at(
            &mut frame,
            &gpu,
            &texture,
            start + Duration::from_millis(millis),
        );
    }
    assert_eq!(recorded.load(Ordering::Relaxed), 3);
}

#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
#[test]
fn on_change_capture_request() {
    let gpu = noop_gpu();
    let texture = offscreen(&gpu);
    let mut frames = 0_u32;
    let app = host("still").on_published(move |hook| {
        frames += 1;
        if frames == 3 {
            hook.capture.stop().expect("capture supported");
        }
    });
    let (mut frame, recorded) = on_change(world(false), app, &gpu);
    let start = Instant::now();
    for (millis, drawn) in [(0, 1), (20, 1), (25, 2), (30, 2)] {
        run_at(
            &mut frame,
            &gpu,
            &texture,
            start + Duration::from_millis(millis),
        );
        assert_eq!(
            recorded.load(Ordering::Relaxed),
            drawn,
            "draws after the frame at {millis} ms"
        );
    }
}

#[test]
fn pass_after_device_recovery() {
    let gpu = noop_gpu();
    let texture = offscreen(&gpu);
    let recorded = Arc::new(AtomicU32::new(0));
    let rebuilt = Arc::new(AtomicU32::new(0));
    let app = host("passes").pass(Box::new(Probe {
        recorded: recorded.clone(),
        rebuilt: rebuilt.clone(),
    }));
    let mut frame = bare(app);
    frame
        .attach(&gpu, FORMAT, None, None, SIZE, 1.0)
        .expect("attached");
    run_one(&mut frame, &gpu, &texture);
    assert_eq!(recorded.load(Ordering::Relaxed), 1);
    assert_eq!(rebuilt.load(Ordering::Relaxed), 1);

    frame.recover(&gpu).expect("recovered");
    run_one(&mut frame, &gpu, &texture);
    assert_eq!(
        rebuilt.load(Ordering::Relaxed),
        2,
        "recovery never rebuilt the application's pass on the new device"
    );
    assert_eq!(
        recorded.load(Ordering::Relaxed),
        2,
        "the recovered presenter no longer records the application's pass"
    );
}

#[test]
fn post_across_frames() {
    let gpu = noop_gpu();
    let texture = offscreen(&gpu);
    let mut frames = 0_u32;
    let app = host::<Bare>("posts").on_published(move |hook| {
        frames += 1;
        if frames == 1 {
            hook.post("rect", &[1.0, 2.0, 3.0, 4.0]);
            hook.post("ready", &[]);
        }
    });
    let mut frame = bare(app);
    frame
        .attach(&gpu, FORMAT, None, None, SIZE, 1.0)
        .expect("attached");

    run_one(&mut frame, &gpu, &texture);
    let posted: Vec<(&str, Vec<f32>)> = frame
        .posts()
        .map(|(topic, values)| (topic, values.to_vec()))
        .collect();
    assert_eq!(
        posted,
        [("rect", vec![1.0, 2.0, 3.0, 4.0]), ("ready", Vec::new())]
    );

    run_one(&mut frame, &gpu, &texture);
    assert_eq!(frame.posts().count(), 0);
}

#[test]
fn frame_between_ticks() {
    let gpu = noop_gpu();
    let texture = offscreen(&gpu);
    let mut session = Session::new(Bare::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let geometry = session.prepare(PreparedGeometry::Lines4 {
        segments: vec![[[0.1, 0.0, 0.0, 0.0], [-0.1, 0.0, 0.0, 0.0]]],
    });
    let material = session.add_material(Material::flat([1.0; 4]));
    let root = session.views().root_id();
    let body = session
        .dispatch(|dispatch| {
            let eye = dispatch.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::Z * 4.0)))?;
            dispatch
                .domains
                .typed(r4)?
                .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))?;
            dispatch.spawn(
                SpawnBundle::new()
                    .at(r4, Pose::at(Vec4::ZERO))
                    .instance(loam_runtime::Instance::new(geometry, material)),
            )
        })
        .expect("the body spawned");
    session.system(
        Phase::Simulation,
        "move",
        move |ctx: loam_runtime::Ctx<'_, Bare>| {
            if ctx.step.tick == Tick(0) {
                ctx.domains.typed(r4)?.set_pose(body, Pose::at(Vec4::X))?;
            }
            Ok(())
        },
    );
    let mut frame = Frame::new(session, host("alpha"), Box::new(Sequential))
        .expect("the frame accepted the config");
    frame
        .attach(&gpu, FORMAT, None, None, SIZE, 1.0)
        .expect("attached");
    let start = Instant::now();
    let dt = Duration::from_nanos(1_000_000_000 / u64::from(SimConfig::default().fixed_hz));
    run_at(&mut frame, &gpu, &texture, start);
    run_at(&mut frame, &gpu, &texture, start + dt * 3 / 2);

    let publication = &frame.state.publication;
    let x = publication.views[0].records.instances.rows()[0]
        .pose
        .coordinates[0];
    assert!((x - 0.5).abs() < 1e-3, "the body was drawn at x = {x}");
}

#[test]
fn publication_eye_write() {
    let gpu = noop_gpu();
    let texture = offscreen(&gpu);
    let app = host::<Bare>("aspect");
    let mut session = Session::new(Bare::default(), SimConfig::default());
    session.register_domain(DomainBuilder::new("r4", EuclideanR4).untracked());
    session.system(
        Phase::Publication,
        "camera",
        |ctx: loam_runtime::Ctx<'_, Bare>| {
            ctx.views.root_mut().eye = Eye::looking_at([0.0, 2.0, 5.0], [0.0; 3], [0.0, 1.0, 0.0]);
            Ok(())
        },
    );
    let mut frame = Frame::new(session, app, Box::new(Sequential))
        .expect("the frame accepted the simulation config");
    frame
        .attach(&gpu, FORMAT, None, None, SIZE, 1.0)
        .expect("attached");
    run_at(&mut frame, &gpu, &texture, Instant::now());

    let aspect = frame.state.session.views().root().eye.aspect;
    assert!(
        (aspect - SIZE.0 as f32 / SIZE.1 as f32).abs() <= 1e-6,
        "a Publication-phase eye write published aspect {aspect}"
    );
}

#[test]
fn reset_key_after_fault() {
    const RESET: ActionId = ActionId("reset");

    let gpu = noop_gpu();
    let texture = offscreen(&gpu);
    let recorded = Arc::new(AtomicU32::new(0));
    let filled = Arc::new(AtomicU32::new(0));
    let fills = filled.clone();
    let app = quiet(SessionApp::<Bare>::with_args(
        HostConfig::new(
            "fault recovery",
            Bindings::new().key(Key::Letter('r'), RESET),
        ),
        Args::default(),
    ))
    .recover_on_fault(RESET)
    .pass(Box::new(Probe {
        recorded: recorded.clone(),
        rebuilt: Arc::new(AtomicU32::new(0)),
    }))
    .on_published(move |hook| {
        fills.fetch_add(1, Ordering::Relaxed);
        hook.capture_cursor(true, CursorPolicy::Toggle);
    });
    let mut session = Session::new(Bare::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let root = session.views().root_id();
    session
        .dispatch(|dispatch| {
            let eye = dispatch.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))?;
            dispatch
                .domains
                .typed(r4)?
                .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))?;
            Ok::<_, loam_runtime::Rejection>(())
        })
        .expect("view");
    let armed = Arc::new(AtomicBool::new(false));
    let trip = armed.clone();
    session.system(
        Phase::Publication,
        "trip",
        move |_ctx: loam_runtime::Ctx<'_, Bare>| {
            if trip.swap(false, Ordering::Relaxed) {
                return Err(loam_runtime::DomainError::InvalidFrame);
            }
            Ok(())
        },
    );
    session.system(
        Phase::Dispatch,
        "authored reset",
        |ctx: loam_runtime::Ctx<'_, Bare>| {
            if ctx.input.pressed(RESET) {
                ctx.commands.submit(Command::Reset);
            }
            Ok(())
        },
    );
    session.set_initial().expect("initial state");
    let epoch = session.scene().epoch();
    let mut frame = Frame::new(session, app, Box::new(Sequential))
        .expect("the frame accepted the simulation config");
    frame
        .attach(&gpu, FORMAT, None, None, SIZE, 1.0)
        .expect("attached");
    let start = Instant::now();

    run_at(&mut frame, &gpu, &texture, start);
    assert_eq!(frame.take_cursor_request(), Some(true));
    frame.cursor_applied(true);
    armed.store(true, Ordering::Relaxed);
    run_at(&mut frame, &gpu, &texture, start + Duration::from_millis(1));
    assert_eq!(
        frame.phase_error().map(|error| error.phase),
        Some(Phase::Publication)
    );
    assert_eq!(filled.load(Ordering::Relaxed), 1);
    assert_eq!(recorded.load(Ordering::Relaxed), 2);
    assert_eq!(frame.take_cursor_request(), Some(false));
    frame.cursor_applied(false);
    for millis in 2_u64..=4 {
        run_at(
            &mut frame,
            &gpu,
            &texture,
            start + Duration::from_millis(millis),
        );
    }
    assert_eq!(recorded.load(Ordering::Relaxed), 5);

    frame.action(Key::Letter('r'), true, false);
    run_at(&mut frame, &gpu, &texture, start + Duration::from_secs(60));
    assert_eq!(recorded.load(Ordering::Relaxed), 6);
    assert_eq!(filled.load(Ordering::Relaxed), 2);

    assert_eq!(frame.phase_error(), None);
    assert_eq!(frame.state.session.scene().epoch(), epoch.advance());
    assert_eq!(frame.state.session.current_tick(), Tick(0));
    assert_eq!(filled.load(Ordering::Relaxed), 2);
    assert_eq!(frame.take_cursor_request(), Some(true));
    run_at(
        &mut frame,
        &gpu,
        &texture,
        start + Duration::from_secs(60) + Duration::from_millis(1),
    );
    assert_eq!(frame.state.session.scene().epoch(), epoch.advance());
}

#[test]
fn cursor_request_in_free_rig() {
    const RESET: ActionId = ActionId("reset");
    const LAND: ActionId = ActionId("land");

    let gpu = noop_gpu();
    let texture = offscreen(&gpu);
    let own = Arc::new(AtomicBool::new(false));
    let owns = own.clone();
    let app = quiet(SessionApp::<Bare>::with_args(
        HostConfig::new(
            "rig cursor",
            Bindings::new()
                .key(Key::Letter('r'), RESET)
                .key(Key::Letter('l'), LAND),
        ),
        Args::default(),
    ))
    .recover_on_fault(RESET)
    .on_published(move |hook| {
        if owns.load(Ordering::Relaxed) {
            hook.capture_cursor(true, CursorPolicy::Toggle);
        }
    });
    let mut session = Session::new(Bare::default(), SimConfig::default());
    session.set_rig(Rig::Free(FreeCamera::default()));
    session.on_action(LAND, |dispatch| {
        dispatch.views.root_mut().rig = Rig::Manual;
        Ok(())
    });
    let armed = Arc::new(AtomicBool::new(false));
    let trip = armed.clone();
    session.system(
        Phase::Publication,
        "trip",
        move |_ctx: loam_runtime::Ctx<'_, Bare>| {
            if trip.swap(false, Ordering::Relaxed) {
                return Err(loam_runtime::DomainError::InvalidFrame);
            }
            Ok(())
        },
    );
    let mut frame = Frame::new(session, app, Box::new(Sequential))
        .expect("the frame accepted the simulation config");
    frame
        .attach(&gpu, FORMAT, None, None, SIZE, 1.0)
        .expect("attached");
    let start = Instant::now();
    let mut millis = 0;
    let mut run = |frame: &mut Frame<Bare>| {
        millis += 1;
        run_at(frame, &gpu, &texture, start + Duration::from_millis(millis));
        frame.take_cursor_request()
    };

    assert_eq!(
        run(&mut frame),
        Some(true),
        "a free rig set before launch never asked for the cursor"
    );
    frame.cursor_applied(true);
    assert_eq!(run(&mut frame), None);
    frame.alt(0, true);
    assert_eq!(frame.take_cursor_request(), Some(false));
    frame.cursor_applied(false);
    frame.alt(0, false);
    assert_eq!(
        run(&mut frame),
        None,
        "the rig took back a cursor the user freed with Alt"
    );

    armed.store(true, Ordering::Relaxed);
    assert_eq!(run(&mut frame), Some(false));
    assert_eq!(
        frame.phase_error().map(|error| error.phase),
        Some(Phase::Publication)
    );
    frame.action(Key::Letter('r'), true, false);
    assert_eq!(
        run(&mut frame),
        Some(true),
        "the recovered free rig did not ask for the cursor again"
    );
    assert_eq!(frame.phase_error(), None);
    frame.cursor_applied(true);

    frame.action(Key::Letter('l'), true, false);
    assert_eq!(
        run(&mut frame),
        Some(false),
        "the cursor stayed captured after the program left the free rig"
    );
    frame.cursor_applied(false);
    assert_eq!(run(&mut frame), None);

    own.store(true, Ordering::Relaxed);
    assert_eq!(run(&mut frame), Some(true));
    frame.cursor_applied(true);
    for _ in 0..2 {
        assert_eq!(
            run(&mut frame),
            None,
            "the shell re-issued or released a capture the frame hook holds under a manual rig"
        );
    }
}

#[test]
fn scripted_line() {
    let gpu = noop_gpu();
    let texture = offscreen(&gpu);
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("demo.script");
    std::fs::write(&path, "0 mark first\n2 mark second\n").expect("wrote the script");

    let marked: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = marked.clone();
    let args = Args::from_argv([format!("--script={}", path.display())]);
    let app = quiet(SessionApp::<Bare>::with_args(
        HostConfig::new("script", Bindings::new()),
        args,
    ))
    .command("mark", "record a marker", move |args, _submit, _out| {
        recorded
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(args.join(" "));
        Ok(())
    });
    let mut frame = bare(app);
    frame
        .attach(&gpu, FORMAT, None, None, SIZE, 1.0)
        .expect("attached");

    let mut after_each = Vec::new();
    for _ in 0..3 {
        run_one(&mut frame, &gpu, &texture);
        after_each.push(
            marked
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .len(),
        );
    }

    assert_eq!(
        after_each,
        [1, 1, 2],
        "the host drove the script off its own frame index"
    );
    assert_eq!(
        *marked.lock().unwrap_or_else(|error| error.into_inner()),
        ["first", "second"]
    );
}

#[test]
fn scripted_input_lines() {
    const MARK: ActionId = ActionId("mark");

    let gpu = noop_gpu();
    let texture = offscreen(&gpu);
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("input.script");
    std::fs::write(&path, "0 key KeyM down\n0 pointer 0.25 0.5 down\n").expect("wrote the script");

    let args = Args::from_argv([format!("--script={}", path.display())]);
    let app = quiet(SessionApp::<Bare>::with_args(
        HostConfig::new("script", Bindings::new().key(Key::Letter('m'), MARK)),
        args,
    ));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let saw = seen.clone();
    let mut session = Session::new(Bare::default(), SimConfig::default());
    session.system(
        Phase::Dispatch,
        "watch",
        move |ctx: loam_runtime::Ctx<'_, Bare>| {
            let began = ctx
                .input
                .pointers
                .iter()
                .find(|pointer| pointer.phase == PointerPhase::Began)
                .map(|pointer| pointer.ndc);
            saw.lock()
                .unwrap_or_else(|error| error.into_inner())
                .push((ctx.input.pressed(MARK), began));
            Ok(())
        },
    );
    let mut frame = Frame::new(session, app, Box::new(Sequential))
        .expect("the frame accepted the simulation config");
    frame
        .attach(&gpu, FORMAT, None, None, SIZE, 1.0)
        .expect("attached");

    for _ in 0..2 {
        run_one(&mut frame, &gpu, &texture);
    }

    assert_eq!(
        *seen.lock().unwrap_or_else(|error| error.into_inner()),
        [(false, None), (true, Some([0.25, 0.5]))],
        "the boundary after the lines' frame did not get their key and pointer"
    );
}

struct Refuse;

impl loam_runtime::AppCommand<Bare> for Refuse {
    fn name(&self) -> &'static str {
        "reject"
    }

    fn apply(
        &mut self,
        _dispatch: &mut loam_runtime::Dispatch<'_, Bare>,
    ) -> Result<loam_runtime::Outcome, loam_runtime::Rejection> {
        Err(loam_runtime::Rejection::Refused("refused"))
    }
}

#[test]
fn scripted_refusals() {
    let gpu = noop_gpu();
    let texture = offscreen(&gpu);
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("refused.script");
    std::fs::write(&path, "0 strict\n0 mark first\n0 reject\n2 mark second\n")
        .expect("wrote the script");

    let marked = Arc::new(AtomicU32::new(0));
    let marks = marked.clone();
    let args = Args::from_argv([format!("--script={}", path.display())]);
    let app = quiet(SessionApp::<Bare>::with_args(
        HostConfig::new("script", Bindings::new()),
        args,
    ))
    .command("mark", "count a marker", move |_args, _submit, _out| {
        marks.fetch_add(1, Ordering::Relaxed);
        Ok(())
    })
    .command("strict", "refuse at dispatch", |_args, _submit, _out| {
        Err(loam_console::CommandError::Usage("strict <n>".into()))
    })
    .command("reject", "refuse at the boundary", |_args, submit, _out| {
        submit.app(Refuse);
        Ok(())
    });
    let mut frame = bare(app);
    frame
        .attach(&gpu, FORMAT, None, None, SIZE, 1.0)
        .expect("attached");

    let mut after_each = Vec::new();
    for _ in 0..3 {
        run_one(&mut frame, &gpu, &texture);
        after_each.push(marked.load(Ordering::Relaxed));
    }

    assert_eq!(
        after_each,
        [1, 1, 2],
        "a refused line stopped the lines after it or moved them off their frames"
    );
}

#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn hook_stopped_capture_gpu_probe() {
    let instance = Instance::default();
    let gpu = pollster::block_on(GpuContext::new(instance, FeatureRequest::default(), None))
        .expect("a wgpu adapter");
    let texture = offscreen(&gpu);
    let directory = tempfile::tempdir().expect("temp dir");
    let into = directory.path().to_path_buf();
    let mut frames = 0_u32;
    let app = host::<Bare>("capture").on_published(move |hook| {
        frames += 1;
        if frames == 1 {
            hook.capture
                .start(crate::capture::CaptureRequest::StartSequence {
                    format: crate::capture::CaptureFormat::Apng,
                    stage: crate::capture::CaptureStage::Post,
                    dir: Some(into.clone()),
                    name: Some("session".into()),
                    fps: Some(CAPTURE_FPS),
                    scale: None,
                    palette: crate::capture::PaletteMode::default(),
                })
                .expect("capture supported");
        }
        if frames == 3 {
            hook.capture.stop().expect("capture supported");
        }
    });
    let mut frame = bare(app);
    frame
        .attach(&gpu, FORMAT, None, None, SIZE, 1.0)
        .expect("attached");
    for _ in 0..4 {
        run_one(&mut frame, &gpu, &texture);
        std::thread::sleep(Duration::from_millis(1000 / u64::from(CAPTURE_FPS) + 4));
    }
    drop(frame);

    let written = directory.path().join("session.apng");
    let bytes = std::fs::read(&written).expect("the stopped capture finished its file");
    assert_eq!(
        animation_frames(&bytes),
        Some(2),
        "the file holds the frames drawn between the start and the stop"
    );
}

#[cfg(feature = "egui")]
#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn debug_layer_in_gamma_view_gpu_probe() {
    const VIEW: TextureFormat = TextureFormat::Rgba8Unorm;

    let instance = Instance::default();
    let gpu = pollster::block_on(GpuContext::new(instance, FeatureRequest::default(), None))
        .expect("a wgpu adapter");
    let texture = gpu.device.create_texture(&TextureDescriptor {
        label: Some("gamma view probe"),
        size: Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[VIEW],
    });
    let app = SessionApp::<Bare>::with_args(
        HostConfig::new("gamma view", Bindings::new()),
        Args::default(),
    )
    .on_published(|hook| {
        if let Some(ui) = hook.ui {
            loam_egui::egui::CentralPanel::default().show(ui, |ui| ui.label("layer"));
        }
    });
    let mut frame = bare(app);
    frame
        .attach(&gpu, FORMAT, Some(VIEW), None, SIZE, 1.0)
        .expect("attached");
    for _ in 0..3 {
        run_one(&mut frame, &gpu, &texture);
    }
    gpu.device.poll(wgpu::PollType::Poll).expect("polled");
    assert!(
        gpu.take_uncaptured_error().is_none(),
        "the layer drew into the view with a pipeline of another format"
    );
}

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Valued {}
    publishes u32;
}

#[test]
fn on_published_hook() {
    let gpu = noop_gpu();
    let texture = offscreen(&gpu);
    let mut session = Session::new(Valued::default(), SimConfig::default());
    let mut publishes = 0_u32;
    session.publish_value("publishes", move |_, value: &mut u32| {
        publishes += 1;
        *value = publishes;
        Ok(())
    });
    let seen: Arc<Mutex<Vec<u32>>> = Arc::default();
    let read = seen.clone();
    let app = host::<Valued>("value").on_published(move |hook| {
        read.lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(hook.published.value);
    });
    let mut frame = Frame::new(session, app, Box::new(Sequential))
        .expect("the frame accepted the simulation config");
    frame
        .attach(&gpu, FORMAT, None, None, SIZE, 1.0)
        .expect("attached");

    run_one(&mut frame, &gpu, &texture);
    run_one(&mut frame, &gpu, &texture);

    assert_eq!(
        *seen.lock().unwrap_or_else(|error| error.into_inner()),
        [1, 2],
        "the hook did not read each frame's published value"
    );
}

#[cfg(not(feature = "egui"))]
fn notes_after_three_frames(labelled: bool, app: SessionApp<Bare>) -> usize {
    let gpu = noop_gpu();
    let texture = offscreen(&gpu);
    let mut session = Session::new(Bare::default(), SimConfig::default());
    session.system(Phase::Publication, "label", move |ctx| {
        if labelled {
            ctx.overlay.push_text(
                "score",
                loam_runtime::TextRecord {
                    origin_pt: [8.0, 16.0],
                    size_pt: 16.0,
                    color: [1.0; 4],
                },
            );
        }
        Ok(())
    });
    let mut frame = Frame::new(session, app, Box::new(Sequential))
        .expect("the frame accepted the simulation config");
    frame
        .attach(&gpu, FORMAT, None, None, SIZE, 1.0)
        .expect("attached");
    for _ in 0..3 {
        run_one(&mut frame, &gpu, &texture);
    }
    frame.state.app.console.ui().history().len()
}

#[cfg(not(feature = "egui"))]
#[test]
fn text_without_font() {
    let silent = notes_after_three_frames(false, host("plain"));
    assert_eq!(
        notes_after_three_frames(true, host("fontless")),
        silent + 1,
        "three frames of dropped text did not leave exactly one note"
    );
    #[cfg(feature = "text")]
    {
        let font = include_bytes!("../../../examples/fonts/lmroman10-bold.otf").to_vec();
        assert_eq!(
            notes_after_three_frames(true, host("font").font(font)),
            silent,
            "text drawn with the app's font was reported as dropped"
        );
    }
}

#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
fn animation_frames(png: &[u8]) -> Option<u32> {
    let mut at = 8;
    while at + 12 <= png.len() {
        let length = u32::from_be_bytes(png[at..at + 4].try_into().ok()?) as usize;
        if &png[at + 4..at + 8] == b"acTL" {
            return Some(u32::from_be_bytes(png[at + 8..at + 12].try_into().ok()?));
        }
        at += 12 + length;
    }
    None
}
