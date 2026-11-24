use loam_math::{EuclideanR3, Space};
use loam_runtime::{
    ActionEvent, ActionId, CameraPair, CursorPolicy, Eye, FreeCamera, Input, Orbit, Pointer,
    PointerButton, PointerPhase, Rig, Session, SimConfig, MOVE_FORWARD, MOVE_RIGHT, MOVE_UP,
};
use loam_time::par::Sequential;

type Vec3 = <EuclideanR3 as Space>::Vector;

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Bare {}
}

const ASPECT: f32 = 1.5;
const TARGET: [f32; 3] = [1.0, 2.0, 3.0];
const DISTANCE: f32 = 4.0;

fn hosted(config: SimConfig, rig: Rig) -> Session<Bare> {
    let mut session = Session::new(Bare::default(), config);
    session.views_mut().root_mut().eye.aspect = ASPECT;
    session.set_rig(rig);
    session
}

fn eye(session: &Session<Bare>) -> Eye {
    let views = session.views();
    views.root().eye
}

fn rig(session: &Session<Bare>) -> Rig {
    let views = session.views();
    views.root().rig
}

fn fly(session: &mut Session<Bare>, flying: bool) {
    session.views_mut().root_mut().set_flying(flying);
}

fn walk_forward(session: &mut Session<Bare>, ticks: u32) {
    for _ in 0..ticks {
        session
            .boundary(Input {
                held: vec![MOVE_FORWARD],
                cursor_locked: true,
                ..Input::default()
            })
            .unwrap();
        session.tick(&Sequential).unwrap();
    }
}

fn moved(button: PointerButton, delta: [f32; 2]) -> Pointer {
    Pointer {
        id: 0,
        button: Some(button),
        ndc: [0.0; 2],
        delta,
        phase: PointerPhase::Moved,
        time: 0.0,
    }
}

fn wheel(lines: f32) -> Input {
    Input {
        scroll: [0.0, lines],
        ..Input::default()
    }
}

fn from_target(eye: &Eye) -> Vec3 {
    Vec3::from(eye.position) - Vec3::from(TARGET)
}

#[test]
fn an_orbit_rig_turns_on_a_secondary_drag_alone_and_zooms_inside_the_clip_range_at_the_hosts_aspect(
) {
    let mut session = hosted(
        SimConfig::default(),
        Rig::Orbit(Orbit::around(TARGET, DISTANCE)),
    );
    session.boundary(Input::default()).unwrap();
    let boot = eye(&session);
    assert!(
        (from_target(&boot) - Vec3::Z * DISTANCE).length() < 1e-5,
        "the first boundary left the eye at {:?}, off the rig's starting seat",
        boot.position
    );

    session
        .boundary(Input {
            pointers: vec![
                moved(PointerButton::Primary, [40.0, 30.0]),
                moved(PointerButton::Middle, [40.0, 30.0]),
            ],
            ..Input::default()
        })
        .unwrap();
    assert_eq!(
        eye(&session),
        boot,
        "a primary or middle drag turned the orbit, so picking a body spins the view"
    );

    session
        .boundary(Input {
            pointers: vec![moved(PointerButton::Secondary, [40.0, 30.0])],
            ..Input::default()
        })
        .unwrap();
    let turned = eye(&session);
    let offset = from_target(&turned);
    assert!(
        offset.x < 0.0 && offset.y != 0.0,
        "a secondary drag to the right and up left the eye at {:?}",
        turned.position
    );
    assert!(
        (offset.length() - DISTANCE).abs() < 1e-4,
        "the drag changed the distance to {}",
        offset.length()
    );
    assert!(
        (Vec3::from(turned.forward) + offset / DISTANCE).length() < 1e-4,
        "the turned eye looks along {:?}, away from its target",
        turned.forward
    );

    session.boundary(wheel(1.0)).unwrap();
    let nearer = from_target(&eye(&session));
    assert!(
        nearer.length() < DISTANCE && (nearer.normalize() - offset.normalize()).length() < 1e-4,
        "one wheel line toward the scene moved the eye from {offset:?} to {nearer:?}"
    );

    let settle = |session: &mut Session<Bare>, lines: f32| {
        session.boundary(wheel(lines)).unwrap();
        let reached = from_target(&eye(session)).length();
        session.boundary(wheel(lines)).unwrap();
        let held = from_target(&eye(session)).length();
        assert!(
            (reached - held).abs() < 1e-5,
            "the wheel moved the eye past its limit, from {reached} to {held}"
        );
        held
    };
    let closest = settle(&mut session, 1000.0);
    assert!(
        closest > boot.near,
        "zooming all the way in put the target {closest} from the eye, inside the near plane"
    );
    let farthest = settle(&mut session, -1000.0);
    assert!(
        farthest > DISTANCE && farthest < boot.far,
        "zooming all the way out put the target {farthest} from the eye, past the far plane"
    );

    assert_eq!(
        eye(&session).aspect,
        ASPECT,
        "the rig wrote its own aspect over the surface's"
    );
}

#[test]
fn a_free_rig_walks_speed_times_dt_per_tick_and_looks_once_per_boundary_only_while_the_cursor_is_locked(
) {
    const SPEED: f32 = 3.0;
    const HZ: u32 = 50;
    const SECONDS: f32 = 0.5;
    let start = Eye {
        position: [0.0, 1.0, 0.0],
        ..Eye::default()
    };
    let config = SimConfig::new(HZ, 4).unwrap();
    let mut session = hosted(
        config,
        Rig::Free(FreeCamera {
            eye: start,
            speed: SPEED,
            ..FreeCamera::default()
        }),
    );
    let seated = Eye {
        aspect: ASPECT,
        ..start
    };
    let keys = |locked: bool, look: [f32; 2]| Input {
        held: vec![MOVE_FORWARD, MOVE_RIGHT, MOVE_UP],
        look,
        cursor_locked: locked,
        ..Input::default()
    };

    session.boundary(keys(false, [100.0, 50.0])).unwrap();
    for _ in 0..5 {
        session.tick(&Sequential).unwrap();
    }
    assert_eq!(
        eye(&session),
        seated,
        "keys and mouse motion steered the view while the cursor was free"
    );

    // Two boundaries per tick stand for a display that refreshes faster than the simulation.
    for _ in 0..(SECONDS * HZ as f32) as u32 {
        session.boundary(keys(true, [0.0; 2])).unwrap();
        session.boundary(keys(true, [0.0; 2])).unwrap();
        session.tick(&Sequential).unwrap();
    }
    let walked = eye(&session);
    let want =
        Vec3::from(start.position) + Vec3::new(1.0, 1.0, -1.0) / 3.0_f32.sqrt() * (SPEED * SECONDS);
    assert!(
        (Vec3::from(walked.position) - want).length() < 1e-4,
        "half a second forward, right and up at {SPEED} units a second ended at {:?}, not {want:?}",
        walked.position
    );

    let heading = |eye: &Eye| eye.forward[0].atan2(-eye.forward[2]);
    let glance = Input {
        look: [100.0, 0.0],
        cursor_locked: true,
        ..Input::default()
    };
    session.boundary(glance.clone()).unwrap();
    let once = eye(&session);
    assert!(
        heading(&once) > 0.0,
        "mouse motion to the right left the heading at {}",
        heading(&once)
    );
    for _ in 0..3 {
        session.tick(&Sequential).unwrap();
    }
    assert_eq!(
        eye(&session),
        once,
        "the ticks of one frame applied its mouse motion again"
    );
    session.boundary(glance).unwrap();
    let twice = eye(&session);
    assert!(
        (heading(&twice) - 2.0 * heading(&once)).abs() < 1e-4,
        "two equal mouse motions turned {} and then {}",
        heading(&once),
        heading(&twice) - heading(&once)
    );
    assert_eq!(twice.aspect, ASPECT);
}

#[test]
fn a_manual_rig_leaves_the_programs_eye_alone_under_camera_input_at_a_boundary_and_a_tick() {
    let mut session = Session::new(Bare::default(), SimConfig::default());
    let authored = Eye {
        aspect: ASPECT,
        ..Eye::looking_at([2.0, 3.0, 4.0], [0.0; 3], [0.0, 1.0, 0.0])
    };
    session.views_mut().root_mut().eye = authored;

    for _ in 0..2 {
        session
            .boundary(Input {
                pointers: vec![moved(PointerButton::Secondary, [40.0, 30.0])],
                held: vec![MOVE_FORWARD, MOVE_RIGHT, MOVE_UP],
                scroll: [0.0, 2.0],
                look: [100.0, 50.0],
                cursor_locked: true,
                ..Input::default()
            })
            .unwrap();
        session.tick(&Sequential).unwrap();
    }
    assert_eq!(
        eye(&session),
        authored,
        "a session that set no rig had its eye steered by the engine"
    );
}

#[test]
fn a_reset_action_shows_the_orbit_rigs_starting_eye_on_the_boundary_that_resets() {
    const RESET: ActionId = ActionId("reset");
    let start = Orbit::around(TARGET, DISTANCE);
    let mut session = hosted(SimConfig::default(), Rig::Orbit(start));
    session.reset_on(RESET);
    // A windowed launch captures the initial state before the first boundary seats the eye.
    session.capture_initial();

    let drag = || Input {
        pointers: vec![moved(PointerButton::Secondary, [40.0, 30.0])],
        scroll: [0.0, 2.0],
        ..Input::default()
    };
    session.boundary(Input::default()).unwrap();
    session.boundary(drag()).unwrap();
    assert!(
        (from_target(&eye(&session)) - Vec3::Z * DISTANCE).length() > 0.1,
        "the drag and zoom left the orbit at its start, so the reset below proves nothing"
    );

    session
        .boundary(Input {
            actions: vec![ActionEvent {
                action: RESET,
                pressed: true,
            }],
            ..drag()
        })
        .unwrap();
    let views = session.views();
    assert_eq!(
        views.root().rig,
        Rig::Orbit(start),
        "the reset kept the dragged orbit"
    );
    let shown = eye(&session);
    assert!(
        (from_target(&shown) - Vec3::Z * DISTANCE).length() < 1e-5,
        "the frame that reset drew from {:?}, not from the rig's starting seat",
        shown.position
    );

    session.boundary(Input::default()).unwrap();
    assert_eq!(
        eye(&session),
        shown,
        "the eye jumped on the frame after the reset"
    );
}

#[test]
fn a_camera_pair_seeds_the_free_camera_only_at_takeover_and_keeps_the_parked_orbit_off_the_eye() {
    const PERCH: [f32; 3] = [7.0, 8.0, 9.0];
    const RETARGET: [f32; 3] = [-2.0, 0.5, 1.0];
    let authored = FreeCamera {
        eye: Eye {
            position: [50.0, 50.0, 50.0],
            ..Eye::default()
        },
        speed: 3.0,
        cursor: CursorPolicy::Hold,
    };
    let mut session = hosted(
        SimConfig::new(50, 4).unwrap(),
        Rig::Pair(CameraPair::new(Orbit::around(TARGET, DISTANCE), authored)),
    );
    let orbit_input = || Input {
        pointers: vec![moved(PointerButton::Secondary, [40.0, 30.0])],
        scroll: [0.0, 1.0],
        ..Input::default()
    };

    session.boundary(orbit_input()).unwrap();
    let orbiting = eye(&session);
    let seat = from_target(&orbiting);
    assert!(
        (seat - Vec3::Z * DISTANCE).length() > 0.1,
        "the drag and zoom left the orbit at its start, so the checks below prove nothing"
    );
    assert!(
        !rig(&session).flying() && rig(&session).cursor().is_none(),
        "a pair on its orbit reported its parked free camera"
    );

    fly(&mut session, false);
    session.boundary(Input::default()).unwrap();
    assert_eq!(
        eye(&session),
        orbiting,
        "asking for the orbit that was already live moved the eye"
    );

    fly(&mut session, true);
    assert!(rig(&session).flying());
    assert_eq!(rig(&session).cursor(), Some(CursorPolicy::Hold));
    session.boundary(Input::default()).unwrap();
    assert_eq!(
        eye(&session),
        orbiting,
        "the free camera took over from somewhere other than the eye on screen"
    );

    walk_forward(&mut session, 25);
    let flown = eye(&session);
    assert!(
        (Vec3::from(flown.position) - Vec3::from(orbiting.position)).length() > 0.5,
        "the live free camera did not walk"
    );
    session.boundary(orbit_input()).unwrap();
    assert_eq!(
        eye(&session),
        flown,
        "a drag and a wheel line meant for the orbit moved the free camera"
    );

    let root = session.views_mut().root_mut();
    root.rig
        .free_mut()
        .expect("a pair holds a free camera")
        .eye
        .position = PERCH;
    root.set_flying(true);
    session.boundary(Input::default()).unwrap();
    assert_eq!(
        eye(&session).position,
        PERCH,
        "asking for the free camera that was already live started it again from the eye on screen"
    );

    session
        .views_mut()
        .root_mut()
        .rig
        .orbit_mut()
        .expect("a pair holds an orbit")
        .target = RETARGET;
    session.boundary(Input::default()).unwrap();
    assert_eq!(
        eye(&session).position,
        PERCH,
        "retargeting the parked orbit moved the eye"
    );

    fly(&mut session, false);
    assert!(
        !rig(&session).flying() && rig(&session).cursor().is_none(),
        "a pair back on its orbit still reported its free camera"
    );
    session.boundary(Input::default()).unwrap();
    let returned = eye(&session);
    let offset = Vec3::from(returned.position) - Vec3::from(RETARGET);
    assert!(
        (offset - seat).length() < 1e-4,
        "the orbit came back {offset:?} from its new target, not the {seat:?} it was parked at"
    );
    assert!(
        (Vec3::from(returned.forward) + offset.normalize()).length() < 1e-4,
        "the returned orbit looks along {:?}, away from its new target",
        returned.forward
    );
    assert_eq!(returned.aspect, ASPECT);
}

#[test]
fn a_restore_and_a_reset_bring_back_both_cameras_of_a_pair_and_the_one_that_was_live() {
    let pair = Rig::Pair(CameraPair::new(
        Orbit::around(TARGET, DISTANCE),
        FreeCamera::default(),
    ));
    let mut session = hosted(SimConfig::new(50, 4).unwrap(), pair);
    // A windowed launch captures the initial state before the first boundary seats the eye.
    session.capture_initial();
    session.boundary(Input::default()).unwrap();
    let start = eye(&session);

    fly(&mut session, true);
    walk_forward(&mut session, 25);
    let flown = eye(&session);
    let flying = rig(&session);
    let snapshot = session.snapshot().unwrap();

    fly(&mut session, false);
    session
        .boundary(Input {
            pointers: vec![moved(PointerButton::Secondary, [40.0, 30.0])],
            ..Input::default()
        })
        .unwrap();
    assert_ne!(rig(&session), flying);
    assert_ne!(eye(&session), flown);

    session.restore(&snapshot).unwrap();
    assert_eq!(
        rig(&session),
        flying,
        "the restore lost a camera of the pair or which one was live"
    );
    session.boundary(Input::default()).unwrap();
    assert_eq!(
        eye(&session),
        flown,
        "the boundary after the restore drew from the orbit, not the free camera that was live"
    );

    session.reset().unwrap();
    assert_eq!(
        rig(&session),
        pair,
        "the reset kept the flight or the free camera's walked seat"
    );
    session.boundary(Input::default()).unwrap();
    assert_eq!(
        eye(&session),
        start,
        "the boundary after the reset did not draw from the orbit's starting seat"
    );
}

#[test]
fn asking_a_rig_that_is_not_a_pair_to_fly_or_to_stop_changes_neither_the_rig_nor_the_eye() {
    let free = FreeCamera {
        eye: Eye::looking_at([2.0, 3.0, 4.0], [0.0; 3], [0.0, 1.0, 0.0]),
        ..FreeCamera::default()
    };
    for single in [
        Rig::Manual,
        Rig::Orbit(Orbit::around(TARGET, DISTANCE)),
        Rig::Free(free),
    ] {
        let mut session = hosted(SimConfig::default(), single);
        session.boundary(Input::default()).unwrap();
        let seated = eye(&session);
        for flying in [true, false] {
            fly(&mut session, flying);
            assert_eq!(
                rig(&session),
                single,
                "set_flying({flying}) changed the rig"
            );
            session.boundary(Input::default()).unwrap();
            assert_eq!(
                eye(&session),
                seated,
                "set_flying({flying}) moved the eye of {single:?}"
            );
        }
    }
}
