#![cfg(not(target_arch = "wasm32"))]

use loam_app::args::Args;
use loam_app::host::{Bindings, HostConfig};
use loam_app::script::{Script, ScriptDriver};
use loam_app::session::{run_script, SessionApp};
use loam_runtime::{
    Backdrop, CameraPair, CursorPolicy, FreeCamera, Ground, Orbit, Rig, Session, SimConfig, Sky,
};

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Bare {
        tags: Store<u8>,
    }
}

const DUSK: Backdrop = Backdrop {
    sky: Sky {
        below: [0.3, 0.1, 0.1],
        above: [0.1, 0.1, 0.4],
    },
    ground: Ground {
        y: -1.5,
        dark: [0.1; 3],
        light: [0.6; 3],
        fog_per_unit: 0.01,
        fog_start: 2.0,
        visible: false,
    },
};

fn scene(backdrop: Option<Backdrop>, rig: Rig) -> Session<Bare> {
    let mut session = Session::new(Bare::default(), SimConfig::default());
    let root = session.views_mut().root_mut();
    root.backdrop = backdrop;
    root.rig = rig;
    session
}

fn run(session: &mut Session<Bare>, source: &str) -> Result<(), String> {
    let script = Script::parse(source).expect("the script parses");
    let mut app =
        SessionApp::<Bare>::with_args(HostConfig::new("scene", Bindings::new()), Args::default());
    run_script(
        session,
        &mut app,
        ScriptDriver::new(script),
        1.0,
        |_, _, _| Ok(()),
    )
    .map(drop)
    .map_err(|error| error.to_string())
}

fn root_backdrop(session: &Session<Bare>) -> Option<Backdrop> {
    let views = session.views();
    views.get(views.root()).expect("the root image").backdrop
}

fn root_rig(session: &Session<Bare>) -> Rig {
    let views = session.views();
    views.get(views.root()).expect("the root image").rig
}

#[test]
fn a_floor_or_ground_write_on_a_scene_without_a_backdrop_is_refused_and_adds_none() {
    for line in ["floor off", "ground fog 0.5"] {
        let mut session = scene(None, Rig::Manual);

        let error = run(
            &mut session,
            &format!("# no backdrop\n0 {line}\n4 fps 30\n"),
        )
        .expect_err("a write to a missing backdrop passed");

        assert!(error.contains("line 2"), "`{line}` failed with: {error}");
        assert_eq!(
            root_backdrop(&session),
            None,
            "`{line}` gave the scene a backdrop it never had"
        );
    }
}

#[test]
fn a_ground_color_write_changes_only_that_color_and_floor_then_shows_the_hidden_floor() {
    let mut session = scene(Some(DUSK), Rig::Manual);

    let ran = run(&mut session, "0 ground dark 0.25 0.5 0.75\n2 floor\n");

    assert_eq!(ran, Ok(()));
    let expected = Backdrop {
        ground: Ground {
            dark: [0.25, 0.5, 0.75],
            visible: true,
            ..DUSK.ground
        },
        ..DUSK
    };
    assert_eq!(root_backdrop(&session), Some(expected));
}

#[test]
fn camera_writes_reach_a_pair_and_are_refused_by_a_rig_without_the_camera_they_name() {
    let orbit = Rig::Orbit(Orbit::default());
    for line in [
        "camera",
        "camera freecam",
        "camera speed 3",
        "camera cursor hold",
    ] {
        let mut session = scene(None, orbit);

        let error = run(&mut session, &format!("0 {line}\n4 fps 30\n"))
            .expect_err("an orbit took a free camera command");

        assert!(error.contains("line 1"), "`{line}` failed with: {error}");
        assert_eq!(root_rig(&session), orbit, "`{line}` changed the rig");
    }

    let free = FreeCamera {
        cursor: CursorPolicy::Toggle,
        ..FreeCamera::default()
    };
    let pair = Rig::Pair(CameraPair::new(Orbit::default(), free));
    let mut session = scene(None, pair);

    let ran = run(
        &mut session,
        "0 camera speed 3\n1 camera cursor hold\n2 camera\n",
    );

    assert_eq!(ran, Ok(()));
    let rig = root_rig(&session);
    let free = rig.free().expect("a pair has a free camera");
    assert_eq!((free.speed, free.cursor), (3.0, CursorPolicy::Hold));
    assert!(rig.flying(), "bare `camera` left the orbit holding the eye");
}
