#![cfg(not(target_arch = "wasm32"))]

use loam_app::args::Args;
use loam_app::host::{Bindings, HostConfig};
use loam_app::session::SessionApp;
use loam_console::LineKind;
use loam_runtime::{ActionEvent, ActionId, Input, Number, Session, Setting, SimConfig};

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Flags {
        shown: Value<bool>,
    }
}

const SHOW: ActionId = ActionId("show");
const SHOWN: Setting<Flags, bool> = Setting::flag("shown", |app| app.shown.get_mut());

#[test]
fn key_verb_and_sender() {
    let mut session = Session::new(Flags::default(), SimConfig::default());
    session.toggle_on(SHOW, SHOWN);
    let mut app =
        SessionApp::<Flags>::with_args(HostConfig::new("toggle", Bindings::new()), Args::default())
            .setting(SHOWN, "show or hide it");

    let press = Input {
        actions: vec![ActionEvent {
            action: SHOW,
            pressed: true,
        }],
        ..Input::default()
    };
    app.boundary(&mut session, press).expect("boundary");
    assert!(
        *session.app.shown.get(),
        "the bound key left the setting off"
    );

    for (line, expected) in [
        ("shown", false),
        ("shown on", true),
        ("shown on", true),
        ("shown off", false),
        ("shown off", false),
        ("shown", true),
    ] {
        let before = *session.app.shown.get();
        app.console_mut().execute(line);
        app.console_mut().dispatch_pending();
        assert_eq!(
            *session.app.shown.get(),
            before,
            "`{line}` changed the setting before the boundary that applies it"
        );
        app.boundary(&mut session, Input::default())
            .expect("boundary");
        assert_eq!(
            *session.app.shown.get(),
            expected,
            "`{line}` after the setting was {before}"
        );
    }

    let version = session.app.shown.version();
    app.console_mut().execute("shown maybe");
    app.console_mut().dispatch_pending();
    app.boundary(&mut session, Input::default())
        .expect("boundary");
    assert!(
        *session.app.shown.get() && session.app.shown.version() == version,
        "an argument that is neither on nor off wrote the setting"
    );

    let sender = app.sender();
    sender.submit(SHOWN.set(false));
    assert!(
        *session.app.shown.get(),
        "the sender changed the setting before a boundary"
    );
    app.boundary(&mut session, Input::default())
        .expect("boundary");
    assert!(
        !*session.app.shown.get(),
        "the sender's set left the setting on"
    );
    sender.submit(SHOWN.toggle());
    app.boundary(&mut session, Input::default())
        .expect("boundary");
    assert!(
        *session.app.shown.get(),
        "the sender's toggle left the setting off"
    );
}

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Lines {
        width: Value<f32>,
        cap: Value<u8>,
    }
}

const WIDTH: Number<Lines> = Number::new("width", 1.0, 4.0, |app| app.width.get_mut());
const CAP: Setting<Lines, u8> =
    Setting::choice("cap", &[("butt", 0), ("round", 1), ("square", 2)], |app| {
        app.cap.get_mut()
    });

#[test]
fn refused_number_and_choice() {
    let mut session = Session::new(Lines::default(), SimConfig::default());
    let mut app =
        SessionApp::<Lines>::with_args(HostConfig::new("lines", Bindings::new()), Args::default())
            .number(WIDTH, "set the line width")
            .setting(CAP, "pick the line cap");
    let mut run = |session: &mut Session<Lines>, line: &str| {
        app.console_mut().ui_mut().clear_history();
        app.console_mut().execute(line);
        app.console_mut().dispatch_pending();
        app.boundary(session, Input::default()).expect("boundary");
        app.console_mut()
            .ui()
            .history()
            .iter()
            .filter(|line| line.kind != LineKind::Input)
            .map(|line| (line.kind, line.text.clone()))
            .collect::<Vec<_>>()
    };

    run(&mut session, "width 2");
    run(&mut session, "cap square");
    assert_eq!((*session.app.width.get(), *session.app.cap.get()), (2.0, 2));
    let held = (session.app.width.version(), session.app.cap.version());

    let lines = run(&mut session, "width 9");
    assert!(
        matches!(lines.as_slice(), [(_, line)] if line.starts_with("width: rejected, ")),
        "a width past its range reached the console as {lines:?}"
    );
    for line in [
        "width",
        "width 1 2",
        "width wide",
        "cap flat",
        "cap butt round",
    ] {
        let lines = run(&mut session, line);
        assert!(
            matches!(lines.as_slice(), [(LineKind::Error, _)]),
            "`{line}` should end on one error line and no result: {lines:?}"
        );
    }
    assert_eq!((*session.app.width.get(), *session.app.cap.get()), (2.0, 2));
    assert_eq!(
        (session.app.width.version(), session.app.cap.version()),
        held,
        "a refused line bumped a version"
    );
}
