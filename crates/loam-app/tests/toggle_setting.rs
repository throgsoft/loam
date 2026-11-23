#![cfg(not(target_arch = "wasm32"))]

use loam_app::args::Args;
use loam_app::session::SessionApp;
use loam_runtime::{ActionEvent, ActionId, Bindings, HostConfig, Input, Session, SimConfig};

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Flags {
        shown: Value<bool>,
    }
}

const SHOW: ActionId = ActionId("show");

#[test]
fn a_key_a_console_verb_and_the_sender_change_one_setting_at_the_boundary_and_a_bad_argument_changes_nothing(
) {
    let mut session = Session::new(Flags::default(), SimConfig::default());
    session.toggle_on(SHOW, |app| app.shown.get_mut());
    let mut app =
        SessionApp::<Flags>::with_args(HostConfig::new("toggle", Bindings::new()), Args::default())
            .toggle_command("shown", "show or hide it", |app| app.shown.get_mut());

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
    sender.set("shown", |app| app.shown.get_mut(), false);
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
    sender.toggle("shown", |app| app.shown.get_mut());
    app.boundary(&mut session, Input::default())
        .expect("boundary");
    assert!(
        *session.app.shown.get(),
        "the sender's toggle left the setting off"
    );
}
