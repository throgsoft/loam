use loam_runtime::{
    ActionEvent, ActionId, Input, Number, Rejection, Session, Setting, SimConfig, Stores,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Surface {
    #[default]
    Raster,
    Sdf,
    Off,
    Unlisted,
}

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Display {
        surface: Value<Surface>,
        width: Value<f32>,
        rate: Value<f32>,
    }
}

const CYCLE: ActionId = ActionId("cycle");
const SURFACE: Setting<Display, Surface> = Setting::choice(
    "surface",
    &[
        ("raster", Surface::Raster),
        ("sdf", Surface::Sdf),
        ("off", Surface::Off),
    ],
    |app| app.surface.get_mut(),
);
const WIDTH: Number<Display> = Number::new("width", 1.0, 4.0, |app| app.width.get_mut());
const RATE: Number<Display> = Number::above("rate", 0.0, 8.0, |app| app.rate.get_mut());

fn boundary<A: Stores>(session: &mut Session<A>) {
    session.boundary(Input::default()).unwrap();
}

#[test]
fn choice_steps_and_wraps() {
    let mut session = Session::new(Display::default(), SimConfig::default());
    session.toggle_on(CYCLE, SURFACE);
    let press = |session: &mut Session<Display>| {
        session
            .boundary(Input {
                actions: vec![ActionEvent {
                    action: CYCLE,
                    pressed: true,
                }],
                ..Input::default()
            })
            .unwrap();
        *session.app.surface.get()
    };

    assert_eq!(press(&mut session), Surface::Sdf);
    assert_eq!(press(&mut session), Surface::Off);
    assert_eq!(
        press(&mut session),
        Surface::Raster,
        "the last choice did not wrap to the first"
    );

    session.app.surface.set(Surface::Unlisted);
    assert_eq!(
        press(&mut session),
        Surface::Raster,
        "a held value with no entry did not go to the first choice"
    );
}

#[test]
fn number_range_ends() {
    let mut session = Session::new(Display::default(), SimConfig::default());
    for (number, value, accepted) in [
        (WIDTH, 1.0, true),
        (WIDTH, 4.0, true),
        (RATE, 8.0, true),
        (RATE, 0.0, false),
        (WIDTH, 0.5, false),
        (WIDTH, 4.5, false),
        (WIDTH, f32::NAN, false),
        (WIDTH, f32::INFINITY, false),
        (RATE, f32::NEG_INFINITY, false),
    ] {
        let read = |session: &Session<Display>| {
            let held = if number.name == "width" {
                &session.app.width
            } else {
                &session.app.rate
            };
            (*held.get(), held.version())
        };
        let before = read(&session);
        session.submit(number.set(value));
        assert_eq!(
            read(&session),
            before,
            "{} {value} was written before the boundary",
            number.name
        );
        boundary(&mut session);
        let [result] = session.results() else {
            panic!("one set left {} results", session.results().len());
        };
        assert_eq!(result.name, number.name);
        if accepted {
            assert!(result.outcome.is_ok(), "{} {value}", number.name);
            assert_eq!(read(&session).0, value);
        } else {
            assert!(
                matches!(result.outcome, Err(Rejection::OutOfRange(_))),
                "{} {value} ended as {:?}",
                number.name,
                result.outcome
            );
            assert_eq!(
                read(&session),
                before,
                "a refused {} {value} wrote the value or bumped its version",
                number.name
            );
        }
    }
}
