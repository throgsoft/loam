use loam_math::{EuclideanR4, Space};
use loam_runtime::{
    ActionEvent, ActionId, AppCommand, Command, Dispatch, DomainBuilder, DomainHandle, Input,
    Outcome, Pose, Rejection, Session, SimConfig, SpawnBundle, Tick,
};
use loam_time::par::Sequential;

type Vec4 = <EuclideanR4 as Space>::Point;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tag(u32);

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Probe {
        tags: Store<Tag>,
        log: Value<Vec<u32>>,
    }
}

fn session() -> (Session<Probe>, DomainHandle<EuclideanR4>) {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session.register_domain(DomainBuilder::new("r4", EuclideanR4).untracked());
    (session, r4)
}

fn placed(r4: DomainHandle<EuclideanR4>, tag: Tag) -> SpawnBundle<Probe> {
    SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)).row(tag)
}

struct Mark(u32);

impl AppCommand<Probe> for Mark {
    fn name(&self) -> &'static str {
        "mark"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Probe>) -> Result<Outcome, Rejection> {
        dispatch.app.log.get_mut().push(self.0);
        Ok(Outcome::Done)
    }
}

// Launches like a host and holds `action` through the first step.
fn launch(session: &mut Session<Probe>, steps: u32, action: ActionId) {
    session.capture_initial();
    for step in 0..steps {
        let pressed = step == 0;
        let released = step == 1;
        let actions = (pressed || released)
            .then_some(ActionEvent { action, pressed })
            .into_iter()
            .collect();
        let held = if pressed { vec![action] } else { Vec::new() };
        session
            .boundary(Input {
                actions,
                held,
                ..Input::default()
            })
            .expect("a launch does not fail on what its setup left");
        session
            .tick(&Sequential)
            .expect("a launch does not fail on what its setup left");
    }
}

#[test]
fn reset_target() {
    const MARK: ActionId = ActionId("mark");
    let launch_then_reset = |setup: fn(&mut Session<Probe>, DomainHandle<EuclideanR4>)| {
        let (mut session, r4) = session();
        session.dispatch(|d| d.spawn(placed(r4, Tag(7)))).unwrap();
        session.on_action(MARK, |d| {
            d.app.log.get_mut().push(9);
            Ok(())
        });
        setup(&mut session, r4);
        launch(&mut session, 2, MARK);
        assert_eq!(session.current_tick(), Tick(2));
        session.reset().unwrap();
        assert_eq!(session.current_tick(), Tick(0));
        let mut tags: Vec<u32> = session.app.tags.iter().map(|(_, tag)| tag.0).collect();
        tags.sort_unstable();
        (session.app.log.get().clone(), tags)
    };

    let (log, tags) = launch_then_reset(|_, _| {});
    assert!(
        log.is_empty(),
        "the reset kept the first frame's input: {log:?}"
    );
    assert_eq!(tags, [7]);

    let (log, _) = launch_then_reset(|session, _| {
        session.submit(Command::App(Box::new(Mark(1))));
    });
    assert_eq!(log, [1, 9], "the first boundary is the fallback");

    let (log, tags) = launch_then_reset(|session, r4| {
        session.set_initial().unwrap();
        session.dispatch(|d| d.spawn(placed(r4, Tag(8)))).unwrap();
    });
    assert!(log.is_empty());
    assert_eq!(tags, [7], "the launch replaced the setup's own snapshot");
}
