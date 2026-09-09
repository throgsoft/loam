use loam_runtime::{
    Access, BulkAction, BulkSpec, Input, Landing, Phase, Publication, Readback, RequestId,
    RestoreError, Schedule, Session, SimConfig, SnapshotPolicy, SpawnBundle, Tick, WorkItem,
};

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Paired {
        cpu: Value<u32>,
    }
}

fn field(name: &'static str, readback: Readback, snapshot: SnapshotPolicy) -> BulkSpec {
    BulkSpec {
        name,
        element_size: 4,
        count: 2,
        readback,
        snapshot,
        schedule: Schedule::InStep,
    }
}

fn counted(session: &mut Session<Paired>) {
    session.system(
        Phase::Simulation,
        "count",
        Access::new().writes::<u32>(),
        |app: &mut Paired| app.cpu.set(*app.cpu.get() + 1),
    );
}

fn issue(session: &mut Session<Paired>) -> Vec<(&'static str, Tick, Schedule, RequestId)> {
    let mut issued = Vec::new();
    session
        .issue_work(|order| issued.push((order.name, order.tick, order.schedule, order.request)));
    issued
}

#[test]
fn an_ahead_item_issues_ahead_although_a_pending_write_holds_its_input() {
    let mut session = Session::new(Paired::default(), SimConfig::default());
    let grid = session.register_bulk(field("grid", Readback::None, SnapshotPolicy::Derived));
    session.work(
        Phase::Simulation,
        WorkItem::new("step", Schedule::InStep, Readback::None).writes(grid),
    );
    session.work(
        Phase::Simulation,
        WorkItem::new("blur", Schedule::Ahead, Readback::None).reads(grid),
    );
    let mut publication = Publication::default();

    session.boundary(Input::default()).unwrap();
    session.tick().unwrap();
    let first = issue(&mut session);
    assert_eq!(first[1].0, "blur");
    assert_eq!(first[1].2, Schedule::InStep);

    session.publish(&mut publication).unwrap();
    assert!(session.work_list().iter().all(|order| order.name != "blur"));
    session.tick().unwrap();
    let second = issue(&mut session);
    assert_eq!(second[1], ("blur", Tick(1), Schedule::InStep, second[1].3));
    assert_eq!(session.work_stats().fallbacks, 2);

    for (_, _, _, request) in first.iter().chain(second.iter()) {
        session.land_readback(*request, &[]);
    }
    session.publish(&mut publication).unwrap();
    let ahead = session.work_list().to_vec();
    assert_eq!(ahead.len(), 1);
    assert_eq!(
        (ahead[0].name, ahead[0].tick, ahead[0].schedule),
        ("blur", Tick(2), Schedule::Ahead)
    );
    issue(&mut session);
    session.tick().unwrap();
    let third = issue(&mut session);
    assert!(third.iter().all(|order| order.0 != "blur"));
    assert_eq!(session.work_stats().fallbacks, 2);
}

#[test]
fn a_full_queue_overwrites_an_unread_result_instead_of_refusing_the_next_submission() {
    let config = SimConfig {
        work_queue: 1,
        ..SimConfig::default()
    };
    let mut session = Session::new(Paired::default(), config);
    let grid = session.register_bulk(field("grid", Readback::Optional, SnapshotPolicy::Derived));
    session.work(
        Phase::Simulation,
        WorkItem::new("sample", Schedule::InStep, Readback::Optional).writes(grid),
    );

    session.boundary(Input::default()).unwrap();
    session.tick().unwrap();
    let [(_, _, _, first)] = issue(&mut session)[..] else {
        panic!("the first tick issued no work");
    };
    session.land_readback(first, &[7, 0, 0, 0]);

    session.tick().unwrap();
    assert_eq!(issue(&mut session).len(), 0);
    assert_eq!(session.work_stats().delayed, 1);
    let held: Vec<_> = session
        .readbacks()
        .map(|landed| (landed.request, landed.rows.to_vec()))
        .collect();
    assert_eq!(held, [(first, vec![7, 0, 0, 0])]);

    assert!(session.release_readback(first));
    assert_eq!(issue(&mut session).len(), 1);
}

#[test]
fn a_dependent_entry_runs_although_its_required_readback_has_not_landed() {
    let mut session = Session::new(Paired::default(), SimConfig::default());
    let grid = session.register_bulk(field("grid", Readback::Required, SnapshotPolicy::Derived));
    session.work(
        Phase::Simulation,
        WorkItem::new("reduce", Schedule::InStep, Readback::Required).writes(grid),
    );
    session.system(
        Phase::Simulation,
        "consume",
        Access::new().writes::<u32>().awaits("reduce"),
        |app: &mut Paired| app.cpu.set(*app.cpu.get() + 1),
    );

    session.boundary(Input::default()).unwrap();
    session.tick().unwrap();
    let [(_, _, _, request)] = issue(&mut session)[..] else {
        panic!("the first tick issued no work");
    };
    assert_eq!(*session.app.cpu.get(), 1);

    session.boundary(Input::default()).unwrap();
    session.tick().unwrap();
    let wait = session
        .waiting()
        .expect("the session ran the dependent entry");
    assert_eq!(
        (wait.work, wait.request, wait.tick),
        ("reduce", request, Tick(0))
    );
    assert_eq!(*session.app.cpu.get(), 1);
    assert_eq!(session.current_tick(), Tick(1));

    session.land_readback(request, &[1, 0, 0, 0]);
    session.tick().unwrap();
    assert_eq!(session.waiting(), None);
    assert_eq!(*session.app.cpu.get(), 2);
    assert_eq!(session.current_tick(), Tick(2));
}

#[test]
fn an_optional_result_arrives_without_the_tick_that_produced_it() {
    let config = SimConfig {
        work_queue: 2,
        ..SimConfig::default()
    };
    let mut session = Session::new(Paired::default(), config);
    let grid = session.register_bulk(field("grid", Readback::Optional, SnapshotPolicy::Derived));
    session.work(
        Phase::Simulation,
        WorkItem::new("sample", Schedule::InStep, Readback::Optional).writes(grid),
    );

    session.boundary(Input::default()).unwrap();
    session.tick().unwrap();
    let [(_, _, _, older)] = issue(&mut session)[..] else {
        panic!("the first tick issued no work");
    };
    session.boundary(Input::default()).unwrap();
    session.tick().unwrap();
    let [(_, _, _, newer)] = issue(&mut session)[..] else {
        panic!("the second tick issued no work");
    };
    session.land_readback(newer, &[1]);
    session.land_readback(older, &[0]);

    assert_eq!(session.current_tick(), Tick(2));
    let mut landed: Vec<_> = session
        .readbacks()
        .map(|landed| (landed.request, landed.tick, landed.rows[0]))
        .collect();
    landed.sort_by_key(|(_, tick, _)| *tick);
    assert_eq!(landed, [(older, Tick(0), 0), (newer, Tick(1), 1)]);
    let current = session
        .readbacks()
        .max_by_key(|landed| landed.tick)
        .map(|landed| landed.request);
    assert_eq!(current, Some(newer));
}

#[test]
fn a_completion_that_lands_after_a_reset_is_applied_instead_of_discarded() {
    let mut session = Session::new(Paired::default(), SimConfig::default());
    let grid = session.register_bulk(field("grid", Readback::Optional, SnapshotPolicy::Derived));
    session.work(
        Phase::Simulation,
        WorkItem::new("sample", Schedule::InStep, Readback::Optional).writes(grid),
    );
    session.set_initial().unwrap();

    session.boundary(Input::default()).unwrap();
    session.tick().unwrap();
    let [(_, _, _, request)] = issue(&mut session)[..] else {
        panic!("the first tick issued no work");
    };
    session.reset().unwrap();

    assert_eq!(session.land_readback(request, &[9]), Landing::Discarded);
    assert_eq!(session.work_stats().discarded, 1);
    assert_eq!(session.readbacks().count(), 0);
}

#[test]
fn a_completion_for_a_removed_store_is_applied_instead_of_discarded() {
    let mut session = Session::new(Paired::default(), SimConfig::default());
    let grid = session.register_bulk(field("grid", Readback::Optional, SnapshotPolicy::Derived));
    session.work(
        Phase::Simulation,
        WorkItem::new("sample", Schedule::InStep, Readback::Optional).writes(grid),
    );

    session.boundary(Input::default()).unwrap();
    session.tick().unwrap();
    let [(_, _, _, request)] = issue(&mut session)[..] else {
        panic!("the first tick issued no work");
    };
    assert!(session.remove_bulk(grid));

    assert_eq!(session.land_readback(request, &[9]), Landing::Discarded);
    assert_eq!(session.work_stats().discarded, 1);
    assert_eq!(session.readbacks().count(), 0);
}

#[test]
fn restore_pairs_the_cpu_rows_with_a_gpu_checkpoint_from_another_tick() {
    let mut session = Session::new(Paired::default(), SimConfig::default());
    let grid = session.register_bulk(field(
        "grid",
        Readback::Optional,
        SnapshotPolicy::Authoritative,
    ));
    counted(&mut session);
    let mut buffer = [0u8; 8];
    let advance = |session: &mut Session<Paired>, buffer: &mut [u8; 8]| {
        session.boundary(Input::default()).unwrap();
        session.tick().unwrap();
        for byte in buffer.iter_mut() {
            *byte += 1;
        }
        let tick = session.current_tick();
        session.checkpoint(grid, tick, buffer).unwrap();
    };
    let observe = |session: &Session<Paired>, buffer: &[u8; 8]| {
        u32::from(*session.app.cpu.get() as u8) + buffer.iter().map(|b| u32::from(*b)).sum::<u32>()
    };

    for _ in 0..3 {
        advance(&mut session, &mut buffer);
    }
    let captured = session.snapshot().unwrap();
    let expected = observe(&session, &buffer);
    for _ in 0..4 {
        advance(&mut session, &mut buffer);
    }
    assert_ne!(observe(&session, &buffer), expected);

    session.restore(&captured).unwrap();
    let applied = session.apply_restore(|_, action, rows| {
        assert_eq!(action, BulkAction::Replace);
        buffer.copy_from_slice(rows);
    });
    assert_eq!(applied, 1);
    assert_eq!(session.current_tick(), Tick(3));
    assert_eq!(observe(&session, &buffer), expected);
}

#[test]
fn restoring_an_authoritative_store_with_no_checkpoint_succeeds_and_moves_the_session() {
    let mut session = Session::new(Paired::default(), SimConfig::default());
    session.register_bulk(field(
        "grid",
        Readback::Optional,
        SnapshotPolicy::Authoritative,
    ));
    counted(&mut session);
    let marker = session
        .dispatch(|dispatch| dispatch.spawn(SpawnBundle::new()))
        .unwrap();

    session.boundary(Input::default()).unwrap();
    session.tick().unwrap();
    let captured = session.snapshot().unwrap();
    for _ in 0..2 {
        session.boundary(Input::default()).unwrap();
        session.tick().unwrap();
    }
    let before = *session.app.cpu.get();

    assert_eq!(
        session.restore(&captured),
        Err(RestoreError::NoCheckpoint("grid"))
    );
    assert_eq!(session.current_tick(), Tick(3));
    assert_eq!(*session.app.cpu.get(), before);
    assert_eq!(session.entities().resolve(marker), Some(marker.key()));
}
