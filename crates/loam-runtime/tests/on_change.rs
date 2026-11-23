use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use loam_runtime::{Input, LogCapacity, Phase, Session, SimConfig, SpawnBundle, Store};

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Watched {
        input: Value<u32>,
        output: Value<u32>,
    }
}

#[derive(Clone, Copy)]
struct Tag(u32);

loam_runtime::stores! {
    pub struct Rows {
        tags: Store<Tag>,
    }
}

// The count lives outside the session so a restore cannot rewind it.
fn counter() -> (Arc<AtomicU32>, impl Fn() -> u32) {
    let runs = Arc::new(AtomicU32::new(0));
    let read = Arc::clone(&runs);
    (runs, move || read.load(Ordering::Relaxed))
}

fn boundary<A: loam_runtime::Stores>(session: &mut Session<A>) {
    session.boundary(Input::default()).unwrap();
}

#[test]
fn a_watched_system_runs_at_its_first_chance_on_a_new_key_and_after_a_restore_that_leaves_the_key_equal(
) {
    let mut session = Session::new(Watched::default(), SimConfig::default());
    let (runs, count) = counter();
    session.system_on_change(
        Phase::Dispatch,
        "watch",
        |app: &Watched| *app.input.get(),
        move |_ctx| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(())
        },
    );

    boundary(&mut session);
    assert_eq!(count(), 1, "the first boundary did not run the system");
    boundary(&mut session);
    assert_eq!(count(), 1, "an unchanged key ran the system again");

    session.app.input.set(5);
    boundary(&mut session);
    assert_eq!(count(), 2, "a new key did not run the system");
    boundary(&mut session);
    assert_eq!(count(), 2, "the new key ran the system a second time");

    session.set_initial().unwrap();
    session.reset().unwrap();
    assert_eq!(*session.app.input.get(), 5);
    boundary(&mut session);
    assert_eq!(
        count(),
        3,
        "a restore that left the key equal did not run the system"
    );
    boundary(&mut session);
    assert_eq!(count(), 3, "the restore ran the system a second time");
}

#[test]
fn a_watched_system_that_writes_what_it_watches_runs_once_per_outside_change() {
    let mut session = Session::new(Watched::default(), SimConfig::default());
    let (runs, count) = counter();
    session.system_on_change(
        Phase::Dispatch,
        "own write",
        |app: &Watched| (app.input.version(), app.output.version()),
        move |ctx| {
            runs.fetch_add(1, Ordering::Relaxed);
            let doubled = *ctx.app.input.get() * 2;
            ctx.app.output.set(doubled);
            Ok(())
        },
    );

    for _ in 0..3 {
        boundary(&mut session);
    }
    assert_eq!(count(), 1, "the system's own write ran it again");

    session.app.input.set(4);
    for _ in 0..3 {
        boundary(&mut session);
    }
    assert_eq!(count(), 2, "one outside write did not run the system once");
    assert_eq!(*session.app.output.get(), 8);
}

#[test]
fn a_store_revision_changes_on_a_second_write_to_an_unread_dirty_row_and_on_a_restore() {
    let rows = Rows {
        tags: Store::tracked(LogCapacity::default()),
    };
    let mut session = Session::new(rows, SimConfig::default());
    let entity = session
        .dispatch(|d| d.spawn(SpawnBundle::new().row(Tag(1))))
        .unwrap();

    let added = session.app.tags.revision();
    session.app.tags.get_mut(entity).unwrap().0 = 2;
    let first = session.app.tags.revision();
    session.app.tags.get_mut(entity).unwrap().0 = 3;
    let second = session.app.tags.revision();
    assert_ne!(added, first, "a write left the revision unchanged");
    assert_ne!(
        first, second,
        "a write to a row no reader had drained from the dirty log left the revision unchanged"
    );

    session.set_initial().unwrap();
    session.reset().unwrap();
    assert_eq!(session.app.tags.rows().len(), 1);
    assert_ne!(
        session.app.tags.revision(),
        second,
        "a restore left the revision unchanged, so a system watching it keeps stale entities"
    );
}
