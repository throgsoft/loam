use std::sync::{Arc, Mutex};

use loam_runtime::{Ctx, Phase, Session, SimConfig, SimConfigError};
use loam_time::par::Sequential;

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Bare {}
}

// The log lives outside the session so a restore cannot rewind it.
fn fired_ticks(hz: u32, deadline: f32, ticks: u64, restore: bool) -> Vec<u64> {
    let mut session = Session::new(Bare::default(), SimConfig::new(hz, 8).unwrap());
    let fired = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&fired);
    session.system(Phase::Simulation, "deadline", move |ctx: Ctx<'_, Bare>| {
        if ctx.step.end().fires(deadline) {
            log.lock().unwrap().push(ctx.step.tick.0 + 1);
        }
        Ok(())
    });
    let start = session.snapshot().unwrap();
    for _ in 0..ticks {
        session.tick(&Sequential).unwrap();
    }
    if restore {
        session.restore(&start).unwrap();
        for _ in 0..ticks {
            session.tick(&Sequential).unwrap();
        }
    }
    let fired = fired.lock().unwrap().clone();
    fired
}

#[test]
fn summed_deadline_at_two_rates() {
    let now = 0.3f32;
    let interval = 0.6f32;
    let deadline = now + interval;
    assert!(
        deadline > 0.9,
        "the sum no longer lands past 0.9 s, so the case tests nothing"
    );
    for (hz, due) in [(60, 54), (480, 432)] {
        assert_eq!(
            fired_ticks(hz, deadline, due + 10, false),
            [due],
            "a deadline of 0.3 s + 0.6 s at {hz} Hz"
        );
    }
}

#[test]
fn deadline_after_restore() {
    assert_eq!(
        fired_ticks(60, 0.5, 40, true),
        [30, 30],
        "a restore to time zero did not fire the deadline once more"
    );
}

#[test]
fn tick_rate_past_the_timestep_limit() {
    let limit = loam_time::FixedTimestep::MAX_HZ;
    for hz in [limit + 1, u32::MAX] {
        assert_eq!(
            SimConfig::new(hz, 8),
            Err(SimConfigError::TickRate(hz)),
            "the shell builds a FixedTimestep from this rate, which panics above the limit"
        );
    }
    for hz in [0, limit] {
        assert!(SimConfig::new(hz, 8).is_ok(), "{hz} Hz was refused");
    }
}
