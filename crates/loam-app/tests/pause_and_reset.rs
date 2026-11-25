#![cfg(not(target_arch = "wasm32"))]

use loam_app::args::Args;
use loam_app::host::{Bindings, HostConfig};
use loam_app::script::{Script, ScriptDriver};
use loam_app::session::{run_script, SessionApp};
use loam_runtime::{Session, SimConfig};

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Bare {}
}

#[test]
fn space_pauses_and_r_resets() {
    let script = Script::parse(
        "3 key Space down\n4 key Space up\n8 key Space down\n9 key Space up\n\
         12 key KeyR down\n13 key KeyR up\n",
    )
    .expect("the script parses");
    let mut session = Session::new(Bare::default(), SimConfig::default());
    let mut app =
        SessionApp::<Bare>::with_args(HostConfig::new("keys", Bindings::new()), Args::default())
            .pause_and_reset(&mut session);

    let mut seen = Vec::new();
    run_script(
        &mut session,
        &mut app,
        ScriptDriver::new(script),
        1.0,
        |_, session, _| {
            seen.push((session.paused(), session.current_tick().0));
            Ok(())
        },
    )
    .expect("the script ran");

    assert!(!seen[2].0, "the session was paused before Space");
    assert!(seen[6].0, "Space did not pause the session");
    assert_eq!(seen[6].1, seen[5].1, "a paused session ticked");
    assert!(!seen[11].0, "a second Space did not resume the session");
    assert!(seen[11].1 > seen[6].1, "a resumed session did not tick");
    assert!(
        seen[14].1 < seen[11].1,
        "R did not take the session back to its start: tick {} after tick {}",
        seen[14].1,
        seen[11].1
    );
}
