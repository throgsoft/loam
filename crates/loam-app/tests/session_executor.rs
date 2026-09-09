#![cfg(not(target_arch = "wasm32"))]

use loam_runtime::{Bindings, HostConfig, SimConfig};

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Ticks {
        counters: Store<u32>,
    }
}

#[test]
fn the_session_host_installs_the_native_executor_before_it_opens_a_window() {
    assert_eq!(loam_time::par::executor().parallelism(), 1);

    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let entered = std::panic::catch_unwind(|| {
        let session = loam_runtime::Session::new(Ticks::default(), SimConfig::default());
        loam_app::session::run(session, HostConfig::new("ticks", Bindings::new()))
    });
    std::panic::set_hook(hook);
    assert!(
        entered.is_err(),
        "winit accepted an event loop off the main thread, so the host ran past the install"
    );

    assert!(
        loam_time::par::executor().parallelism() > 1,
        "one core, or the host opened its window before installing the executor"
    );
}
