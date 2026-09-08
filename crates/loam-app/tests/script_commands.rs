use loam_app::script::{Script, ScriptDriver};
use loam_console::{cmd, CommandQueue, Console};

#[test]
fn frame_scripts_use_console_dispatch() {
    let mut console = Console::<Vec<String>>::new();
    console.register(cmd(
        "mark",
        "record a marker",
        |args, markers: &mut Vec<String>, _| {
            markers.extend(args.iter().map(|arg| (*arg).to_string()));
            Ok(())
        },
    ));
    let mut driver = ScriptDriver::new(
        Script::parse("# comment\n\n0 mark \"a b\" #ff8800\n1 detach\n1 nonesuch").unwrap(),
    );
    let mut queue = CommandQueue::new();
    let mut markers = Vec::new();
    for _ in 0..3 {
        driver.advance_with(|command| queue.submit(command));
        for command in queue.drain() {
            console.dispatch(&command.name, &command.arg_refs(), &mut markers);
        }
    }
    assert_eq!(markers, ["a b", "#ff8800"]);
    assert!(console.is_detached());
    assert!(console
        .history()
        .iter()
        .any(|line| line.text.contains("no command 'nonesuch'")));
}
