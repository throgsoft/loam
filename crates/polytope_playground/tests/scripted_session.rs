use std::path::PathBuf;
use std::process::{Command, Output};

fn run(name: &str, script: &str, flags: &[&str]) -> Output {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}.script"));
    std::fs::write(&path, script).expect("the script file is written");
    Command::new(env!("CARGO_BIN_EXE_polytope_playground"))
        .args(flags)
        .arg(format!("--script={}", path.display()))
        .output()
        .expect("the playground binary runs")
}

fn records(name: &str, script: &str) -> Output {
    run(name, script, &["--headless", "--records"])
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

// The entity of every instance line, grouped by the publication it follows.
fn instances(output: &Output) -> Vec<Vec<String>> {
    assert!(output.status.success(), "{}", stderr(output));
    let mut dumps: Vec<Vec<String>> = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if line.starts_with("publication ") {
            dumps.push(Vec::new());
        } else if let Some(rest) = line.strip_prefix("instance entity ") {
            let entity = rest.split(' ').next().expect("an entity id follows");
            dumps
                .last_mut()
                .expect("an instance follows a publication")
                .push(entity.to_owned());
        }
    }
    dumps
}

#[test]
fn single_view_dumps_only_the_active_slot_and_shapes_brings_back_the_row() {
    let output = records(
        "single_view",
        "2 view single\n4 active 2\n30 view shapes\n50 active 2\n",
    );
    let dumps = instances(&output);
    let row = dumps.last().expect("the run dumps a publication");
    assert!(row.len() > 2, "the whole row is back: {row:?}");
    assert_eq!(dumps[0], [row[2].clone()]);
}

#[test]
fn view_turns_the_filmstrip_off() {
    let output = records(
        "filmstrip_view",
        "2 key KeyF down\n3 key KeyF up\n25 view shapes\n45 view shapes\n",
    );
    let dumps = instances(&output);
    assert!(dumps[0].is_empty(), "the filmstrip hides the row");
    let row = dumps.last().expect("the run dumps a publication");
    assert!(row.len() > 1, "{row:?}");
}

#[test]
fn view_with_an_unknown_argument_fails_the_run_with_its_usage() {
    let output = records("view_usage", "2 view all\n20 view shapes\n");
    assert!(!output.status.success());
    assert!(stderr(&output).contains("usage: view"));
}

#[test]
fn a_script_without_records_is_refused() {
    let output = run("no_records", "2 view shapes\n", &["--headless"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("--records"));
}

// The overlay lines of each publication, and the string of each overlay text label.
fn overlays(output: &Output) -> Vec<(usize, Vec<String>)> {
    assert!(output.status.success(), "{}", stderr(output));
    let mut dumps: Vec<(usize, Vec<String>)> = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if line.starts_with("publication ") {
            dumps.push((0, Vec::new()));
        } else if line.starts_with("overlay line ") {
            dumps.last_mut().expect("a line follows a publication").0 += 1;
        } else if let Some(rest) = line.strip_prefix("overlay text ") {
            let (text, _) = rest.rsplit_once(" at ").expect("a place follows the text");
            dumps
                .last_mut()
                .expect("a label follows a publication")
                .1
                .push(text.to_owned());
        }
    }
    dumps
}

#[test]
fn the_toybox_guides_and_the_readout_reach_the_publication_and_leave_it_when_turned_off() {
    let output = records(
        "overlay",
        "2 demo toybox\n2 hud on\n25 demo rotate\n25 handles off\n25 hud off\n45 view shapes\n",
    );
    let dumps = overlays(&output);
    let (lines, labels) = &dumps[0];
    assert_eq!(*lines, 4, "the arena rectangle has four sides");
    let [shadow, body] = &labels[..] else {
        panic!("the readout is a shadow and a body: {labels:?}");
    };
    assert_eq!(shadow, body);
    let last = dumps.last().expect("the run dumps a publication");
    assert_eq!(*last, (0, Vec::new()), "rotate mode with handles off");
}
