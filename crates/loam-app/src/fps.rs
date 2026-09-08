use loam_egui::{cmd, Console};

use crate::Runtime;

const MAX_ACCEPTED_FPS: f32 = 1000.0;

fn print_current(runtime: &Runtime, out: &mut loam_egui::ConsoleWriter) {
    let f = runtime.target_fps();
    if f <= 0.0 {
        out.line("fps: unlimited (uncapped; surface/vsync or browser RAF is the upper bound)");
    } else {
        out.line(format!("fps: target {f:.1}"));
    }
}

pub fn register_command<Ctx: 'static>(console: &mut Console<Ctx>, runtime: &Runtime) {
    let runtime = runtime.clone();
    console.register(
        cmd(
            "fps",
            "show or set the frame cap; unlimited removes it",
            move |args, _ctx: &mut Ctx, out| {
                match args.first().copied() {
                    None => print_current(&runtime, out),
                    Some("unlimited") | Some("off") | Some("0") => {
                        runtime.set_target_fps(0.0);
                        out.line("fps: unlimited (cap removed)");
                    }
                    Some(s) => match s.parse::<f32>() {
                        Ok(f) if f > 0.0 && f <= MAX_ACCEPTED_FPS => {
                            runtime.set_target_fps(f);
                            out.line(format!("fps: target set to {f:.1}"));
                        }
                        _ => {
                            out.line(format!(
                                "usage: fps [<n> | unlimited]  (n in (0, {MAX_ACCEPTED_FPS:.0}])"
                            ));
                        }
                    },
                }
                Ok(())
            },
        )
        .with_args(&[&["unlimited", "off", "30", "60", "120", "144", "240"]]),
    );
}
