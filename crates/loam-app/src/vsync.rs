//! Under `PresentMode::Fifo` the acquire in `RenderDevice::begin_frame` blocks
//! at vsync, so the [`crate::fps`] cap cannot exceed the refresh rate;
//! `vsync off` swaps to `Mailbox` or `Immediate`.

use loam_egui::{cmd, Console, ConsoleWriter};

use crate::Runtime;

/// Reached from the runner's verb table ([`crate::command`]) before any App hook.
pub(crate) fn apply(runtime: &Runtime, args: &[&str], out: &mut ConsoleWriter) {
    match args.first().copied() {
        None => {
            out.line("vsync: use 'vsync on' (Fifo) or 'vsync off' (Mailbox/Immediate)");
        }
        Some("on") => {
            runtime.request_vsync(true);
            out.line("vsync: requested ON (Fifo); applies on next frame");
        }
        Some("off") => {
            runtime.request_vsync(false);
            out.line(
                "vsync: requested OFF (Mailbox preferred, Immediate fallback) \
                 ; applies on next frame",
            );
        }
        Some(other) => {
            out.line(format!(
                "vsync: unknown subcommand '{other}' (try 'on' or 'off')"
            ));
        }
    }
}

pub fn register_command<Ctx: 'static>(console: &mut Console<Ctx>, runtime: &Runtime) {
    let runtime = runtime.clone();
    console.register(
        cmd(
            "vsync",
            "show or set the surface present mode (on = Fifo, off = Mailbox/Immediate)",
            move |args, _ctx: &mut Ctx, out| {
                apply(&runtime, args, out);
                Ok(())
            },
        )
        .with_args(&[&["on", "off"]]),
    );
}
