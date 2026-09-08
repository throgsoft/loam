//! Runners drain the console inbox before each frame's simulation work.

use loam_egui::{Console, ConsoleWriter, HistoryLine};
use loam_render::device::RenderDevice;

use crate::App;

pub use loam_console::{CommandLine, CommandQueue};

const MAX_BUFFERED_OUTPUT: usize = 256;

impl crate::Runtime {
    pub fn submit(&self, command: CommandLine) {
        self.0.commands.borrow_mut().submit(command);
    }
    pub fn submit_line(&self, line: &str) -> bool {
        self.0.commands.borrow_mut().submit_line(line)
    }
    fn collect_pending(&self, batch: &mut CommandQueue) {
        std::mem::swap(&mut *self.0.commands.borrow_mut(), batch);
    }
}

pub struct CommandCtx<'a> {
    pub shader_db: &'a mut crate::ShaderDb,
    pub runtime: &'a crate::Runtime,
    pub rd: &'a RenderDevice,
    /// Next simulation tick; command dispatch also runs when simulation is paused.
    pub tick: u64,
    /// Wall-clock seconds since the runner started.
    pub time: f32,
    /// Drained into the output buffer after the command returns.
    pub out: &'a mut ConsoleWriter,
}

fn apply_engine_verb(
    runtime: &crate::Runtime,
    command: &CommandLine,
    out: &mut ConsoleWriter,
) -> bool {
    match command.name.as_str() {
        "vsync" => {
            crate::vsync::apply(runtime, &command.arg_refs(), out);
            true
        }
        _ => false,
    }
}

fn echo_line(command: &CommandLine) -> HistoryLine {
    HistoryLine::input(format!(
        "> {}",
        loam_egui::render_line(&command.name, &command.arg_refs())
    ))
}

pub(crate) fn apply_drained<A: App>(
    app: &mut A,
    runtime: &crate::Runtime,
    shader_db: &mut crate::ShaderDb,
    rd: &RenderDevice,
    tick: u64,
    time: f32,
    batch: &mut CommandQueue,
) {
    runtime.collect_pending(batch);
    for command in batch.drain() {
        let mut writer = ConsoleWriter::new();
        let claimed = apply_engine_verb(runtime, &command, &mut writer);
        let result = if claimed {
            Ok(())
        } else {
            let mut ctx = CommandCtx {
                shader_db,
                runtime,
                rd,
                tick,
                time,
                out: &mut writer,
            };
            app.apply_command(&command, &mut ctx)
        };
        let mut lines = writer.take_lines();
        if claimed {
            lines.insert(0, echo_line(&command));
        }
        if let Err(e) = result {
            let text = format!("error: {}: {e:#}", command.name);
            tracing::error!("command: {text}");
            lines.push(HistoryLine::error(text));
        }
        if !lines.is_empty() {
            let mut buffered = runtime.0.output.borrow_mut();
            buffered.extend(lines);
            let overflow = buffered.len().saturating_sub(MAX_BUFFERED_OUTPUT);
            buffered.drain(..overflow);
        }
    }
}

impl crate::Runtime {
    /// Drain before painting the console.
    pub fn pump_console<Ctx: 'static>(&self, console: &mut Console<Ctx>) {
        for line in self.0.output.borrow_mut().drain(..) {
            console.write(line);
        }
    }
    /// Forward after the console UI returns.
    pub fn forward_console<Ctx: 'static>(&self, console: &mut Console<Ctx>) {
        for line in console.drain_pending() {
            self.submit_line(&line);
        }
    }
}

#[cfg(test)]
pub(crate) fn run_on_console<Ctx: 'static>(console: &mut Console<Ctx>, line: &str, ctx: &mut Ctx) {
    console.execute(line);
    for pending in console.drain_pending() {
        if let Some(parsed) = CommandLine::parse(&pending) {
            console.dispatch(&parsed.name, &parsed.arg_refs(), ctx);
        }
    }
}
