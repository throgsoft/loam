use std::num::NonZeroU64;
use std::path::Path;

use anyhow::{anyhow, bail, Context as _, Result};

use loam_console::{render_line, CommandLine};
use loam_runtime::{PointerButton, Stores};

use crate::args::Args;
use crate::host::{Bindings, HostError, Key};
use crate::session::input::{dom_key, InputMap};
use crate::session::SessionConsole;

// Captures must finish after the last scripted frame presents.
const SETTLE_FRAMES: u64 = 60;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptStep {
    /// 1-based, counting blank and comment lines.
    pub line: usize,
    pub frame: u64,
    pub command: CommandLine,
}

/// Equal frame indices preserve file order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Script {
    steps: Vec<ScriptStep>,
}

impl Script {
    /// Skips blank and `#` lines; a mid-line `#` belongs to the command.
    pub fn parse(source: &str) -> Result<Self> {
        let mut steps: Vec<ScriptStep> = Vec::new();
        for (index, raw) in source.lines().enumerate() {
            let number = index + 1;
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (frame_token, rest) = line
                .split_once(char::is_whitespace)
                .ok_or_else(|| anyhow!("line {number}: `{line}` names a frame and no command"))?;
            let frame: u64 = frame_token
                .parse()
                .map_err(|e| anyhow!("line {number}: `{frame_token}` is not a frame index: {e}"))?;
            let command = rest.trim();
            if command.is_empty() {
                bail!("line {number}: `{line}` names a frame and no command");
            }
            if let Some(previous) = steps.last() {
                if frame < previous.frame {
                    bail!(
                        "line {number}: frame {frame} precedes frame {}",
                        previous.frame
                    );
                }
            }
            steps.push(ScriptStep {
                line: number,
                frame,
                command: CommandLine::parse(command)
                    .ok_or_else(|| anyhow!("line {number}: no command tokens"))?,
            });
        }
        Ok(Self { steps })
    }

    pub fn load(path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("reading script {}", path.display()))?;
        Self::parse(&source).with_context(|| format!("parsing script {}", path.display()))
    }

    pub fn last_frame(&self) -> u64 {
        self.steps.last().map_or(0, |step| step.frame)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScriptStatus {
    Running,
    Finished,
}

#[derive(Debug)]
pub struct ScriptDriver {
    steps: Vec<ScriptStep>,
    next: usize,
    frame: u64,
    exit_frame: u64,
}

impl ScriptDriver {
    pub fn new(script: Script) -> Self {
        let exit_frame = script.last_frame().saturating_add(SETTLE_FRAMES);
        Self {
            steps: script.steps,
            next: 0,
            frame: 0,
            exit_frame,
        }
    }

    pub(crate) fn with_frames(mut self, frames: NonZeroU64) -> Self {
        self.exit_frame = frames.get() - 1;
        self
    }

    pub fn frame(&self) -> u64 {
        self.frame
    }

    pub fn steps(&self) -> &[ScriptStep] {
        &self.steps
    }

    /// Hands `submit` each step due by this frame with its index in [`ScriptDriver::steps`].
    pub fn advance_with(&mut self, mut submit: impl FnMut(usize, &ScriptStep)) -> ScriptStatus {
        while let Some(step) = self
            .steps
            .get(self.next)
            .filter(|step| step.frame <= self.frame)
        {
            submit(self.next, step);
            self.next += 1;
        }
        let status = if self.frame >= self.exit_frame {
            ScriptStatus::Finished
        } else {
            ScriptStatus::Running
        };
        self.frame = self.frame.saturating_add(1);
        status
    }
}

pub(crate) fn driver_from_args(args: &Args) -> Result<Option<ScriptDriver>> {
    if args.has_bare_flag("script") {
        bail!("--script needs its path attached: --script=path/to/file.script");
    }
    match args.get("script") {
        Some(path) => Ok(Some(ScriptDriver::new(Script::load(Path::new(path))?))),
        None => Ok(None),
    }
}

// Windowed and headless hosts both run a script through this, with its lines checked against the console up front.
pub(crate) struct ScriptRun {
    driver: ScriptDriver,
    lines: Vec<Line>,
}

impl ScriptRun {
    pub(crate) fn new<A: Stores>(
        driver: ScriptDriver,
        console: &SessionConsole<A>,
    ) -> Result<Self, HostError> {
        let lines = driver
            .steps()
            .iter()
            .map(|step| parse(console, &step.command).map_err(|error| refused(step, &error)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { driver, lines })
    }

    pub(crate) fn frame(&self) -> u64 {
        self.driver.frame()
    }

    // Says whether a line ran. The error names the first line refused here or rejected at the last boundary.
    pub(crate) fn advance<A: Stores>(
        &mut self,
        input: &mut InputMap,
        bindings: &Bindings,
        console: &mut SessionConsole<A>,
    ) -> Result<(ScriptStatus, bool), HostError> {
        if let Some((number, error)) = console.take_refused() {
            return Err(HostError::Host(format!("script line {number}: {error}")));
        }
        let lines = &self.lines;
        let mut ran = false;
        let mut failed = None;
        let status = self.driver.advance_with(|index, step| {
            ran = true;
            match &lines[index] {
                Line::Input(event) => event.apply(input, bindings),
                Line::Console => {
                    let text = render_line(&step.command.name, &step.command.arg_refs());
                    let error = console.dispatch_script_line(&text, step.line);
                    if failed.is_none() {
                        failed = error.map(|error| refused(step, &error));
                    }
                }
            }
        });
        failed.map_or(Ok((status, ran)), Err)
    }
}

enum Line {
    Input(Event),
    Console,
}

#[derive(Clone, Copy)]
enum Event {
    Key(Key, bool),
    Move([f32; 2]),
    Button([f32; 2], PointerButton, bool),
    Wheel(f32),
    Look(f32, f32),
}

impl Event {
    fn apply(self, input: &mut InputMap, bindings: &Bindings) {
        match self {
            Event::Key(key, down) => input.action(bindings, key, down),
            Event::Move(at) => input.moved(at),
            Event::Button(at, button, down) => input.button(at, button, down),
            Event::Wheel(lines) => input.wheel([0.0, lines]),
            Event::Look(dx, dy) => input.raw_motion(dx, dy),
        }
    }
}

fn parse<A: Stores>(
    console: &SessionConsole<A>,
    line: &CommandLine,
) -> std::result::Result<Line, String> {
    let args = line.arg_refs();
    let event = match (line.name.as_str(), args.as_slice()) {
        ("key", [code, edge]) => Event::Key(
            dom_key(code).ok_or_else(|| format!("`{code}` is not a key code"))?,
            pressed(edge)?,
        ),
        ("pointer", [x, y, "move"]) => Event::Move([number(x)?, number(y)?]),
        ("pointer", [x, y, edge, button @ ..]) => {
            let button = match button {
                [] | ["primary"] => PointerButton::Primary,
                ["secondary"] => PointerButton::Secondary,
                ["middle"] => PointerButton::Middle,
                _ => return Err("the button is primary, secondary or middle".into()),
            };
            Event::Button([number(x)?, number(y)?], button, pressed(edge)?)
        }
        ("wheel", [lines]) => Event::Wheel(number(lines)?),
        ("look", [dx, dy]) => Event::Look(number(dx)?, number(dy)?),
        ("key", _) => return Err("usage: key <code> down|up".into()),
        ("pointer", _) => return Err("usage: pointer <x> <y> down|up [button] | move".into()),
        ("wheel", _) => return Err("usage: wheel <lines>".into()),
        ("look", _) => return Err("usage: look <dx> <dy>".into()),
        (name, _) if console.has(name) => return Ok(Line::Console),
        (name, _) => return Err(format!("no command '{name}'")),
    };
    Ok(Line::Input(event))
}

fn refused(step: &ScriptStep, error: &str) -> HostError {
    let text = render_line(&step.command.name, &step.command.arg_refs());
    HostError::Host(format!("script line {}: `{text}`: {error}", step.line))
}

fn pressed(edge: &str) -> std::result::Result<bool, String> {
    match edge {
        "down" => Ok(true),
        "up" => Ok(false),
        _ => Err(format!("`{edge}` is not down or up")),
    }
}

fn number(token: &str) -> std::result::Result<f32, String> {
    token
        .parse::<f32>()
        .ok()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("`{token}` is not a finite number"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_filename() {
        for source in ["10", " 10  ", "-1 reset", "1.5 reset", "abc reset"] {
            assert!(Script::parse(source).is_err(), "{source}");
        }
        let error = Script::parse("0 reset\n10 spin\n5 hud").unwrap_err();
        assert!(error.to_string().contains("line 3"));

        let args = Args::from_argv(["--script", "some.script"]);
        assert!(driver_from_args(&args)
            .unwrap_err()
            .to_string()
            .contains("--script="));
    }
}
