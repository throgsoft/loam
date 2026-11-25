use std::fmt;
use std::io;
use std::num::ParseIntError;
use std::path::Path;

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

/// Why a script did not load; a line number counts from 1.
#[derive(Debug)]
pub enum ScriptError {
    Read(io::Error),
    NoCommand {
        line: usize,
        text: String,
    },
    BadFrame {
        line: usize,
        token: String,
        error: ParseIntError,
    },
    OutOfOrder {
        line: usize,
        frame: u64,
        previous: u64,
    },
}

impl fmt::Display for ScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(error) => write!(f, "cannot read: {error}"),
            Self::NoCommand { line, text } => {
                write!(f, "line {line}: `{text}` names a frame and no command")
            }
            Self::BadFrame { line, token, error } => {
                write!(f, "line {line}: `{token}` is not a frame index: {error}")
            }
            Self::OutOfOrder {
                line,
                frame,
                previous,
            } => write!(f, "line {line}: frame {frame} precedes frame {previous}"),
        }
    }
}

impl std::error::Error for ScriptError {}

/// Equal frame indices preserve file order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Script {
    steps: Vec<ScriptStep>,
}

impl Script {
    /// Skips blank and `#` lines; a mid-line `#` belongs to the command.
    pub fn parse(source: &str) -> Result<Self, ScriptError> {
        let mut steps: Vec<ScriptStep> = Vec::new();
        for (index, raw) in source.lines().enumerate() {
            let number = index + 1;
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let no_command = || ScriptError::NoCommand {
                line: number,
                text: line.to_string(),
            };
            let (frame_token, rest) = line
                .split_once(char::is_whitespace)
                .ok_or_else(no_command)?;
            let frame: u64 = frame_token.parse().map_err(|error| ScriptError::BadFrame {
                line: number,
                token: frame_token.to_string(),
                error,
            })?;
            if let Some(previous) = steps.last() {
                if frame < previous.frame {
                    return Err(ScriptError::OutOfOrder {
                        line: number,
                        frame,
                        previous: previous.frame,
                    });
                }
            }
            steps.push(ScriptStep {
                line: number,
                frame,
                command: CommandLine::parse(rest.trim()).ok_or_else(no_command)?,
            });
        }
        Ok(Self { steps })
    }

    pub fn load(path: &Path) -> Result<Self, ScriptError> {
        let source = std::fs::read_to_string(path).map_err(ScriptError::Read)?;
        Self::parse(&source)
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

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn with_frames(mut self, frames: std::num::NonZeroU64) -> Self {
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

pub(crate) fn driver_from_args(args: &Args) -> Result<Option<ScriptDriver>, HostError> {
    if args.has_bare_flag("script") {
        return Err(HostError::Detached {
            flag: "script",
            example: "--script=path/to/file.script",
        });
    }
    let Some(path) = args.get("script") else {
        return Ok(None);
    };
    let script = Script::load(Path::new(path)).map_err(|error| HostError::Script {
        path: path.into(),
        error,
    })?;
    Ok(Some(ScriptDriver::new(script)))
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
            .map(|step| parse(console, &step.command).map_err(|error| refused(step, error)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { driver, lines })
    }

    pub(crate) fn frame(&self) -> u64 {
        self.driver.frame()
    }

    // Says whether a line ran. The driver advances on an error too, so a host that carries on stays on its frame index.
    pub(crate) fn advance<A: Stores>(
        &mut self,
        input: &mut InputMap,
        bindings: &Bindings,
        console: &mut SessionConsole<A>,
    ) -> Result<(ScriptStatus, bool), HostError> {
        let lines = &self.lines;
        let mut ran = false;
        let mut failed = console
            .take_refused()
            .map(|(line, result)| HostError::ScriptRefused { line, result });
        let status = self.driver.advance_with(|index, step| {
            ran = true;
            match &lines[index] {
                Line::Input(event) => event.apply(input, bindings),
                Line::Console => {
                    let text = render_line(&step.command.name, &step.command.arg_refs());
                    let error = console.dispatch_script_line(&text, step.line);
                    if failed.is_none() {
                        failed = error.map(|error| refused(step, error));
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

fn refused(step: &ScriptStep, reason: String) -> HostError {
    HostError::ScriptLine {
        line: step.line,
        text: render_line(&step.command.name, &step.command.arg_refs()),
        reason,
    }
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
        for source in ["10", " 10  "] {
            assert!(
                matches!(
                    Script::parse(source),
                    Err(ScriptError::NoCommand { line: 1, .. })
                ),
                "{source}"
            );
        }
        for source in ["-1 reset", "1.5 reset", "abc reset"] {
            assert!(
                matches!(
                    Script::parse(source),
                    Err(ScriptError::BadFrame { line: 1, .. })
                ),
                "{source}"
            );
        }
        assert!(matches!(
            Script::parse("0 reset\n10 spin\n5 hud"),
            Err(ScriptError::OutOfOrder {
                line: 3,
                frame: 5,
                previous: 10
            })
        ));

        let args = Args::from_argv(["--script", "some.script"]);
        assert!(matches!(
            driver_from_args(&args),
            Err(HostError::Detached { flag: "script", .. })
        ));
    }
}
