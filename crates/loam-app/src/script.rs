//! Frame scripts automate demos through the command queue; they are not simulation replays.

use std::path::Path;

use anyhow::{anyhow, bail, Context as _, Result};

use crate::args::Args;
use crate::command::CommandLine;

// Captures must finish after the last scripted frame presents.
const SETTLE_FRAMES: u64 = 60;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptStep {
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

    pub fn steps(&self) -> &[ScriptStep] {
        &self.steps
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
    steps: std::iter::Peekable<std::vec::IntoIter<ScriptStep>>,
    frame: u64,
    exit_frame: u64,
}

impl ScriptDriver {
    pub fn new(script: Script) -> Self {
        let exit_frame = script.last_frame().saturating_add(SETTLE_FRAMES);
        Self {
            steps: script.steps.into_iter().peekable(),
            frame: 0,
            exit_frame,
        }
    }

    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// The sink receives owned commands without a window or global inbox.
    pub fn advance_with(&mut self, mut submit: impl FnMut(CommandLine)) -> ScriptStatus {
        while self
            .steps
            .peek()
            .is_some_and(|step| step.frame <= self.frame)
        {
            if let Some(step) = self.steps.next() {
                submit(step.command);
            }
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

pub fn driver_from_args(args: &Args) -> Result<Option<ScriptDriver>> {
    if args.has_bare_flag("script") {
        bail!("--script needs its path attached: --script=path/to/file.script");
    }
    match args.get("script") {
        Some(path) => Ok(Some(ScriptDriver::new(Script::load(Path::new(path))?))),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::CommandQueue;

    #[test]
    fn invalid_frame_or_empty_command_is_rejected() {
        for source in ["10", " 10  ", "-1 reset", "1.5 reset", "abc reset"] {
            assert!(Script::parse(source).is_err(), "{source}");
        }
    }

    #[test]
    fn descending_frames_report_the_line() {
        let error = Script::parse("0 reset\n10 spin\n5 hud").unwrap_err();
        assert!(error.to_string().contains("line 3"));
    }

    #[test]
    fn script_requires_an_attached_path() {
        let args = Args::from_argv(["--script", "some.script"]);
        assert!(driver_from_args(&args)
            .unwrap_err()
            .to_string()
            .contains("--script="));
    }

    #[test]
    fn commands_fire_once_in_frame_and_file_order() {
        let mut driver = ScriptDriver::new(
            Script::parse("0 mark boot\n2 mark second-a\n2 mark second-b\n5 mark last").unwrap(),
        );
        let mut queue = CommandQueue::new();
        let mut fired = Vec::new();
        for frame in 0..8 {
            driver.advance_with(|command| queue.submit(command));
            fired.extend(
                queue
                    .drain()
                    .map(|command| (frame, command.args[0].clone())),
            );
        }
        assert_eq!(
            fired,
            [
                (0, "boot".to_string()),
                (2, "second-a".to_string()),
                (2, "second-b".to_string()),
                (5, "last".to_string()),
            ]
        );
    }

    #[test]
    fn finish_waits_for_capture_margin() {
        for source in ["", "5 mark end"] {
            let script = Script::parse(source).unwrap();
            let finish = script.last_frame() + SETTLE_FRAMES;
            let mut driver = ScriptDriver::new(script);
            for _ in 0..finish {
                assert_eq!(driver.advance_with(|_| {}), ScriptStatus::Running);
            }
            assert_eq!(driver.advance_with(|_| {}), ScriptStatus::Finished);
        }
    }
}
