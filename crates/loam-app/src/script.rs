use std::path::Path;

use anyhow::{anyhow, bail, Context as _, Result};

use loam_console::CommandLine;
use loam_runtime::Stores;

use crate::args::Args;
use crate::session::SessionConsole;

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

    pub fn advance_console<A: Stores>(&mut self, console: &mut SessionConsole<A>) -> ScriptStatus {
        self.advance_with(|command| {
            console.execute(&loam_console::render_line(
                &command.name,
                &command.arg_refs(),
            ));
        })
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

    #[test]
    fn a_malformed_script_or_a_detached_script_path_is_refused_and_a_misordered_one_names_its_line()
    {
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
