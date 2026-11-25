//! Interaction model follows the idTech console (Quake, 1996).

mod command_line;

pub use command_line::{parse_line, render_line, CommandLine};
use command_line::{quote_token, tokenize_spans};

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;

mod key;

pub use key::Key;

pub const MAX_HISTORY_LINES: usize = 2000;

pub const MAX_INPUT_HISTORY: usize = 100;

#[derive(Clone, Debug)]
pub struct HistoryLine {
    pub kind: LineKind,
    pub text: String,
}

impl HistoryLine {
    pub fn input(text: impl Into<String>) -> Self {
        Self {
            kind: LineKind::Input,
            text: text.into(),
        }
    }
    pub fn output(text: impl Into<String>) -> Self {
        Self {
            kind: LineKind::Output,
            text: text.into(),
        }
    }
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            kind: LineKind::Error,
            text: text.into(),
        }
    }
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            kind: LineKind::System,
            text: text.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineKind {
    Input,
    Output,
    Error,
    System,
}

pub struct ConsoleWriter {
    lines: Vec<HistoryLine>,
}

impl ConsoleWriter {
    pub fn new() -> Self {
        Self { lines: Vec::new() }
    }

    pub fn line(&mut self, text: impl Into<String>) {
        self.lines.push(HistoryLine::output(text));
    }

    pub fn error(&mut self, text: impl Into<String>) {
        self.lines.push(HistoryLine::error(text));
    }

    pub fn take_lines(&mut self) -> Vec<HistoryLine> {
        std::mem::take(&mut self.lines)
    }
}

impl Default for ConsoleWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Why a command refused its arguments.
#[derive(Clone, Debug, PartialEq)]
pub enum CommandError {
    /// The text is the command's form without the `usage: ` prefix.
    Usage(String),
    Invalid {
        name: &'static str,
        token: String,
        reason: String,
    },
    Unknown {
        name: &'static str,
        token: String,
        choices: &'static str,
    },
    OutOfRange {
        name: &'static str,
        value: f32,
        expected: &'static str,
    },
}

impl CommandError {
    pub fn invalid(name: &'static str, token: &str, reason: impl fmt::Display) -> Self {
        Self::Invalid {
            name,
            token: token.to_string(),
            reason: reason.to_string(),
        }
    }
}

impl fmt::Display for CommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(form) => write!(f, "usage: {form}"),
            Self::Invalid {
                name,
                token,
                reason,
            } => write!(f, "{name}: invalid `{token}`: {reason}"),
            Self::Unknown {
                name,
                token,
                choices,
            } => write!(f, "{name}: unknown `{token}` (try {choices})"),
            Self::OutOfRange {
                name,
                value,
                expected,
            } => write!(f, "{name}: {value} is out of range; expected {expected}"),
        }
    }
}

impl std::error::Error for CommandError {}

type CommandFn<Ctx> =
    Box<dyn FnMut(&[&str], &mut Ctx, &mut ConsoleWriter) -> Result<(), CommandError>>;

pub struct FnCommand<Ctx> {
    name: &'static str,
    help: &'static str,
    long_help: Option<&'static str>,
    arg_choices: Vec<Vec<&'static str>>,
    value_choices: HashMap<&'static str, Vec<&'static str>>,
    f: CommandFn<Ctx>,
}

impl<Ctx> FnCommand<Ctx> {
    /// One inner slice per position; trailing free-form args may be omitted.
    pub fn with_args(mut self, choices: &[&[&'static str]]) -> Self {
        self.arg_choices = choices.iter().map(|s| s.to_vec()).collect();
        self
    }

    /// First Tab completes to `key=`, subsequent Tabs cycle these values.
    pub fn with_value_choices(mut self, key: &'static str, values: &[&'static str]) -> Self {
        self.value_choices.insert(key, values.to_vec());
        self
    }

    /// `\n` breaks paint as separate scrollback entries.
    pub fn with_long_help(mut self, long: &'static str) -> Self {
        self.long_help = Some(long);
        self
    }

    fn long_help(&self) -> &'static str {
        self.long_help.unwrap_or(self.help)
    }

    fn arg_choices(&self, arg_index: usize) -> &[&'static str] {
        self.arg_choices
            .get(arg_index)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// `key` carries no trailing `=`; empty means free-form.
    fn value_choices(&self, key: &str) -> &[&'static str] {
        self.value_choices
            .get(key)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

/// Recoverable issues go to `out.error(..)`; unrecoverable ones return `Err`.
pub fn cmd<Ctx, F>(name: &'static str, help: &'static str, f: F) -> FnCommand<Ctx>
where
    F: FnMut(&[&str], &mut Ctx, &mut ConsoleWriter) -> Result<(), CommandError> + 'static,
{
    FnCommand {
        name,
        help,
        long_help: None,
        arg_choices: Vec::new(),
        value_choices: HashMap::new(),
        f: Box::new(f),
    }
}

pub struct Console<Ctx> {
    commands: BTreeMap<String, FnCommand<Ctx>>,
    toggle_key: Key,
    history: VecDeque<HistoryLine>,
    input: String,
    input_history: VecDeque<String>,
    input_history_pos: Option<usize>,
    tab: Option<TabState>,
    open: bool,
    pending_focus: bool,
    status: String,
    detached: bool,
    user_defocused: bool,
    pending_cursor_to_end: bool,
    pending: Vec<String>,
}

struct TabState {
    matches: Vec<String>,
    index: usize,
    ctx: CompletionContext,
}

#[derive(Clone, Debug)]
enum CompletionContext {
    Command {
        prefix: String,
    },
    Arg {
        cmd_name: String,
        arg_index: usize,
        prior: Vec<String>,
        prefix: String,
        start: usize,
    },
}

impl CompletionContext {
    fn prefix(&self) -> &str {
        match self {
            CompletionContext::Command { prefix } => prefix,
            CompletionContext::Arg { prefix, .. } => prefix,
        }
    }
}

impl<Ctx: 'static> Default for Console<Ctx> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Ctx: 'static> Console<Ctx> {
    pub fn new() -> Self {
        Self {
            commands: BTreeMap::new(),
            toggle_key: Key::Backtick,
            history: VecDeque::new(),
            input: String::new(),
            input_history: VecDeque::new(),
            input_history_pos: None,
            tab: None,
            open: false,
            pending_focus: false,
            status: String::new(),
            detached: false,
            user_defocused: false,
            pending_cursor_to_end: false,
            pending: Vec::new(),
        }
    }

    pub fn with_toggle_key(mut self, key: Key) -> Self {
        self.toggle_key = key;
        self
    }

    /// Silently replaces any existing command of the same name.
    pub fn register(&mut self, command: FnCommand<Ctx>) {
        self.commands.insert(command.name.to_string(), command);
    }

    pub fn has_command(&self, name: &str) -> bool {
        Builtin::from_name(name).is_some() || self.commands.contains_key(name)
    }

    pub fn open(&mut self) {
        if !self.open {
            self.open = true;
            self.pending_focus = true;
            self.user_defocused = false;
        }
    }

    pub fn close(&mut self) {
        self.open = false;
    }

    pub fn toggle(&mut self) {
        if self.open {
            self.close()
        } else {
            self.open()
        }
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    pub fn detach(&mut self) {
        self.detached = true;
    }

    pub fn dock(&mut self) {
        self.detached = false;
    }

    pub fn is_detached(&self) -> bool {
        self.detached
    }

    pub fn write(&mut self, line: HistoryLine) {
        self.push_history(line);
    }

    pub fn set_status(&mut self, text: impl Into<String>) {
        self.status = text.into();
    }

    pub fn toggle_key(&self) -> Key {
        self.toggle_key
    }

    pub fn history(&self) -> &VecDeque<HistoryLine> {
        &self.history
    }

    pub fn input(&self) -> &str {
        &self.input
    }

    /// An edit through this must be followed by [`Console::cancel_tab_cycle`].
    pub fn input_mut(&mut self) -> &mut String {
        &mut self.input
    }

    pub fn status(&self) -> &str {
        &self.status
    }

    pub fn submit(&mut self) {
        let line = std::mem::take(&mut self.input);
        self.execute(&line);
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    pub fn clear_input(&mut self) {
        self.input.clear();
        self.input_history_pos = None;
        self.tab = None;
    }

    pub fn cancel_tab_cycle(&mut self) {
        self.tab = None;
    }

    pub fn take_pending_focus(&mut self) -> bool {
        std::mem::take(&mut self.pending_focus)
    }

    pub fn wants_persistent_focus(&self) -> bool {
        !self.detached && !self.user_defocused
    }

    pub fn set_user_defocused(&mut self, defocused: bool) {
        self.user_defocused = defocused;
    }

    pub fn take_pending_cursor_to_end(&mut self) -> bool {
        std::mem::take(&mut self.pending_cursor_to_end)
    }

    pub fn history_prev(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        let pos = match self.input_history_pos {
            None => self.input_history.len() - 1,
            Some(0) => 0,
            Some(p) => p - 1,
        };
        self.input.clone_from(&self.input_history[pos]);
        self.input_history_pos = Some(pos);
        self.tab = None;
        self.pending_cursor_to_end = true;
    }

    pub fn history_next(&mut self) {
        let Some(pos) = self.input_history_pos else {
            return;
        };
        if pos + 1 >= self.input_history.len() {
            self.input.clear();
            self.input_history_pos = None;
        } else {
            self.input_history_pos = Some(pos + 1);
            self.input.clone_from(&self.input_history[pos + 1]);
        }
        self.tab = None;
        self.pending_cursor_to_end = true;
    }

    pub fn tab_complete(&mut self) {
        if let Some(tab) = self.tab.as_mut() {
            if !tab.matches.is_empty() {
                tab.index = (tab.index + 1) % tab.matches.len();
                let new_input = apply_completion(&self.input, &tab.ctx, &tab.matches[tab.index]);
                self.input = new_input;
                self.pending_cursor_to_end = true;
                return;
            }
        }
        let ctx = self
            .completion_context()
            .unwrap_or(CompletionContext::Command {
                prefix: String::new(),
            });
        let matches = self.completion_matches(&ctx);
        if matches.is_empty() {
            return;
        }
        self.input = apply_completion(&self.input, &ctx, &matches[0]);
        self.pending_cursor_to_end = true;
        if matches.len() > 1 {
            self.tab = Some(TabState {
                matches,
                index: 0,
                ctx,
            });
        } else {
            self.tab = None;
        }
    }

    fn push_history(&mut self, line: HistoryLine) {
        self.history.push_back(line);
        while self.history.len() > MAX_HISTORY_LINES {
            self.history.pop_front();
        }
    }

    fn all_command_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.commands.keys().cloned().collect();
        names.extend(Builtin::ALL.iter().map(|b| b.name().to_string()));
        names.sort();
        names
    }

    fn completion_context(&self) -> Option<CompletionContext> {
        if self.input.is_empty() {
            return None;
        }
        let parsed = tokenize_spans(&self.input);
        if parsed.is_empty() {
            return None;
        }
        let trailing_ws = parsed.last()?.end < self.input.len();

        if !trailing_ws {
            if let [only] = parsed.as_slice() {
                return Some(CompletionContext::Command {
                    prefix: only.value.clone(),
                });
            }
        }

        let mut parts = parsed;
        let cmd_name = parts.remove(0).value;
        let (arg_index, prefix, start) = if trailing_ws {
            let idx = parts.len();
            (idx, String::new(), self.input.len())
        } else {
            let partial = parts.pop()?;
            (parts.len(), partial.value, partial.start)
        };
        Some(CompletionContext::Arg {
            cmd_name,
            arg_index,
            prior: parts.into_iter().map(|part| part.value).collect(),
            prefix,
            start,
        })
    }

    fn completion_matches(&self, ctx: &CompletionContext) -> Vec<String> {
        match ctx {
            CompletionContext::Command { prefix } => self
                .all_command_names()
                .into_iter()
                .filter(|name| name.starts_with(prefix.as_str()))
                .collect(),
            CompletionContext::Arg {
                cmd_name,
                arg_index,
                prior,
                prefix,
                ..
            } => {
                let Some(cmd) = self.commands.get(cmd_name) else {
                    return Vec::new();
                };

                if let Some(eq) = prefix.find('=') {
                    let key = &prefix[..eq];
                    let value_prefix = &prefix[eq + 1..];
                    let mut matches: Vec<String> = cmd
                        .value_choices(key)
                        .iter()
                        .filter(|v| v.starts_with(value_prefix))
                        .map(|v| format!("{key}={v}"))
                        .collect();
                    matches.sort();
                    return matches;
                }

                let used_kv_prefixes: Vec<&str> = prior
                    .iter()
                    .filter_map(|t| t.find('=').map(|i| &t[..=i]))
                    .collect();

                let mut matches: Vec<String> = cmd
                    .arg_choices(*arg_index)
                    .iter()
                    .filter(|choice| choice.starts_with(prefix.as_str()))
                    .filter(|choice| match choice.find('=') {
                        None => true,
                        Some(eq) => {
                            let key = &choice[..=eq];
                            !used_kv_prefixes.contains(&key)
                        }
                    })
                    .map(|choice| (*choice).to_string())
                    .collect();
                matches.sort();
                matches
            }
        }
    }

    /// Suffix of the first (sort-order) completion; what the next Tab inserts.
    pub fn tab_preview(&self) -> Option<String> {
        let ctx = self.completion_context()?;
        let matches = self.completion_matches(&ctx);
        let first = matches.first()?;
        let prefix_len = ctx.prefix().len();
        if first.len() > prefix_len {
            Some(first[prefix_len..].to_string())
        } else {
            None
        }
    }

    /// Runs a built-in on the spot; parks anything else for [`Console::drain_pending`].
    pub fn execute(&mut self, line: &str) {
        let line = line.trim();
        if line.is_empty() {
            return;
        }

        if self.input_history.back().map(String::as_str) != Some(line) {
            self.input_history.push_back(line.to_string());
            while self.input_history.len() > MAX_INPUT_HISTORY {
                self.input_history.pop_front();
            }
        }
        self.input_history_pos = None;
        self.tab = None;

        let Some((name, args)) = parse_line(line) else {
            return;
        };

        if let Some(builtin) = Builtin::from_name(&name) {
            self.push_history(HistoryLine::input(format!("> {line}")));
            self.run_builtin(builtin, args.first().map(String::as_str));
            return;
        }

        self.pending.push(line.to_string());
    }

    pub fn drain_pending(&mut self) -> Vec<String> {
        std::mem::take(&mut self.pending)
    }

    /// Builtins run in `execute` and never reach this, so `name` is a registered command.
    pub fn dispatch(&mut self, name: &str, args: &[&str], ctx: &mut Ctx) {
        self.push_history(HistoryLine::input(format!("> {}", render_line(name, args))));

        let mut writer = ConsoleWriter::new();
        let result = match self.commands.get_mut(name) {
            Some(cmd) => (cmd.f)(args, ctx, &mut writer),
            None => {
                self.push_history(HistoryLine::error(format!(
                    "no command '{name}'. try: help"
                )));
                return;
            }
        };
        for hl in writer.lines {
            self.push_history(hl);
        }
        if let Err(e) = result {
            self.push_history(HistoryLine::error(format!("error: {e}")));
        }
    }

    fn run_builtin(&mut self, builtin: Builtin, target: Option<&str>) {
        match builtin {
            Builtin::Help => self.builtin_help(target),
            Builtin::Clear => self.history.clear(),
            Builtin::Detach => {
                self.detached = true;
                self.push_history(HistoryLine::system("console detached"));
            }
            Builtin::Dock => {
                self.detached = false;
                self.push_history(HistoryLine::system("console docked"));
            }
        }
    }

    fn builtin_help(&mut self, target: Option<&str>) {
        match target {
            Some(name) => {
                if let Some(b) = Builtin::from_name(name) {
                    self.push_history(HistoryLine::output(format!("{}: {}", b.name(), b.help())));
                } else {
                    let prepared: Option<Vec<String>> = self.commands.get(name).map(|c| {
                        let header_prefix = format!("{}: ", c.name);
                        let body = c.long_help();
                        let indent = " ".repeat(c.name.len() + 2);
                        let mut lines = body.lines();
                        let first = lines.next().unwrap_or("");
                        let mut rendered = vec![format!("{header_prefix}{first}")];
                        for line in lines {
                            rendered.push(format!("{indent}{line}"));
                        }
                        rendered
                    });
                    if let Some(lines) = prepared {
                        for line in lines {
                            self.push_history(HistoryLine::output(line));
                        }
                    } else {
                        self.push_history(HistoryLine::error(format!("no command '{name}'")));
                    }
                }
            }
            None => {
                self.push_history(HistoryLine::output("commands:"));
                let mut entries: Vec<(String, String)> = self
                    .commands
                    .values()
                    .map(|c| (c.name.to_string(), c.help.to_string()))
                    .collect();
                for b in Builtin::ALL {
                    entries.push((b.name().to_string(), b.help().to_string()));
                }
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                for (name, help) in entries {
                    self.push_history(HistoryLine::output(format!("  {name:16} {help}")));
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Builtin {
    Help,
    Clear,
    Detach,
    Dock,
}

impl Builtin {
    // Sorted by `name()`.
    const ALL: &'static [Builtin] = &[
        Builtin::Clear,
        Builtin::Detach,
        Builtin::Dock,
        Builtin::Help,
    ];

    fn from_name(name: &str) -> Option<Builtin> {
        match name {
            "help" => Some(Builtin::Help),
            "clear" => Some(Builtin::Clear),
            "detach" => Some(Builtin::Detach),
            "dock" => Some(Builtin::Dock),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Builtin::Help => "help",
            Builtin::Clear => "clear",
            Builtin::Detach => "detach",
            Builtin::Dock => "dock",
        }
    }

    fn help(self) -> &'static str {
        match self {
            Builtin::Help => "list commands or describe one",
            Builtin::Clear => "clear the scrollback buffer",
            Builtin::Detach => "render as a draggable window",
            Builtin::Dock => "render as a half-screen drop-down (default)",
        }
    }
}

fn apply_completion(input: &str, ctx: &CompletionContext, choice: &str) -> String {
    match ctx {
        CompletionContext::Command { .. } => choice.to_string(),
        CompletionContext::Arg { start, .. } => {
            format!("{}{}", &input[..*start], quote_token(choice))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Ctx = u32;

    fn run<C: 'static>(console: &mut Console<C>, line: &str, ctx: &mut C) {
        console.execute(line);
        drain_and_dispatch(console, ctx);
    }

    fn drain_and_dispatch<C: 'static>(console: &mut Console<C>, ctx: &mut C) {
        for line in console.drain_pending() {
            let Some((name, args)) = parse_line(&line) else {
                continue;
            };
            let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
            console.dispatch(&name, &arg_refs, ctx);
        }
    }

    fn echo_cmd() -> FnCommand<Ctx> {
        cmd("echo", "echo args back", |args, _ctx, out| {
            out.line(args.join(" "));
            Ok(())
        })
    }

    fn add_cmd() -> FnCommand<Ctx> {
        cmd("add", "add args to ctx", |args, ctx, out| {
            for a in args {
                let n: u32 = a
                    .parse()
                    .map_err(|error| CommandError::invalid("add", a, error))?;
                *ctx += n;
            }
            out.line(format!("ctx={ctx}"));
            Ok(())
        })
    }

    #[test]
    fn tab_completion() {
        let mut c = Console::<Ctx>::new();
        c.register(cmd("capture", "", |_, _, _| Ok(())).with_args(&[
            &["png", "frames", "toggle", "stop"],
            &["pre", "post", "both"],
        ]));
        for (input, preview) in [
            ("", None),
            ("zzz", None),
            ("de", Some("tach")),
            ("capture t", Some("oggle")),
            ("capture png ", Some("both")),
            ("capture png po", Some("st")),
            ("capture png post extra ", None),
        ] {
            c.input = input.into();
            assert_eq!(c.tab_preview().as_deref(), preview, "`{input}`");
        }

        c.input = "capture p".into();
        c.tab_complete();
        assert_eq!(c.input, "capture png");
        c.input = "capture png p".into();
        c.tab_complete();
        assert_eq!(c.input, "capture png post");
        c.tab_complete();
        assert_eq!(c.input, "capture png pre");

        let mut c = Console::<Ctx>::new();
        for name in ["capture.start", "capture.stop", "capture.toggle"] {
            c.register(cmd::<Ctx, _>(name, "x", |_, _, _| Ok(())));
        }
        c.input = "capture.s".into();
        c.tab_complete();
        assert_eq!(c.input, "capture.start");
        c.tab_complete();
        assert_eq!(c.input, "capture.stop");
        c.tab_complete();
        assert_eq!(c.input, "capture.start");
    }

    #[test]
    fn quoted_and_multibyte_completion() {
        let mut console = Console::<()>::new();
        console.register(cmd("mode", "set mode", |_, _, _| Ok(())).with_args(&[&["solid"]]));
        *console.input_mut() = "mode\u{2003}so".to_owned();
        console.tab_complete();
        assert_eq!(console.input(), "mode\u{2003}solid");

        for input in [
            "mode \"solid s",
            "mode 'solid s'",
            "mode \"solid ",
            "mode solid",
        ] {
            let mut console = Console::<()>::new();
            console.register(
                cmd("mode", "set mode", |_, _, _| Ok(()))
                    .with_args(&[&["solid state", "solid surface"]]),
            );
            *console.input_mut() = input.to_owned();
            console.tab_complete();
            assert_eq!(parse_line(console.input()).unwrap().1, ["solid state"]);
            console.tab_complete();
            assert_eq!(parse_line(console.input()).unwrap().1, ["solid surface"]);
        }

        let mut console = Console::<()>::new();
        console.register(cmd("mode", "set mode", |_, _, _| Ok(())).with_args(&[&["a\"b", "a\\b"]]));
        *console.input_mut() = "mode \"a\\\"".to_owned();
        console.tab_complete();
        assert_eq!(parse_line(console.input()).unwrap().1, ["a\"b"]);
    }

    #[test]
    fn builtin_help() {
        let mut c = Console::<Ctx>::new();
        c.register(echo_cmd());
        let mut ctx: Ctx = 0;
        run(&mut c, "help echo", &mut ctx);
        let last = c.history.back().unwrap();
        assert_eq!(last.kind, LineKind::Output);
        assert!(last.text.contains("echo"));
        assert!(last.text.contains("echo args back"));
    }

    #[test]
    fn input_history_recall() {
        let mut c = Console::<Ctx>::new();
        c.register(echo_cmd());
        let mut ctx: Ctx = 0;
        for line in ["echo first", "echo first", "echo second"] {
            run(&mut c, line, &mut ctx);
        }
        let recalled: Vec<&str> = c.input_history.iter().map(String::as_str).collect();
        assert_eq!(recalled, ["echo first", "echo second"]);

        c.history_prev();
        assert_eq!(c.input, "echo second");
        c.history_prev();
        assert_eq!(c.input, "echo first");
        c.history_prev();
        assert_eq!(c.input, "echo first");
        c.history_next();
        assert_eq!(c.input, "echo second");
        c.history_next();
        assert_eq!(c.input, "");
    }

    #[test]
    fn dispatch_echo() {
        let mut c = Console::<Ctx>::new();
        c.register(echo_cmd());
        c.register(cmd("fail", "always fails", |_, _, _| {
            Err(CommandError::Usage("nope".into()))
        }));
        let mut ctx: Ctx = 0;

        c.execute("echo hi");
        assert!(
            c.history.is_empty(),
            "a parked line has not run, so nothing is owed to the scrollback yet"
        );
        drain_and_dispatch(&mut c, &mut ctx);
        let kinds: Vec<LineKind> = c.history.iter().map(|l| l.kind).collect();
        assert_eq!(kinds, [LineKind::Input, LineKind::Output]);
        assert!(c.history[0].text.contains("echo hi"), "{:?}", c.history[0]);

        c.clear_history();
        c.dispatch("nonesuch", &[], &mut ctx);
        let kinds: Vec<LineKind> = c.history.iter().map(|l| l.kind).collect();
        assert_eq!(kinds, [LineKind::Input, LineKind::Error]);

        run(&mut c, "fail", &mut ctx);
        let last = c.history.back().unwrap();
        assert_eq!(last.kind, LineKind::Error);
        assert!(last.text.contains("nope"));
    }

    #[test]
    fn submit_typed_and_blank_lines() {
        let mut c = Console::<Ctx>::new();
        c.register(add_cmd());
        let mut ctx: Ctx = 1;

        *c.input_mut() = "   ".into();
        c.submit();
        drain_and_dispatch(&mut c, &mut ctx);
        assert!(c.history.is_empty());
        assert!(c.input_history.is_empty());

        *c.input_mut() = "add 4".into();
        c.submit();
        drain_and_dispatch(&mut c, &mut ctx);
        assert_eq!(ctx, 5);
        assert!(c.input.is_empty());
    }

    #[test]
    fn builtin_and_registry_timing() {
        let mut c = Console::<Ctx>::new();
        c.register(add_cmd());
        let mut ctx: Ctx = 0;

        c.execute("add 5");
        c.execute("clear");
        assert!(c.history.is_empty(), "clear must act on the typed frame");

        assert_eq!(
            c.drain_pending(),
            ["add 5"],
            "only the registry line queued"
        );

        c.execute("add 5");
        drain_and_dispatch(&mut c, &mut ctx);
        assert_eq!(ctx, 5);
    }
}
