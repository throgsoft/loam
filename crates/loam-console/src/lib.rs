//! Interaction model follows the idTech console (Quake, 1996).

mod queue;

pub use queue::CommandLine;

use std::collections::{BTreeMap, HashMap, VecDeque};

mod key;

pub use key::Key;

pub const MAX_HISTORY_LINES: usize = 2000;

pub const MAX_INPUT_HISTORY: usize = 100;

#[cfg(target_arch = "wasm32")]
static ECHO_TO_BROWSER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// wasm32 only; a no-op on native.
pub fn set_console_echo(enabled: bool) {
    #[cfg(target_arch = "wasm32")]
    ECHO_TO_BROWSER.store(enabled, std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(target_arch = "wasm32"))]
    let _ = enabled;
}

pub fn console_echo_enabled() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        ECHO_TO_BROWSER.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        false
    }
}

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

pub trait Command<Ctx>: 'static {
    fn name(&self) -> &str;

    fn help(&self) -> &str;

    /// `\n` breaks paint as separate scrollback entries.
    fn long_help(&self) -> String {
        self.help().to_string()
    }

    fn arg_choices(&self, arg_index: usize) -> &[&'static str] {
        let _ = arg_index;
        &[]
    }

    /// `prior` is the arg tokens before the cursor.
    fn arg_choices_ctx<'a>(&'a self, arg_index: usize, prior: &[&str]) -> &'a [&'static str] {
        let _ = prior;
        self.arg_choices(arg_index)
    }

    /// `key` carries no trailing `=`; empty means free-form.
    fn arg_value_choices(&self, arg_index: usize, key: &str) -> &[&'static str] {
        let _ = (arg_index, key);
        &[]
    }

    fn arg_value_choices_ctx<'a>(
        &'a self,
        arg_index: usize,
        key: &str,
        prior: &[&str],
    ) -> &'a [&'static str] {
        let _ = prior;
        self.arg_value_choices(arg_index, key)
    }

    /// Recoverable issues go to `out.error(..)`; unrecoverable ones return `Err`.
    fn run(&mut self, args: &[&str], ctx: &mut Ctx, out: &mut ConsoleWriter) -> anyhow::Result<()>;
}

pub struct FnCommand<F> {
    name: &'static str,
    help: &'static str,
    long_help: Option<&'static str>,
    arg_choices: Vec<Vec<&'static str>>,
    value_choices: HashMap<&'static str, Vec<&'static str>>,
    f: F,
}

impl<F> FnCommand<F> {
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

    pub fn with_long_help(mut self, long: &'static str) -> Self {
        self.long_help = Some(long);
        self
    }
}

pub fn cmd<Ctx, F>(name: &'static str, help: &'static str, f: F) -> FnCommand<F>
where
    F: FnMut(&[&str], &mut Ctx, &mut ConsoleWriter) -> anyhow::Result<()> + 'static,
{
    FnCommand {
        name,
        help,
        long_help: None,
        arg_choices: Vec::new(),
        value_choices: HashMap::new(),
        f,
    }
}

impl<Ctx, F> Command<Ctx> for FnCommand<F>
where
    F: FnMut(&[&str], &mut Ctx, &mut ConsoleWriter) -> anyhow::Result<()> + 'static,
{
    fn name(&self) -> &str {
        self.name
    }
    fn help(&self) -> &str {
        self.help
    }
    fn long_help(&self) -> String {
        self.long_help
            .map(str::to_string)
            .unwrap_or_else(|| self.help.to_string())
    }
    fn arg_choices(&self, arg_index: usize) -> &[&'static str] {
        self.arg_choices
            .get(arg_index)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
    fn arg_value_choices(&self, _arg_index: usize, key: &str) -> &[&'static str] {
        self.value_choices
            .get(key)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
    fn run(&mut self, args: &[&str], ctx: &mut Ctx, out: &mut ConsoleWriter) -> anyhow::Result<()> {
        (self.f)(args, ctx, out)
    }
}

type ToggleHandler<Ctx> = Box<dyn FnMut(&mut Ctx, Option<bool>) -> anyhow::Result<()>>;

type ChoiceHandler<Ctx> = Box<dyn FnMut(&mut Ctx, Option<&str>) -> anyhow::Result<()>>;

type BareHandler<Ctx> = Box<dyn FnMut(&mut Ctx) -> anyhow::Result<()>>;

type CustomHandler<Ctx> =
    Box<dyn FnMut(&mut Ctx, &[&str], &mut ConsoleWriter) -> anyhow::Result<()>>;

enum SubcommandKind<Ctx> {
    Toggle {
        handler: ToggleHandler<Ctx>,
    },
    Choice {
        choices: Vec<&'static str>,
        handler: ChoiceHandler<Ctx>,
    },
    Custom {
        arg_choices: Vec<Vec<&'static str>>,
        value_choices: HashMap<&'static str, Vec<&'static str>>,
        handler: CustomHandler<Ctx>,
    },
}

struct SubcommandEntry<Ctx> {
    help: &'static str,
    kind: SubcommandKind<Ctx>,
}

pub struct SubcommandSet<Ctx> {
    name: &'static str,
    help: &'static str,
    subs: BTreeMap<&'static str, SubcommandEntry<Ctx>>,
    name_cache: std::cell::OnceCell<Vec<&'static str>>,
    bare: Option<BareHandler<Ctx>>,
    long_help: Option<&'static str>,
}

impl<Ctx: 'static> SubcommandSet<Ctx> {
    /// Parses `on|off|true|false|1|0`; a bare invocation passes `None`.
    pub fn toggle<F>(mut self, name: &'static str, help: &'static str, handler: F) -> Self
    where
        F: FnMut(&mut Ctx, Option<bool>) -> anyhow::Result<()> + 'static,
    {
        self.name_cache.take();
        self.subs.insert(
            name,
            SubcommandEntry {
                help,
                kind: SubcommandKind::Toggle {
                    handler: Box::new(handler),
                },
            },
        );
        self
    }

    /// The handler receives the raw value string, unvalidated against `choices`.
    pub fn choice<F>(
        mut self,
        name: &'static str,
        help: &'static str,
        choices: &[&'static str],
        handler: F,
    ) -> Self
    where
        F: FnMut(&mut Ctx, Option<&str>) -> anyhow::Result<()> + 'static,
    {
        self.name_cache.take();
        self.subs.insert(
            name,
            SubcommandEntry {
                help,
                kind: SubcommandKind::Choice {
                    choices: choices.to_vec(),
                    handler: Box::new(handler),
                },
            },
        );
        self
    }

    /// `arg_choices[i]` completes the i-th arg after the subcommand name.
    pub fn custom<F>(
        mut self,
        name: &'static str,
        help: &'static str,
        arg_choices: &[&[&'static str]],
        value_choices: &[(&'static str, &[&'static str])],
        handler: F,
    ) -> Self
    where
        F: FnMut(&mut Ctx, &[&str], &mut ConsoleWriter) -> anyhow::Result<()> + 'static,
    {
        let mut vc = HashMap::new();
        for (k, vs) in value_choices {
            vc.insert(*k, vs.to_vec());
        }
        self.name_cache.take();
        self.subs.insert(
            name,
            SubcommandEntry {
                help,
                kind: SubcommandKind::Custom {
                    arg_choices: arg_choices.iter().map(|slot| slot.to_vec()).collect(),
                    value_choices: vc,
                    handler: Box::new(handler),
                },
            },
        );
        self
    }

    /// The subcommand table is appended either way.
    pub fn with_long_help(mut self, long: &'static str) -> Self {
        self.long_help = Some(long);
        self
    }

    /// A bare command name runs `handler` instead of a usage error.
    pub fn on_bare<F>(mut self, handler: F) -> Self
    where
        F: FnMut(&mut Ctx) -> anyhow::Result<()> + 'static,
    {
        self.bare = Some(Box::new(handler));
        self
    }

    fn cached_names(&self) -> &[&'static str] {
        self.name_cache
            .get_or_init(|| self.subs.keys().copied().collect())
    }
}

pub fn subcommands<Ctx: 'static>(name: &'static str, help: &'static str) -> SubcommandSet<Ctx> {
    SubcommandSet {
        name,
        help,
        subs: BTreeMap::new(),
        name_cache: std::cell::OnceCell::new(),
        bare: None,
        long_help: None,
    }
}

impl<Ctx: 'static> Command<Ctx> for SubcommandSet<Ctx> {
    fn name(&self) -> &str {
        self.name
    }
    fn help(&self) -> &str {
        self.help
    }

    fn long_help(&self) -> String {
        let preamble = self.long_help.unwrap_or(self.help);
        let mut out = String::with_capacity(128 + preamble.len() + self.subs.len() * 64);
        out.push_str(preamble);
        if !self.subs.is_empty() {
            out.push_str("\nsubcommands:");
            for (name, entry) in &self.subs {
                let kind = match entry.kind {
                    SubcommandKind::Toggle { .. } => "<on|off>",
                    SubcommandKind::Choice { .. } => "<choice>",
                    SubcommandKind::Custom { .. } => "<args...>",
                };
                out.push_str(&format!("\n  {name:14} {kind:9}  {}", entry.help));
            }
        }
        out
    }

    fn arg_choices(&self, arg_index: usize) -> &[&'static str] {
        if arg_index == 0 {
            self.cached_names()
        } else {
            &[]
        }
    }

    fn arg_choices_ctx<'a>(&'a self, arg_index: usize, prior: &[&str]) -> &'a [&'static str] {
        if arg_index == 0 {
            return self.cached_names();
        }
        let Some(&sub_name) = prior.first() else {
            return &[];
        };
        let Some(entry) = self.subs.get(sub_name) else {
            return &[];
        };
        let sub_slot = arg_index - 1;
        match &entry.kind {
            SubcommandKind::Toggle { .. } => &[],
            SubcommandKind::Choice { choices, .. } => {
                if sub_slot == 0 {
                    choices.as_slice()
                } else {
                    &[]
                }
            }
            SubcommandKind::Custom { arg_choices, .. } => arg_choices
                .get(sub_slot)
                .map(|v| v.as_slice())
                .unwrap_or(&[]),
        }
    }

    fn arg_value_choices_ctx<'a>(
        &'a self,
        _arg_index: usize,
        key: &str,
        prior: &[&str],
    ) -> &'a [&'static str] {
        let Some(&sub_name) = prior.first() else {
            return &[];
        };
        let Some(entry) = self.subs.get(sub_name) else {
            return &[];
        };
        match &entry.kind {
            SubcommandKind::Custom { value_choices, .. } => {
                value_choices.get(key).map(|v| v.as_slice()).unwrap_or(&[])
            }
            _ => &[],
        }
    }

    fn run(&mut self, args: &[&str], ctx: &mut Ctx, out: &mut ConsoleWriter) -> anyhow::Result<()> {
        let Some((sub_name, rest)) = args.split_first() else {
            if let Some(handler) = self.bare.as_mut() {
                return handler(ctx);
            }
            let mut msg = format!("usage: {} <subcommand> <value>; subcommands:", self.name);
            for (name, entry) in &self.subs {
                msg.push_str(&format!("\n  {name:12} {}", entry.help));
            }
            return Err(anyhow::anyhow!(msg));
        };
        let Some(entry) = self.subs.get_mut(*sub_name) else {
            let names: Vec<&str> = self.subs.keys().copied().collect();
            return Err(anyhow::anyhow!(
                "unknown subcommand `{sub_name}` for `{}` (try {})",
                self.name,
                names.join(", ")
            ));
        };
        match &mut entry.kind {
            SubcommandKind::Toggle { handler } => {
                let v: Option<bool> = match rest.first() {
                    None => None,
                    Some(value) => match value.to_ascii_lowercase().as_str() {
                        "on" | "true" | "1" => Some(true),
                        "off" | "false" | "0" => Some(false),
                        other => {
                            return Err(anyhow::anyhow!(
                                "unknown value `{other}` for `{} {sub_name}` (try on|off)",
                                self.name
                            ))
                        }
                    },
                };
                let _ = out;
                handler(ctx, v)
            }
            SubcommandKind::Choice { handler, .. } => {
                let value: Option<&str> = rest.first().copied();
                let _ = out;
                handler(ctx, value)
            }
            SubcommandKind::Custom { handler, .. } => handler(ctx, rest, out),
        }
    }
}

pub struct Console<Ctx> {
    commands: BTreeMap<String, Box<dyn Command<Ctx>>>,
    binds: BTreeMap<Key, String>,
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
            binds: BTreeMap::new(),
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
    pub fn register<C: Command<Ctx> + 'static>(&mut self, command: C) {
        let name = command.name().to_string();
        self.commands.insert(name, Box::new(command));
    }

    pub fn has_command(&self, name: &str) -> bool {
        Builtin::from_name(name).is_some() || self.commands.contains_key(name)
    }

    /// No modifiers; fires only while the console is closed. Re-binding overwrites.
    pub fn bind(&mut self, key: Key, command_line: impl Into<String>) {
        self.binds.insert(key, command_line.into());
    }

    pub fn unbind(&mut self, key: Key) {
        self.binds.remove(&key);
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

    /// In [`Key`] order.
    pub fn binds(&self) -> impl Iterator<Item = (Key, &str)> {
        self.binds.iter().map(|(key, line)| (*key, line.as_str()))
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
        // Not via `tracing`: `loam_app::log::ConsoleLayer` would echo it back here.
        #[cfg(target_arch = "wasm32")]
        if ECHO_TO_BROWSER.load(std::sync::atomic::Ordering::Relaxed) {
            web_sys::console::log_1(&line.text.as_str().into());
        }
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
                let prior_refs: Vec<&str> = prior.iter().map(String::as_str).collect();

                if let Some(eq) = prefix.find('=') {
                    let key = &prefix[..eq];
                    let value_prefix = &prefix[eq + 1..];
                    let mut matches: Vec<String> = cmd
                        .arg_value_choices_ctx(*arg_index, key, &prior_refs)
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
                    .arg_choices_ctx(*arg_index, &prior_refs)
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

    /// Built-ins resolve here too, so a queued built-in runs like a typed one.
    pub fn dispatch(&mut self, name: &str, args: &[&str], ctx: &mut Ctx) {
        self.push_history(HistoryLine::input(format!("> {}", render_line(name, args))));
        if let Some(builtin) = Builtin::from_name(name) {
            self.run_builtin(builtin, args.first().copied());
            return;
        }

        let mut writer = ConsoleWriter::new();
        let result = match self.commands.get_mut(name) {
            Some(cmd) => cmd.run(args, ctx, &mut writer),
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
            self.push_history(HistoryLine::error(format!("error: {e:#}")));
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
                        let header_prefix = format!("{}: ", c.name());
                        let body = c.long_help();
                        let indent = " ".repeat(c.name().len() + 2);
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
                    .map(|c| (c.name().to_string(), c.help().to_string()))
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

/// Double quotes honor `\"` and `\\`, single quotes are literal, and an open
/// quote runs to the end of the line.
pub fn parse_line(line: &str) -> Option<(String, Vec<String>)> {
    let mut tokens = tokenize(line);
    if tokens.is_empty() {
        return None;
    }
    let name = tokens.remove(0);
    Some((name, tokens))
}

/// Inverse of [`parse_line`].
pub fn render_line(name: &str, args: &[&str]) -> String {
    let mut line = quote_token(name);
    for arg in args {
        line.push(' ');
        line.push_str(&quote_token(arg));
    }
    line
}

fn quote_token(token: &str) -> String {
    let bare = !token.is_empty()
        && !token
            .chars()
            .any(|c| c.is_whitespace() || c == '"' || c == '\'');
    if bare {
        return token.to_string();
    }
    let mut quoted = String::with_capacity(token.len() + 2);
    quoted.push('"');
    for c in token.chars() {
        if c == '"' || c == '\\' {
            quoted.push('\\');
        }
        quoted.push(c);
    }
    quoted.push('"');
    quoted
}

fn tokenize(line: &str) -> Vec<String> {
    tokenize_spans(line)
        .into_iter()
        .map(|token| token.value)
        .collect()
}

struct Token {
    value: String,
    start: usize,
    end: usize,
}

fn tokenize_spans(line: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut start = None;
    let mut chars = line.char_indices().peekable();
    while let Some((offset, c)) = chars.next() {
        if !c.is_whitespace() {
            start.get_or_insert(offset);
        }
        match c {
            '"' => {
                while let Some(&(_, next)) = chars.peek() {
                    chars.next();
                    if next == '"' {
                        break;
                    }
                    if next == '\\' {
                        if let Some(&(_, escaped)) = chars.peek() {
                            if matches!(escaped, '"' | '\\') {
                                cur.push(escaped);
                                chars.next();
                                continue;
                            }
                        }
                    }
                    cur.push(next);
                }
            }
            '\'' => {
                while let Some(&(_, next)) = chars.peek() {
                    chars.next();
                    if next == '\'' {
                        break;
                    }
                    cur.push(next);
                }
            }
            c if c.is_whitespace() => {
                if let Some(start) = start.take() {
                    out.push(Token {
                        value: std::mem::take(&mut cur),
                        start,
                        end: offset,
                    });
                }
            }
            c => {
                cur.push(c);
            }
        }
    }
    if let Some(start) = start {
        out.push(Token {
            value: cur,
            start,
            end: line.len(),
        });
    }
    out
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

    fn echo_cmd() -> impl Command<Ctx> {
        cmd("echo", "echo args back", |args, _ctx, out| {
            out.line(args.join(" "));
            Ok(())
        })
    }

    fn add_cmd() -> impl Command<Ctx> {
        cmd("add", "add args to ctx", |args, ctx, out| {
            for a in args {
                let n: u32 = a.parse()?;
                *ctx += n;
            }
            out.line(format!("ctx={ctx}"));
            Ok(())
        })
    }

    #[test]
    fn tab_completes_a_command_name_then_each_declared_argument_and_cycles_among_matches() {
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
    fn completion_keeps_quoted_and_multibyte_input_intact() {
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
    fn builtin_help_describes_one_command() {
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
    fn the_prompt_recalls_submitted_lines_newest_first_without_consecutive_repeats() {
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
    fn a_dispatched_line_is_echoed_once_ahead_of_its_output_and_a_failure_is_an_error_line() {
        let mut c = Console::<Ctx>::new();
        c.register(echo_cmd());
        c.register(cmd("fail", "always fails", |_, _, _| anyhow::bail!("nope")));
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
    fn submit_runs_the_typed_line_through_its_handler_and_a_blank_prompt_does_nothing() {
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
    fn a_typed_line_splits_on_spaces_outside_quotes_and_a_rendered_line_reparses_to_its_invocation()
    {
        let cases: [(&str, &[&str]); 5] = [
            (r#"foo "bar baz" qux"#, &["foo", "bar baz", "qux"]),
            (r#"a "he said \"hi\"" b"#, &["a", r#"he said "hi""#, "b"]),
            (r#""back\\slash""#, &[r"back\slash"]),
            (r"'a \n b'", &[r"a \n b"]),
            (r#"foo "unterminated"#, &["foo", "unterminated"]),
        ];
        for (line, tokens) in cases {
            assert_eq!(tokenize(line), tokens, "`{line}`");
        }

        let cases: [(&str, &[&str]); 7] = [
            ("clear", &[]),
            ("echo", &["a", "b"]),
            ("load", &["5 cell", "fast"]),
            ("mark", &["#ff8800"]),
            ("say", &[r#"a "quoted" word"#]),
            ("say", &["it's"]),
            ("say", &["", "back\\slash"]),
        ];
        for (name, args) in cases {
            let rendered = render_line(name, args);
            let (back_name, back_args) = parse_line(&rendered)
                .unwrap_or_else(|| panic!("`{rendered}` tokenizes to nothing"));
            let back_args: Vec<&str> = back_args.iter().map(String::as_str).collect();
            assert_eq!(back_name, name, "`{rendered}`");
            assert_eq!(back_args, args, "`{rendered}`");
        }
    }

    #[test]
    fn builtins_run_on_the_typed_frame_and_registry_commands_wait() {
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
