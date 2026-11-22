/// Tokenized with `crate::parse_line`, so queue and console share one grammar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandLine {
    pub name: String,
    pub args: Vec<String>,
}

impl CommandLine {
    pub fn parse(line: &str) -> Option<Self> {
        let (name, args) = crate::parse_line(line)?;
        Some(Self { name, args })
    }

    pub fn arg_refs(&self) -> Vec<&str> {
        self.args.iter().map(String::as_str).collect()
    }
}

#[derive(Debug, Default)]
pub struct CommandQueue {
    pending: Vec<CommandLine>,
}

impl CommandQueue {
    pub const fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    pub fn submit(&mut self, command: CommandLine) {
        self.pending.push(command);
    }

    pub fn submit_line(&mut self, line: &str) -> bool {
        match CommandLine::parse(line) {
            Some(command) => {
                self.submit(command);
                true
            }
            None => false,
        }
    }

    pub fn drain(&mut self) -> std::vec::Drain<'_, CommandLine> {
        self.pending.drain(..)
    }
}
