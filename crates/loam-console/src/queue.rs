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

    /// Draining retains the allocation for the next batch.
    pub fn drain(&mut self) -> std::vec::Drain<'_, CommandLine> {
        self.pending.drain(..)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queues_preserve_order_without_sharing_state() {
        let mut first = CommandQueue::new();
        let mut second = CommandQueue::new();
        first.submit_line("alpha 1");
        second.submit_line("other");
        first.submit_line("beta");
        assert_eq!(
            first
                .drain()
                .map(|command| command.name)
                .collect::<Vec<_>>(),
            ["alpha", "beta"]
        );
        assert_eq!(second.drain().next().unwrap().name, "other");
        assert!(first.drain().next().is_none());
    }

    #[test]
    fn submitted_arguments_keep_console_quoting() {
        let mut queue = CommandQueue::new();
        assert!(queue.submit_line(r#"load "5 cell" fast"#));
        assert!(!queue.submit_line("  "));
        let commands: Vec<_> = queue.drain().collect();
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "load");
        assert_eq!(commands[0].args, ["5 cell", "fast"]);
    }
}
