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
