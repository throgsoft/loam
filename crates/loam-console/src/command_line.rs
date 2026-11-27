#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandLine {
    pub name: String,
    pub args: Vec<String>,
}

impl CommandLine {
    pub fn parse(line: &str) -> Option<Self> {
        let (name, args) = parse_line(line)?;
        Some(Self { name, args })
    }

    pub fn arg_refs(&self) -> Vec<&str> {
        self.args.iter().map(String::as_str).collect()
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

pub(crate) fn quote_token(token: &str) -> String {
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

pub(crate) struct Token {
    pub(crate) value: String,
    pub(crate) start: usize,
    pub(crate) end: usize,
}

pub(crate) fn tokenize_spans(line: &str) -> Vec<Token> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_and_render_round_trip() {
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
}
