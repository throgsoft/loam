use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct Args {
    map: HashMap<String, String>,
    bare_flags: Vec<String>,
    bare_values: HashMap<String, String>,
}

impl Args {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn current() -> Self {
        Self::from_argv(std::env::args().skip(1))
    }

    #[cfg(target_arch = "wasm32")]
    pub fn current() -> Self {
        let mut map = HashMap::new();
        if let Some(window) = web_sys::window() {
            if let Ok(search) = window.location().search() {
                parse_query_into(&search, &mut map);
            }
            if let Ok(hash) = window.location().hash() {
                parse_query_into(&hash, &mut map);
            }
        }
        QUERY_OVERRIDE.with(|q| {
            if let Some((search, hash)) = q.borrow().as_ref() {
                parse_query_into(search, &mut map);
                parse_query_into(hash, &mut map);
            }
        });
        Self {
            map,
            bare_flags: Vec::new(),
            bare_values: HashMap::new(),
        }
    }

    pub fn from_argv<I, S>(argv: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut map = HashMap::new();
        let mut bare_flags = Vec::new();
        let mut bare_values = HashMap::new();
        let mut awaiting: Option<String> = None;
        for arg in argv {
            if arg.as_ref() == "--" {
                break;
            }
            let Some(stripped) = arg.as_ref().strip_prefix("--") else {
                if let Some(flag) = awaiting.take() {
                    bare_values.insert(flag, arg.as_ref().to_string());
                }
                continue;
            };
            awaiting = None;
            match stripped.split_once('=') {
                Some((k, v)) if !k.is_empty() => {
                    map.insert(k.to_string(), v.to_string());
                }
                None if !stripped.is_empty() => {
                    bare_flags.push(stripped.to_string());
                    awaiting = Some(stripped.to_string());
                }
                _ => {}
            }
        }
        Self {
            map,
            bare_flags,
            bare_values,
        }
    }

    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self {
            map: pairs
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
            bare_flags: Vec::new(),
            bare_values: HashMap::new(),
        }
    }

    pub fn has_bare_flag(&self, key: &str) -> bool {
        self.bare_flags.iter().any(|flag| flag == key)
    }

    pub fn bare_flag_value(&self, key: &str) -> Option<&str> {
        self.bare_values.get(key).map(String::as_str)
    }

    pub fn flag_value(&self, key: &str) -> Option<&str> {
        self.bare_flag_value(key).or_else(|| self.get(key))
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.map.get(key).map(String::as_str)
    }

    pub fn parse<T: std::str::FromStr>(&self, key: &str) -> Option<T> {
        self.get(key)?.parse().ok()
    }

    pub fn get_many<'a>(&'a self, key: &str) -> Vec<&'a str> {
        match self.get(key) {
            Some(v) => v.split(',').filter(|s| !s.is_empty()).collect(),
            None => Vec::new(),
        }
    }

    /// Order is HashMap-arbitrary.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.map.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    // Workers have no `window.location`; the init message forwards the query.
    static QUERY_OVERRIDE: std::cell::RefCell<Option<(String, String)>> =
        const { std::cell::RefCell::new(None) };
}

/// Hash wins on collision, matching the main-thread parse order.
#[cfg(target_arch = "wasm32")]
pub fn set_query_override(search: String, hash: String) {
    QUERY_OVERRIDE.with(|q| *q.borrow_mut() = Some((search, hash)));
}

#[cfg(any(target_arch = "wasm32", test))]
fn parse_query_into(raw: &str, map: &mut HashMap<String, String>) {
    let trimmed = raw.trim_start_matches(['?', '#']);
    if trimmed.is_empty() {
        return;
    }
    for pair in trimmed.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if !k.is_empty() {
                let value = v.replace('+', " ");
                map.insert(k.to_string(), value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_arguments_parse_from_a_native_command_line_and_from_a_page_query_and_fragment() {
        let args = Args::from_argv([
            "--seed=42",
            "sub",
            "--=x",
            "--state=a=b",
            "--shapes",
            "5-cell,8-cell",
            "--",
            "--late=1",
        ]);
        assert_eq!(args.get("seed"), Some("42"));
        assert_eq!(args.get("state"), Some("a=b"));
        assert_eq!(args.get("sub"), None);
        assert_eq!(args.get(""), None);
        assert_eq!(args.get("late"), None);

        assert_eq!(args.get("shapes"), None);
        assert!(args.has_bare_flag("shapes"));
        assert_eq!(args.bare_flag_value("shapes"), Some("5-cell,8-cell"));
        assert_eq!(args.flag_value("shapes"), Some("5-cell,8-cell"));
        assert!(!args.has_bare_flag("seed"));
        assert_eq!(args.bare_flag_value("seed"), None);
        assert_eq!(args.flag_value("seed"), Some("42"));
        assert_eq!(args.flag_value("headless"), None);

        let mut map = HashMap::new();
        for fragment in [
            "?title=hello+world&flag&=1&empty=&state=a=b&a=1&a=search",
            "#a=hash",
            "?",
            "",
        ] {
            parse_query_into(fragment, &mut map);
        }
        let mut pairs: Vec<(&str, &str)> =
            map.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        pairs.sort();
        assert_eq!(
            pairs,
            [
                ("a", "hash"),
                ("empty", ""),
                ("state", "a=b"),
                ("title", "hello world"),
            ]
        );
    }
}
