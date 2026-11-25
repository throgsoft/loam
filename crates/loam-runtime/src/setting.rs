use crate::command::{Command, Outcome};

/// The accessor must reach the value through [`crate::Value::get_mut`] so the change bumps the version readers watch.
pub struct Setting<A, T: 'static> {
    pub name: &'static str,
    choices: &'static [(&'static str, T)],
    at: fn(&mut A) -> &mut T,
}

impl<A, T: 'static> Clone for Setting<A, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A, T: 'static> Copy for Setting<A, T> {}

impl<A> Setting<A, bool> {
    pub const fn flag(name: &'static str, at: fn(&mut A) -> &mut bool) -> Self {
        Self::choice(name, &[("off", false), ("on", true)], at)
    }
}

impl<A, T: 'static> Setting<A, T> {
    /// Panics when `choices` is empty.
    pub const fn choice(
        name: &'static str,
        choices: &'static [(&'static str, T)],
        at: fn(&mut A) -> &mut T,
    ) -> Self {
        assert!(!choices.is_empty(), "a setting has at least one choice");
        Self { name, choices, at }
    }

    pub const fn choices(&self) -> &'static [(&'static str, T)] {
        self.choices
    }
}

impl<A, T: Copy + 'static> Setting<A, T> {
    pub fn parse(&self, token: &str) -> Option<T> {
        self.choices
            .iter()
            .find(|(name, _)| *name == token)
            .map(|&(_, value)| value)
    }
}

impl<A: 'static, T: Copy + PartialEq + Send + Sync + 'static> Setting<A, T> {
    pub fn set(self, value: T) -> Command<A> {
        Command::try_app_fn(self.name, move |dispatch| {
            *(self.at)(dispatch.app) = value;
            Ok(Outcome::Done)
        })
    }

    /// Writes the choice after the held one, wrapping, or the first choice when the held value has no entry.
    pub fn toggle(self) -> Command<A> {
        Command::try_app_fn(self.name, move |dispatch| {
            let held = (self.at)(dispatch.app);
            let next = self
                .choices
                .iter()
                .position(|(_, value)| value == held)
                .map_or(0, |index| (index + 1) % self.choices.len());
            *held = self
                .choices
                .get(next)
                .expect("a setting has at least one choice")
                .1;
            Ok(Outcome::Done)
        })
    }
}
