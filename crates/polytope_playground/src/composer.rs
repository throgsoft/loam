use std::fmt::Write as _;

use loam_math::{Bivector4, Plane4};

use crate::consts::BASE_ROTATION_RATE;

pub(crate) const MAX_TERMS: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Term {
    pub(crate) planes: [u8; 6],
    pub(crate) scalar: Option<f32>,
}

impl Term {
    pub(crate) fn is_empty(&self) -> bool {
        self.planes.iter().all(|count| *count == 0)
    }

    fn bivector(&self) -> Bivector4 {
        let phi = self.scalar.unwrap_or(1.0);
        Plane4::ALL
            .into_iter()
            .enumerate()
            .fold(Bivector4::ZERO, |sum, (index, plane)| {
                sum + plane.unit_bivector() * (phi * f32::from(self.planes[index]))
            })
    }

    pub(crate) fn write(&self, out: &mut String) {
        let multi: u32 = self.planes.iter().map(|count| u32::from(*count)).sum();
        if let Some(phi) = self.scalar {
            let _ = write!(out, "{:.0}deg ", phi.to_degrees());
        }
        if multi > 1 {
            out.push('(');
        }
        let mut first = true;
        for (index, plane) in Plane4::ALL.into_iter().enumerate() {
            for _ in 0..self.planes[index] {
                if !first {
                    out.push_str(" + ");
                }
                out.push_str(plane.label());
                first = false;
            }
        }
        if multi > 1 {
            out.push(')');
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Composer {
    terms: [Term; MAX_TERMS],
    len: usize,
    pub(crate) draft: [u8; 6],
    pub(crate) scrub: f32,
}

impl Composer {
    pub(crate) fn terms(&self) -> &[Term] {
        &self.terms[..self.len]
    }

    pub(crate) fn push(&mut self, term: Term) -> bool {
        if term.is_empty() || self.len == MAX_TERMS {
            return false;
        }
        self.terms[self.len] = term;
        self.len += 1;
        true
    }

    pub(crate) fn remove(&mut self, index: usize) -> bool {
        if index >= self.len {
            return false;
        }
        self.terms.copy_within(index + 1..self.len, index);
        self.len -= 1;
        self.terms[self.len] = Term::default();
        true
    }

    pub(crate) fn clear(&mut self) {
        *self = Composer::default();
    }

    pub(crate) fn commit_draft(&mut self) -> bool {
        let term = Term {
            planes: self.draft,
            scalar: None,
        };
        let pushed = self.push(term);
        if pushed {
            self.draft = [0; 6];
        }
        pushed
    }

    pub(crate) fn omega(&self) -> Bivector4 {
        self.terms()
            .iter()
            .fold(Bivector4::ZERO, |sum, term| sum + term.bivector())
    }

    pub(crate) fn angular_velocity(&self) -> Bivector4 {
        self.omega() * BASE_ROTATION_RATE
    }

    pub(crate) fn axis(&self) -> Option<Bivector4> {
        let omega = self.omega();
        let magnitude_squared = omega.magnitude_squared();
        (magnitude_squared > 1e-12).then(|| omega * (1.0 / magnitude_squared.sqrt()))
    }

    pub(crate) fn write(&self, out: &mut String) {
        for (index, term) in self.terms().iter().enumerate() {
            if index > 0 {
                out.push_str(" . ");
            }
            term.write(out);
        }
    }
}

pub(crate) fn parse_term(input: &str) -> Result<Term, String> {
    let normalized = input
        .trim()
        .replace('\u{b7}', "*")
        .replace('\u{b0}', "deg ");
    let text = normalized.trim();
    if text.is_empty() {
        return Err("empty input".into());
    }
    let (scalar, rest) = peel_scalar(text)?;
    let bivector = rest.trim();
    let inner = match bivector.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        Some(inner) => inner.trim(),
        None => bivector,
    };
    if inner.is_empty() {
        return Err("missing bivector after scalar".into());
    }
    let mut planes = [0u8; 6];
    for part in inner.split('+') {
        let token = part.trim();
        if token.is_empty() {
            return Err("empty plane between '+'".into());
        }
        let plane = parse_plane(token)?;
        let slot = &mut planes[plane as usize];
        *slot = slot
            .checked_add(1)
            .ok_or_else(|| format!("too many copies of `{token}`"))?;
    }
    Ok(Term { planes, scalar })
}

fn peel_scalar(text: &str) -> Result<(Option<f32>, &str), String> {
    let bytes = text.as_bytes();
    let mut at = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    let digits = at;
    while at < bytes.len() && (bytes[at].is_ascii_digit() || bytes[at] == b'.') {
        at += 1;
    }
    if at == digits {
        return Ok((None, text));
    }
    let number = &text[..at];
    let value: f32 = number
        .parse()
        .map_err(|_| format!("not a number: `{number}`"))?;
    if !value.is_finite() {
        return Err("angle must be finite".into());
    }
    let mut tail = text[at..].trim_start();
    let radians = if let Some(rest) = tail.strip_prefix("rad") {
        tail = rest.trim_start();
        value
    } else {
        if let Some(rest) = tail.strip_prefix("deg") {
            tail = rest.trim_start();
        }
        value.to_radians()
    };
    if let Some(rest) = tail.strip_prefix('*') {
        tail = rest.trim_start();
    }
    Ok((Some(radians), tail))
}

fn parse_plane(token: &str) -> Result<Plane4, String> {
    Plane4::ALL
        .into_iter()
        .find(|plane| plane.label() == token)
        .ok_or_else(|| format!("unknown plane `{token}` (expected xy/xz/xw/yz/yw/zw)"))
}

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_PI_2;

    use super::*;

    #[test]
    fn the_summed_bivector_puts_each_terms_angle_on_the_plane_its_index_names() {
        let mut composer = Composer::default();
        assert!(composer.push(parse_term("90deg (xy + zw)").expect("the formula parses")));
        assert!(composer.push(parse_term("xw").expect("the formula parses")));

        let omega = composer.omega();
        let expected = Plane4::Xy.unit_bivector() * FRAC_PI_2
            + Plane4::Zw.unit_bivector() * FRAC_PI_2
            + Plane4::Xw.unit_bivector();
        for plane in Plane4::ALL {
            assert!(
                (omega.component(plane) - expected.component(plane)).abs() < 1e-6,
                "the sum put {} on the {} plane, not {}",
                omega.component(plane),
                plane.label(),
                expected.component(plane)
            );
        }
    }

    #[test]
    fn the_scrub_axis_is_the_unit_bivector_of_the_sum() {
        let mut composer = Composer::default();
        composer.push(parse_term("90deg (xy + zw)").expect("the formula parses"));
        let axis = composer
            .axis()
            .expect("a non-degenerate sequence has an axis");
        let half = std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (axis.component(Plane4::Xy) - half).abs() < 1e-6
                && (axis.component(Plane4::Zw) - half).abs() < 1e-6,
            "the axis is {axis:?}, not the normalized xy + zw sum"
        );
        assert!(Composer::default().axis().is_none());
    }

    #[test]
    fn formula_units_and_grouping_agree() {
        let degrees = parse_term("90deg * (xy + zw)").expect("degrees parse");
        let radians = parse_term("1.5707963rad (xy + zw)").expect("radians parse");
        assert_eq!(degrees.planes, [1, 0, 0, 0, 0, 1]);
        assert!(
            (degrees.scalar.expect("an angle") - radians.scalar.expect("an angle")).abs() < 1e-6
        );
        assert_eq!(parse_term("xw").expect("a bare plane").scalar, None);
    }

    #[test]
    fn an_invalid_formula_never_becomes_a_term() {
        for input in [
            "",
            "90",
            "90 ()",
            "xy +",
            "xx",
            "3..4 xy",
            "NaN xy",
            "999999999999999999999999999999999999999999 xy",
        ] {
            assert!(parse_term(input).is_err(), "`{input}` parsed");
        }
    }

    #[test]
    fn removing_a_term_keeps_the_order_of_the_rest() {
        let mut composer = Composer::default();
        for label in ["xy", "xz", "xw"] {
            composer.push(parse_term(label).expect("a bare plane"));
        }
        assert!(composer.remove(1));
        let mut text = String::new();
        composer.write(&mut text);
        assert_eq!(text, "xy . xw");
        assert!(!composer.remove(2));
    }
}
