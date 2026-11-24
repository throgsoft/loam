use anyhow::{anyhow, Result};
use loam_render::sky_ground::{
    DEFAULT_FOG_PER_UNIT, DEFAULT_SKY, GROUND_DARK_GREY, GROUND_LIGHT_GREY,
};
use loam_render::{Ground, Sky};
use loam_runtime::Backdrop;

// Above this the checker blends into the sky inside one body length.
const MAX_FOG_PER_UNIT: f32 = 1.0;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Environment {
    pub sky: Sky,
    pub dark: [f32; 3],
    pub light: [f32; 3],
    pub fog_per_unit: f32,
    pub fog_start: f32,
    pub floor_visible: bool,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            sky: DEFAULT_SKY,
            dark: GROUND_DARK_GREY,
            light: GROUND_LIGHT_GREY,
            fog_per_unit: DEFAULT_FOG_PER_UNIT,
            fog_start: 0.0,
            floor_visible: true,
        }
    }
}

impl Environment {
    pub fn ground(&self, y: f32, visible: bool) -> Ground {
        Ground {
            y,
            dark: self.dark,
            light: self.light,
            fog_per_unit: self.fog_per_unit,
            fog_start: self.fog_start,
            visible,
        }
    }

    pub fn backdrop(&self, y: f32) -> Backdrop {
        Backdrop {
            sky: self.sky,
            ground: self.ground(y, self.floor_visible),
        }
    }

    pub fn of(backdrop: &Backdrop) -> Self {
        let ground = backdrop.ground;
        Self {
            sky: backdrop.sky,
            dark: ground.dark,
            light: ground.light,
            fog_per_unit: ground.fog_per_unit,
            fog_start: ground.fog_start,
            floor_visible: ground.visible,
        }
    }

    pub fn report(&self, field: Option<&str>) -> String {
        match field {
            Some("dark") => format!("ground dark: {}", rgb_text(self.dark)),
            Some("light") => format!("ground light: {}", rgb_text(self.light)),
            Some("fog") => format!(
                "ground fog: {:.4} per unit (half sky at {:.1} units)",
                self.fog_per_unit,
                half_blend_distance(self.fog_per_unit)
            ),
            _ => format!(
                "ground dark {} light {} fog {:.4} (half sky at {:.1} units)",
                rgb_text(self.dark),
                rgb_text(self.light),
                self.fog_per_unit,
                half_blend_distance(self.fog_per_unit)
            ),
        }
    }

    pub fn apply(&mut self, args: &[&str]) -> Result<String> {
        let Some(field) = args.first().copied() else {
            return Ok(self.report(None));
        };
        match field {
            "reset" => {
                *self = Self {
                    sky: self.sky,
                    floor_visible: self.floor_visible,
                    ..Self::default()
                };
                Ok(format!("ground: reset ({})", self.report(None)))
            }
            "dark" | "light" => {
                if args.len() == 1 {
                    return Ok(self.report(Some(field)));
                }
                let rgb = parse_rgb(field, &args[1..])?;
                if field == "dark" {
                    self.dark = rgb;
                } else {
                    self.light = rgb;
                }
                Ok(format!("ground {field}: set to {}", rgb_text(rgb)))
            }
            "fog" => {
                if args.len() == 1 {
                    return Ok(self.report(Some(field)));
                }
                self.fog_per_unit = parse_fog(args[1])?;
                Ok(format!("ground fog: set to {}", self.report(Some("fog"))))
            }
            other => Err(anyhow!(
                "ground: unknown field `{other}` (try dark|light|fog|reset)"
            )),
        }
    }
}

pub fn half_blend_distance(fog_per_unit: f32) -> f32 {
    if fog_per_unit <= 0.0 {
        return f32::INFINITY;
    }
    std::f32::consts::LN_2 / fog_per_unit
}

fn rgb_text(rgb: [f32; 3]) -> String {
    format!("{:.3} {:.3} {:.3}", rgb[0], rgb[1], rgb[2])
}

fn parse_rgb(field: &str, args: &[&str]) -> Result<[f32; 3]> {
    let [r, g, b] = args else {
        return Err(anyhow!(
            "usage: ground {field} <r> <g> <b>, each a float in [0, 1]"
        ));
    };
    let mut rgb = [0.0_f32; 3];
    for (slot, token) in rgb.iter_mut().zip([r, g, b]) {
        let value: f32 = token
            .parse()
            .map_err(|e| anyhow!("ground {field}: invalid channel `{token}`: {e}"))?;
        if !(value.is_finite() && (0.0..=1.0).contains(&value)) {
            return Err(anyhow!(
                "ground {field}: channel {value} out of range; expected [0, 1]"
            ));
        }
        *slot = value;
    }
    Ok(rgb)
}

fn parse_fog(token: &str) -> Result<f32> {
    let value: f32 = token
        .parse()
        .map_err(|e| anyhow!("ground fog: invalid density `{token}`: {e}"))?;
    if !(value.is_finite() && (0.0..=MAX_FOG_PER_UNIT).contains(&value)) {
        return Err(anyhow!(
            "ground fog: density {value} out of range; expected [0, {MAX_FOG_PER_UNIT}]"
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_parse_rejects_out_of_range_and_malformed_arguments() {
        let mut env = Environment::default();
        for line in [
            vec!["fog", "-0.1"],
            vec!["fog", "2.0"],
            vec!["fog", "nan"],
            vec!["fog", "wide"],
            vec!["dark", "0.5"],
            vec!["dark", "0.5", "0.5"],
            vec!["dark", "0.5", "0.5", "0.5", "0.5"],
            vec!["dark", "0.5", "0.5", "1.5"],
            vec!["dark", "0.5", "0.5", "red"],
            vec!["sky"],
        ] {
            assert!(
                env.apply(&line).is_err(),
                "`ground {}` was accepted",
                line.join(" ")
            );
        }
        assert_eq!(env, Environment::default(), "a rejected line still wrote");
    }

    #[test]
    fn ground_reset_keeps_a_page_set_sky() {
        let sky = Sky {
            below: [0.9, 0.8, 0.7],
            above: [0.6, 0.5, 0.4],
        };
        let mut env = Environment {
            sky,
            ..Environment::default()
        };
        env.apply(&["fog", "0.5"]).expect("set");
        env.apply(&["reset"]).expect("reset");
        assert_eq!(env.fog_per_unit, DEFAULT_FOG_PER_UNIT);
        assert_eq!(env.sky, sky, "the ground verb wiped the sky");
    }
}
