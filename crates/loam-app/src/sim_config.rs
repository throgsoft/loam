use loam_time::FixedTimestep;

pub const DEFAULT_MAX_TICKS_PER_FRAME: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimConfig {
    pub fixed_hz: u32,
    pub catch_up: CatchUp,
    pub seed: u64,
    pub overlap: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatchUp {
    /// Ticks beyond this many per frame are dropped; `0` stops the sim.
    Cap(u32),
    /// Every owed tick runs, whatever the frame cost.
    All,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            fixed_hz: 60,
            catch_up: CatchUp::Cap(DEFAULT_MAX_TICKS_PER_FRAME),
            seed: 0,
            overlap: false,
        }
    }
}

impl SimConfig {
    pub fn timestep(&self) -> FixedTimestep {
        let cap = match self.catch_up {
            CatchUp::Cap(n) => n,
            CatchUp::All => u32::MAX,
        };
        FixedTimestep::new(self.fixed_hz).with_max_catch_up(cap)
    }

    #[cfg(any(target_arch = "wasm32", test))]
    pub(crate) fn encode(&self, mut set: impl FnMut(&str, f64)) {
        set("fixed_hz", f64::from(self.fixed_hz));
        let catch_up = match self.catch_up {
            CatchUp::Cap(n) => f64::from(n),
            CatchUp::All => -1.0,
        };
        set("catch_up", catch_up);
        // A JS number holds 53 bits, so the seed crosses as two u32 halves.
        set("seed_lo", f64::from(self.seed as u32));
        set("seed_hi", f64::from((self.seed >> 32) as u32));
        set("overlap", f64::from(u8::from(self.overlap)));
    }

    #[cfg(test)]
    fn decode(get: impl Fn(&str) -> Option<f64>) -> anyhow::Result<Self> {
        let field = |key: &str| {
            get(key).ok_or_else(|| anyhow::anyhow!("init message has no `{key}` field"))
        };
        let catch_up = match field("catch_up")? {
            all if all < 0.0 => CatchUp::All,
            cap => CatchUp::Cap(cap as u32),
        };
        let seed_lo = u64::from(field("seed_lo")? as u32);
        let seed_hi = u64::from(field("seed_hi")? as u32);
        Ok(Self {
            fixed_hz: field("fixed_hz")? as u32,
            catch_up,
            seed: seed_hi << 32 | seed_lo,
            overlap: field("overlap")? != 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn decoded_sim_config_sets_the_worker_rate_and_cap() {
        let capped = SimConfig {
            fixed_hz: 144,
            catch_up: CatchUp::Cap(3),
            seed: 0x1234_5678_9abc_def0,
            overlap: true,
        };
        let uncapped = SimConfig {
            catch_up: CatchUp::All,
            ..capped
        };
        for sent in [capped, uncapped] {
            let mut fields = Vec::new();
            sent.encode(|key, value| fields.push((key.to_string(), value)));
            let received =
                SimConfig::decode(|key| fields.iter().find(|(k, _)| k == key).map(|(_, v)| *v))
                    .unwrap();
            assert_eq!(received, sent);
        }

        let mut timestep = capped.timestep();
        assert_eq!(timestep.dt(), Duration::from_nanos(1_000_000_000 / 144));
        let base = web_time::Instant::now();
        timestep.advance(base);
        let ticks = timestep.advance(base + timestep.dt() * 10);
        assert_eq!(ticks.end - ticks.start, 3);
    }
}
