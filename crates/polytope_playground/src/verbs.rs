use crate::projections::{apply_projection_selection_defaults, WireframeProjection};
use anyhow::anyhow;
use loam_egui::SubcommandSet;

pub(crate) const DEFAULT_WIREFRAME_WIDTH_PX: f32 = 1.8;

const MAX_WIREFRAME_WIDTH_PX: f32 = 16.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct WireframeControls {
    pub(crate) enabled: bool,
    pub(crate) width_px: f32,
    pub(crate) alpha: f32,
    pub(crate) projection: WireframeProjection,
}

impl Default for WireframeControls {
    fn default() -> Self {
        Self {
            enabled: false,
            width_px: DEFAULT_WIREFRAME_WIDTH_PX,
            alpha: 1.0,
            projection: WireframeProjection::default(),
        }
    }
}

pub(crate) fn wireframe_subcommands<Ctx: 'static>(
    reach: fn(&mut Ctx) -> &mut WireframeControls,
) -> SubcommandSet<Ctx> {
    loam_egui::subcommands::<Ctx>("wireframe", "4D hull wireframe overlay")
        .with_long_help(
            "Draws hull edges with depth shading relative to the current slice.",
        )
        .on_bare(move |ctx| {
            let w = reach(ctx);
            w.enabled = !w.enabled;
            Ok(())
        })
        .custom(
            "width",
            "edge thickness in pixels (bare reads)",
            &[&[]],
            &[],
            move |ctx, args, out| {
                let w = reach(ctx);
                match args.first().copied() {
                    None => out.line(format!("wireframe width: {:.2} px", w.width_px)),
                    Some(token) => {
                        let px: f32 = token
                            .parse()
                            .map_err(|e| anyhow!("invalid width `{token}`: {e}"))?;
                        if !(px > 0.0 && px <= MAX_WIREFRAME_WIDTH_PX) {
                            return Err(anyhow!(
                                "wireframe width {px} out of range; expected (0, {MAX_WIREFRAME_WIDTH_PX}]"
                            ));
                        }
                        w.width_px = px;
                        out.line(format!("wireframe width: set to {px:.2} px"));
                    }
                }
                Ok(())
            },
        )
        .custom(
            "alpha",
            "uniform edge alpha (bare reads)",
            &[&[]],
            &[],
            move |ctx, args, out| {
                let w = reach(ctx);
                match args.first().copied() {
                    None => out.line(format!("wireframe alpha: {:.3}", w.alpha)),
                    Some(token) => {
                        let a: f32 = token
                            .parse()
                            .map_err(|e| anyhow!("invalid alpha `{token}`: {e}"))?;
                        if !(a > 0.0 && a <= 1.0) {
                            return Err(anyhow!(
                                "wireframe alpha {a} out of range; expected (0, 1]"
                            ));
                        }
                        w.alpha = a;
                        out.line(format!("wireframe alpha: set to {a:.3}"));
                    }
                }
                Ok(())
            },
        )
        .custom(
            "perspective",
            "4D->R³ projection (bare cycles): shadow | w-pinhole | stereographic | hyperslice",
            &[&WireframeProjection::TOKENS],
            &[],
            move |ctx, args, out| {
                let w = reach(ctx);
                w.projection = match args.first().copied() {
                    None => {
                        let all = WireframeProjection::ALL;
                        let i = all
                            .iter()
                            .position(|p| p.same_variant(w.projection))
                            .unwrap_or(0);
                        all[(i + 1) % all.len()]
                    }
                    Some(token) => WireframeProjection::from_token(token).ok_or_else(|| {
                        anyhow!(
                            "unknown projection `{token}` (try {})",
                            WireframeProjection::TOKENS.join("|")
                        )
                    })?,
                };
                apply_projection_selection_defaults(w.projection, &mut w.enabled);
                out.line(format!(
                    "wireframe perspective: {}",
                    w.projection.label().to_lowercase()
                ));
                Ok(())
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_egui::Console;

    #[derive(Default)]
    struct Scene {
        wireframe: WireframeControls,
    }

    fn console() -> Console<Scene> {
        let mut c = Console::<Scene>::new();
        c.register(wireframe_subcommands::<Scene>(|s| &mut s.wireframe));
        c
    }

    #[test]
    fn invalid_wireframe_values_preserve_state() {
        let (mut c, mut scene) = (console(), Scene::default());
        for args in [
            ["width", "0"].as_slice(),
            &["width", "17"],
            &["alpha", "0"],
            &["alpha", "1.5"],
            &["width", "wide"],
            &["width", "NaN"],
            &["width", "inf"],
            &["alpha", "NaN"],
            &["alpha", "inf"],
        ] {
            c.dispatch("wireframe", args, &mut scene);
        }
        assert_eq!(scene.wireframe, WireframeControls::default());
    }
}
