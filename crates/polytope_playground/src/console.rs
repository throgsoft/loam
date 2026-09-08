use crate::*;

impl RotateScene {
    pub(crate) fn build_console(
        runtime: &loam_app::Runtime,
        control: &loam_app::shell::SceneControl,
    ) -> Console<Demo> {
        let mut c = Console::<Demo>::new();
        loam_app::shell::register_shell_commands::<Demo, shell::Playground>(
            &mut c,
            loam_app::build_info!(),
            runtime,
            control,
        );
        c.register(loam_egui::cmd(
            "reset",
            "reset slice, rate, active set, orientation and time in place",
            |_args, demo: &mut Demo, _out| {
                demo.reset();
                Ok(())
            },
        ));
        c.register(loam_egui::cmd(
            "spin",
            "toggle continuous rotation (Space / T)",
            |_args, demo: &mut Demo, _out| {
                demo.rotate = !demo.rotate;
                Ok(())
            },
        ));
        c.register(loam_egui::cmd(
            "hud",
            "toggle the top-left loam-text state readout (w, t, rate, planes)",
            |_args, demo: &mut Demo, out| {
                demo.show_text_hud = !demo.show_text_hud;
                out.line(format!(
                    "hud: {}",
                    if demo.show_text_hud { "on" } else { "off" }
                ));
                Ok(())
            },
        ));
        c.register(loam_egui::cmd(
            "seek",
            "set rotation time in seconds",
            |args, demo: &mut Demo, _out| {
                let [seconds] = args else {
                    return Err(anyhow!("usage: seek <seconds>"));
                };
                let seconds: f32 = seconds.parse()?;
                if !(0.0..=1.0e6).contains(&seconds) {
                    return Err(anyhow!("time must be between 0 and 1000000 seconds"));
                }
                demo.rot_time = seconds;
                demo.t_slider_max = demo.t_slider_max.max(seconds);
                demo.recompose_spins_at(seconds);
                demo.rebuild_bodies();
                Ok(())
            },
        ));
        c.register(loam_egui::cmd(
            "rate",
            "set rotation speed from 0.25 to 4",
            |args, demo: &mut Demo, _out| {
                let [rate] = args else {
                    return Err(anyhow!("usage: rate <multiplier>"));
                };
                let rate: f32 = rate.parse()?;
                if !(0.25..=4.0).contains(&rate) {
                    return Err(anyhow!("rate must be between 0.25 and 4"));
                }
                demo.rate_scale = rate;
                Ok(())
            },
        ));
        c.register(loam_egui::cmd(
            "slice",
            "set the w cross-section",
            |args, demo: &mut Demo, _out| {
                let [slice] = args else {
                    return Err(anyhow!("usage: slice <w>"));
                };
                let slice: f32 = slice.parse()?;
                let range = demo.effective_w_range();
                if !(-range..=range).contains(&slice) {
                    return Err(anyhow!("slice must be between {} and {range}", -range));
                }
                demo.w_slice = slice;
                Ok(())
            },
        ));
        c.register(loam_egui::cmd(
            "controls",
            "toggle the bottom controls overlay (H)",
            |_args, demo: &mut Demo, _out| {
                demo.show_controls = !demo.show_controls;
                Ok(())
            },
        ));
        c.register(loam_egui::cmd(
            "formula",
            "toggle the top-right formula popup",
            |_args, demo: &mut Demo, _out| {
                demo.show_formula = !demo.show_formula;
                Ok(())
            },
        ));
        c.register(
            crate::verbs::wireframe_subcommands::<Demo>(|d| &mut d.wireframe)
                .choice(
                    "color",
                    "parent-edge color mode (bare cycles): vertex-gradient|unique-edge|w-depth|active",
                    &["vertex-gradient", "unique-edge", "w-depth", "active"],
                    |d, name| {
                        d.wireframe_color_mode = match name {
                            Some(n) => WireframeColorMode::from_token(n).ok_or_else(|| {
                                anyhow!(
                                    "unknown color mode `{n}` (try vertex-gradient|unique-edge|w-depth|active)"
                                )
                            })?,
                            None => {
                                let all = WireframeColorMode::ALL;
                                let i = all
                                    .iter()
                                    .position(|m| *m == d.wireframe_color_mode)
                                    .unwrap_or(0);
                                all[(i + 1) % all.len()]
                            }
                        };
                        Ok(())
                    },
                )
                .toggle(
                    "nearest-active",
                    "per-edge alpha gradient by cell-crossing strength (bare flips)",
                    |d, v| {
                        d.wireframe_nearest_active = v.unwrap_or(!d.wireframe_nearest_active);
                        Ok(())
                    },
                )
                .custom(
                    "pole",
                    "stereographic projection pole (bare reports; sub: reset | +w | <x y z w>)",
                    &[&["reset", "+w"]],
                    &[],
                    |d, args, out| {
                        match args.first().copied() {
                            None => {
                                let p = d.stereographic_pole;
                                out.line(format!(
                                    "stereographic pole: ({:.3}, {:.3}, {:.3}, {:.3})",
                                    p.x, p.y, p.z, p.w
                                ));
                            }
                            Some("reset") | Some("default") => {
                                d.stereographic_pole = state::STEREOGRAPHIC_DEFAULT_POLE;
                                out.line("stereographic pole: reset to +w");
                            }
                            Some("+w") => {
                                d.stereographic_pole = Vec4::W;
                                out.line("stereographic pole: set to +w (textbook map)");
                            }
                            Some(_) => {
                                let [x, y, z, w] = args else {
                                    return Err(anyhow!("pole needs four components: x y z w"));
                                };
                                let raw = Vec4::new(x.parse()?, y.parse()?, z.parse()?, w.parse()?);
                                let pole = raw.try_normalize().ok_or_else(|| anyhow!(
                                    "pole must have finite, nonzero length"
                                ))?;
                                d.stereographic_pole = pole;
                                out.line(format!(
                                    "stereographic pole: set to ({:.3}, {:.3}, {:.3}, {:.3})",
                                    pole.x, pole.y, pole.z, pole.w
                                ));
                            }
                        }
                        Ok(())
                    },
                )
                .custom(
                    "hyperslice",
                    "cull parent edges to a w-slab around the slice (bare flips; sub: on|off|thickness <N>)",
                    &[&["on", "off", "thickness"]],
                    &[],
                    |d, args, out| {
                        match args.first().copied() {
                            None => {
                                d.wireframe_hyperslice = !d.wireframe_hyperslice;
                                out.line(format!(
                                    "wireframe hyperslice: {} (slab full-width {:.3})",
                                    if d.wireframe_hyperslice { "on" } else { "off" },
                                    d.wireframe_hyperslice_thickness
                                ));
                            }
                            Some("on") => {
                                d.wireframe_hyperslice = true;
                                out.line(format!(
                                    "wireframe hyperslice: on (slab full-width {:.3})",
                                    d.wireframe_hyperslice_thickness
                                ));
                            }
                            Some("off") => {
                                d.wireframe_hyperslice = false;
                                out.line("wireframe hyperslice: off (full edge graph)");
                            }
                            Some("thickness") => match args.get(1).copied() {
                                None => out.line(format!(
                                    "wireframe hyperslice thickness: {:.3}",
                                    d.wireframe_hyperslice_thickness
                                )),
                                Some(token) => {
                                    let t: f32 = token.parse().map_err(|e| {
                                        anyhow!("invalid thickness `{token}`: {e}")
                                    })?;
                                    let max = 2.0 * consts::W_RANGE;
                                    if !(HYPERSLICE_MIN_THICKNESS..=max).contains(&t) {
                                        return Err(anyhow!(
                                            "hyperslice thickness {t} out of range; expected {HYPERSLICE_MIN_THICKNESS}..={max}"
                                        ));
                                    }
                                    d.wireframe_hyperslice_thickness = t;
                                    out.line(format!(
                                        "wireframe hyperslice thickness: set to {t:.3}"
                                    ));
                                }
                            },
                            Some(other) => {
                                return Err(anyhow!(
                                    "unknown hyperslice subcommand `{other}` (try on|off|thickness)"
                                ));
                            }
                        }
                        Ok(())
                    },
                )
                .custom(
                    "points",
                    "vertex + cell-center sprite overlay (bare flips; sub: vertices|cell-centers|size <N>)",
                    &[&["vertices", "cell-centers", "size"]],
                    &[],
                    |d, args, out| {
                        match args.first().copied() {
                            None => {
                                d.points_enabled = !d.points_enabled;
                                out.line(format!(
                                    "points: {}",
                                    if d.points_enabled { "on" } else { "off" }
                                ));
                            }
                            Some("vertices") => {
                                d.points_show_vertices = !d.points_show_vertices;
                                out.line(format!(
                                    "points vertices: {}",
                                    if d.points_show_vertices { "on" } else { "off" }
                                ));
                            }
                            Some("cell-centers") => {
                                d.points_show_cell_centers = !d.points_show_cell_centers;
                                out.line(format!(
                                    "points cell-centers: {}",
                                    if d.points_show_cell_centers { "on" } else { "off" }
                                ));
                            }
                            Some("size") => match args.get(1) {
                                None => out.line(format!(
                                    "points size: {:.1} px",
                                    d.points_size_px
                                )),
                                Some(token) => {
                                    let px: f32 = token.parse().map_err(|e| {
                                        anyhow!("invalid pixel value `{token}`: {e}")
                                    })?;
                                    if !(1.0..=64.0).contains(&px) {
                                        return Err(anyhow!(
                                            "points size {px} out of range; expected 1..=64"
                                        ));
                                    }
                                    d.points_size_px = px;
                                    out.line(format!("points size: set to {px:.1} px"));
                                }
                            },
                            Some(other) => {
                                return Err(anyhow!(
                                    "unknown points subcommand `{other}` (try vertices|cell-centers|size)"
                                ));
                            }
                        }
                        Ok(())
                    },
                ),
        );

        c.register(
            loam_egui::cmd(
                "surface",
                "polychoral surface mode: raster | sdf | off (bare = off); `scale <N>` to resize (per-layer cap alpha lives under `section`)",
                |args, demo: &mut Demo, out| {
                    if matches!(args.first().copied(), Some("scale")) {
                        match args.get(1).copied() {
                            None => {
                                out.line(format!(
                                    "surface scale: {:.3} (multiplies BODY_SIZE)",
                                    demo.surface_scale
                                ));
                            }
                            Some(token) => {
                                let parsed: f32 = token.parse().map_err(|e| {
                                    anyhow!("invalid scale `{token}`: {e}")
                                })?;
                                if !(0.05..=10.0).contains(&parsed) {
                                    return Err(anyhow!(
                                        "surface scale {parsed} out of range; expected 0.05..=10.0"
                                    ));
                                }
                                demo.surface_scale = parsed;
                                demo.rebuild_bodies();
                                let w_range = demo.effective_w_range();
                                demo.w_slice = demo.w_slice.clamp(-w_range, w_range);
                                out.line(format!(
                                    "surface scale: set to {parsed:.3}"
                                ));
                            }
                        }
                        return Ok(());
                    }
                    let next = match args.first().copied() {
                        Some(token) => SurfaceMode::from_token(token).ok_or_else(|| {
                            anyhow!("unknown arg `{token}` (try raster|sdf|off|scale; cap alpha lives under `section`)")
                        })?,
                        None => SurfaceMode::Off,
                    };
                    if next == SurfaceMode::Sdf && demo.sdf_blocked_by_heavy_polychora() {
                        return Err(anyhow!(
                            "SDF rendering for the 120-cell and 600-cell is not yet verified in the browser; use surface raster"
                        ));
                    }
                    if next != demo.surface_mode {
                        demo.surface_mode = next;
                        demo.rebuild_bodies();
                    }
                    Ok(())
                },
            )
            .with_args(&[&["raster", "sdf", "off", "scale"]])
            .with_long_help(
                "raster: draw cross-section caps with face-normal lighting.\n\
                 sdf: raymarch the surface; unavailable for the 120-cell and 600-cell.\n\
                 off: hide surfaces and keep enabled overlays.\n\
                 scale <N>: multiply body size by N (0.05..=10).\n\
                 Bare `surface` selects `off`. `section` controls cap and cross-section alpha.\n\
                 Smooth shapes always use SDF rendering.",
            ),
        );

        c.register(
            loam_egui::subcommands::<Demo>(
                "section",
                "rasterized cross-section layers: cross (drop-w) + cap (projection-following), each with perimeter + alpha",
            )
            .toggle(
                "cross-perimeter",
                "drop-w cross-section perimeter outline (bare flips)",
                |d, v| {
                    d.cross_section.perimeter = v.unwrap_or(!d.cross_section.perimeter);
                    Ok(())
                },
            )
            .toggle(
                "cap-perimeter",
                "projected-cap perimeter outline (bare flips)",
                |d, v| {
                    d.projected_cap.perimeter = v.unwrap_or(!d.projected_cap.perimeter);
                    Ok(())
                },
            )
            .custom(
                "cross-alpha",
                "cross-section fill alpha (0 = off; range (0, 1])",
                &[&[]],
                &[],
                |d, args, out| run_section_alpha("cross", &mut d.cross_section, args, out),
            )
            .custom(
                "cap-alpha",
                "projected-cap fill alpha (0 = off; range (0, 1])",
                &[&[]],
                &[],
                |d, args, out| run_section_alpha("cap", &mut d.projected_cap, args, out),
            ),
        );

        loam_app::camera_rig::register_camera_command(&mut c, |demo| &mut demo.rig);

        c.register(
            loam_egui::cmd::<Demo, _>(
                "handles",
                "toggle the 4D transform handles (on | off; bare flips)",
                |args, demo, out| {
                    let next = match args {
                        [] => !demo.gimbal.enabled,
                        ["on"] => true,
                        ["off"] => false,
                        _ => return Err(anyhow!("usage: handles [on|off]")),
                    };
                    demo.gimbal.enabled = next;
                    out.line(format!("handles: {}", if next { "on" } else { "off" }));
                    Ok(())
                },
            )
            .with_args(&[&["on", "off"]])
            .with_long_help(
                "Drag a ring to rotate the row in its plane. Drag an arrow to translate\n\
                 along its axis. The violet arrow controls w.\n\
                 Handles start hidden and are unavailable in Filmstrip view.",
            ),
        );

        loam_app::environment::register_ground_command(&mut c, |demo| &mut demo.environment);
        loam_app::environment::register_floor_command(&mut c, |demo| &mut demo.environment);

        c
    }
}
