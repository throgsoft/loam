use anyhow::{anyhow, Result};
use glam::{Mat4, Vec2, Vec3, Vec4};
use loam_app::{args::Args, egui, Camera, FrameCtx, OrbitController, RunConfig, SetupCtx};
use loam_egui::{Console, ConsoleUi};
use loam_math::WPlane;
use loam_math::{Bivector4, EuclideanR3, Rotor, Rotor4};
use loam_render::{
    device::RenderDevice,
    raymarch::{
        polytope_extended_sdfs_wgsl, BodyUniform, Hyperslice4DNode, HYPERSLICE_KERNEL_WGSL,
    },
    DepthBuffer, DepthMode, LineRasterNode, PointRasterNode, SkyGroundNode, SkyGroundUniforms,
    TriangleRasterNode, Viewport,
};
use loam_shape::polytope::{
    polytope_section_faces_append, polytope_section_perimeter_append, vertex_color_by_position,
    SectionScratch,
};

// 24-bit depth cracks the 600-cell's densely-packed caps.
const SECTION_FACES_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

// Shared by the marched half-space and the background ground, or the swap shows a seam.
const FLOOR_Y: f32 = 0.0;

use loam_scene::{Scene4, SceneNode4};
use loam_shape::LineMesh;
use winit::window::WindowAttributes;

mod active;
mod catalog;
mod color;
mod composer;
mod console;
mod consts;
mod director;
mod filmstrip;
mod hud;
mod hypergimbal;
mod physics;
mod projections;
mod render;
mod sections;
mod shapes;
mod shell;
mod spins;
mod state;
mod toybox;
mod ui;
mod verbs;
mod wireframe_geom;

// At the crate root: `#[global_allocator]` is a per-binary singleton (E0152).
#[cfg(test)]
pub(crate) mod alloc_probe {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    // try_with skips destroyed TLS; the const Cell initializer and callbacks cannot panic.
    thread_local! {
        static BYTES: Cell<usize> = const { Cell::new(0) };
    }

    pub struct Counting;

    // SAFETY: Methods preserve System contracts; const TLS and wrapping Cell updates cannot unwind.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let _ = BYTES.try_with(|bytes| bytes.set(bytes.get().wrapping_add(layout.size())));
            // SAFETY: The caller supplies a valid nonzero allocation layout.
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: The caller supplies a live System allocation and its original layout.
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let _ = BYTES.try_with(|bytes| bytes.set(bytes.get().wrapping_add(new_size)));
            // SAFETY: The caller supplies a live System allocation, its layout, and a valid new size.
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    pub fn bytes_allocated_by(body: impl FnOnce()) -> usize {
        let before = BYTES.with(Cell::get);
        body();
        BYTES.with(Cell::get).wrapping_sub(before)
    }
}

#[cfg(test)]
#[global_allocator]
static COUNTING_ALLOCATOR: alloc_probe::Counting = alloc_probe::Counting;

use active::combo_name;
use catalog::{parse_row, SHAPE_CATALOG};
use color::{unique_edge_palette, w_depth_color};
#[cfg(test)]
use consts::SPACE_TESSELLATION_SAMPLES;
use consts::{
    BODY_SIZE, BODY_Y, HYPERSLICE_MIN_THICKNESS, T_SCRUB_RATE, T_SLIDER_INITIAL, W_SCRUB_RATE,
};
use director::Playback;
#[cfg(test)]
use loam_math::Bivector;
#[cfg(test)]
use loam_time::Director;
use physics::PlaygroundPhysics;
use state::{
    set_if_changed, Demo, RotationMode, RowFrame, SurfaceMode, ViewMode, WireframeColorMode,
};
use verbs::WireframeControls;
use wireframe_geom::*;

fn compute_cell_strengths(
    cells: &[&[u32]],
    local_vertices: &[Vec4],
    w_slice: f32,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.extend(cells.iter().map(|cell| {
        let (w_min, w_max) = cell_w_range(cell, local_vertices);
        let half_extent = (w_max - w_min) * 0.5;
        if half_extent <= 0.0 {
            return 0.0;
        }
        let mid = (w_min + w_max) * 0.5;
        let dist = (w_slice - mid).abs();
        (1.0 - dist / half_extent).clamp(0.0, 1.0)
    }));
}

// `content_rect` does not shrink for a panel and would seat under the menu bar.
fn formula_popup_seat(ctx: &egui::Context) -> egui::Pos2 {
    const RIGHT_INSET: f32 = 280.0;
    const TOP_INSET: f32 = 16.0;
    let area = ctx.available_rect();
    egui::pos2(area.right() - RIGHT_INSET, area.top() + TOP_INSET)
}

fn shader_source() -> String {
    let scene = Scene4::new(SceneNode4::halfspace(Vec4::Y, FLOOR_Y));
    format!(
        "{kernel}\n{polytope}\n{scene}\n",
        kernel = HYPERSLICE_KERNEL_WGSL,
        polytope = polytope_extended_sdfs_wgsl(),
        scene = scene.to_hyperslice_wgsl_gated("u.w_slice", "u.params.x"),
    )
}

struct DemoNodes {
    marcher: Hyperslice4DNode,
    section_edges: LineRasterNode,
    parent_wireframe: LineRasterNode,
    gimbal: LineRasterNode,
    points: PointRasterNode,
    section_faces: TriangleRasterNode,
    section_faces_translucent: TriangleRasterNode,
}

fn build_nodes(device: &wgpu::Device, format: wgpu::TextureFormat, samples: u32) -> DemoNodes {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("polytope_playground shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source().into()),
    });
    DemoNodes {
        marcher: Hyperslice4DNode::new(device, format, &module, samples),
        section_edges: LineRasterNode::new(
            device,
            format,
            DepthMode::ReadOnly {
                format: SECTION_FACES_DEPTH_FORMAT,
            },
            samples,
        ),
        parent_wireframe: LineRasterNode::new(
            device,
            format,
            DepthMode::ReadOnly {
                format: SECTION_FACES_DEPTH_FORMAT,
            },
            samples,
        ),
        gimbal: LineRasterNode::new(device, format, DepthMode::Off, samples),
        // A depth test hides a vertex behind its own cap under drop-w.
        points: PointRasterNode::new(device, format, DepthMode::Off, samples),
        section_faces: TriangleRasterNode::new(
            device,
            format,
            DepthMode::ReadWrite {
                format: SECTION_FACES_DEPTH_FORMAT,
            },
            loam_render::FragmentShading::FaceNormalLambert,
            samples,
        ),
        section_faces_translucent: TriangleRasterNode::new(
            device,
            format,
            DepthMode::ReadOnly {
                format: SECTION_FACES_DEPTH_FORMAT,
            },
            loam_render::FragmentShading::FaceNormalLambert,
            samples,
        ),
    }
}

impl Demo {
    pub(crate) fn new(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let row = parse_row(&Args::current())?;

        let DemoNodes {
            marcher: mut node,
            section_edges,
            parent_wireframe,
            gimbal: gimbal_node,
            points: points_node,
            section_faces,
            section_faces_translucent,
        } = build_nodes(
            &ctx.rd.device,
            ctx.rd.target_format(),
            ctx.rd.sample_count(),
        );

        let surface_mode = SurfaceMode::default();
        let row_len = row.len();
        let physics = PlaygroundPhysics::new(row_len, BODY_SIZE)
            .ok_or_else(|| anyhow::anyhow!("invalid playground body configuration"))?;
        let bodies: Vec<BodyUniform> = row
            .iter()
            .enumerate()
            .map(|(slot, entry)| {
                state::sdf_body_uniform(
                    &physics,
                    entry,
                    slot,
                    row.len(),
                    Rotor4::IDENTITY,
                    BODY_SIZE,
                    surface_mode,
                )
            })
            .collect();
        node.set_bodies(&bodies);

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 3.0, 9.0);
        camera.near = 0.1;
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        orbit.set_orbit(8.0, -0.25);

        let initial_w = 0.0;

        Ok(Self {
            physics,
            left_was_down: false,
            gimbal: hypergimbal::GimbalUi::default(),
            gimbal_node,
            camera,
            orbit,
            rig: loam_app::camera_rig::CameraRig::default(),
            node,
            sky_ground: SkyGroundNode::new(
                &ctx.rd.device,
                ctx.rd.target_format(),
                SECTION_FACES_DEPTH_FORMAT,
                ctx.rd.sample_count(),
            ),
            sdf_upload_pending: true,
            uploaded_rotors: Vec::new(),
            section_edges,
            parent_wireframe,
            wireframe: WireframeControls::default(),
            wireframe_nearest_active: true,
            cross_section: state::SectionLayer::CROSS_SECTION_DEFAULT,
            projected_cap: state::SectionLayer::PROJECTED_CAP_DEFAULT,
            wireframe_color_mode: WireframeColorMode::default(),
            schlegel_params: None,
            stereographic_pole: state::STEREOGRAPHIC_DEFAULT_POLE,
            wireframe_hyperslice: false,
            wireframe_hyperslice_thickness: consts::HYPERSLICE_DEFAULT_THICKNESS,
            unique_edge_palette_cache: std::collections::HashMap::new(),
            cell_centers_cache: std::collections::HashMap::new(),
            surface_scale: 1.0,
            environment: loam_app::environment::Environment::default(),
            section_faces,
            section_faces_translucent,
            section_faces_projected_scratch: loam_shape::TriangleMesh::<3>::default(),
            section_clip_projected_scratch: Vec::new(),
            points_node,
            points_enabled: false,
            points_show_vertices: true,
            points_show_cell_centers: true,
            points_size_px: 4.0,
            points_mesh_scratch: loam_shape::PointMesh::<3>::default(),
            section_faces_depth: None,
            section_world_vertices_scratch: Vec::new(),
            section_faces_mesh_scratch: loam_shape::TriangleMesh::<3>::default(),
            body_uniform_scratch: Vec::new(),
            strip_cells_scratch: Vec::new(),
            wireframe_section_edges_scratch: LineMesh::<3>::default(),
            body_perimeter_scratch: LineMesh::<3>::default(),
            section_cap_scratch: SectionScratch::default(),
            wireframe_parent_lines_scratch: LineMesh::<3>::default(),
            overlay_local_vertices_scratch: Vec::new(),
            overlay_center_locals_scratch: Vec::new(),
            overlay_cell_strengths_scratch: Vec::new(),
            surface_mode,
            row,
            w_slice: initial_w,
            slider_up_held: false,
            slider_down_held: false,
            slider_left_held: false,
            slider_right_held: false,
            rotate: false,
            spins: spins::SlotSpins::new(row_len),
            playback: load_director(&Args::current(), row_len)?,
            rate_scale: 1.0,
            rot_time: 0.0,
            t_slider_max: T_SLIDER_INITIAL,
            expanded: false,
            show_help: false,
            show_render_panel: false,
            example_callout: loam_egui::CalloutState {
                window_pos: egui::Pos2::new(220.0, 120.0),
                open: false,
            },
            mode_annotation_open: loam_egui::CalloutState {
                window_pos: egui::Pos2::new(220.0, 300.0),
                open: false,
            },
            show_formula: false,
            show_controls: true,
            show_text_hud: false,
            view_mode: ViewMode::Shapes,
            strip_w: true,
            strip_t: false,
            strip_swap_axes: false,
            strip_count_w: 11,
            strip_count_t: 5,
            strip_t_extent: T_SLIDER_INITIAL,
            strip_subject: SHAPE_CATALOG[3],
            rotation_mode: RotationMode::Active,
            pending_mode: None,
            pending_view_mode: None,
            pending_actions: Vec::new(),
            seq: Vec::new(),
            draft: Vec::new(),
            formula_input: String::new(),
            formula_error: None,
        })
    }

    pub(crate) fn tick(&mut self, dt: f32) {
        self.apply_pending_gimbal();
        let dir = (self.slider_up_held as i32 - self.slider_down_held as i32) as f32;
        let host_owns_w = !self
            .playback
            .as_ref()
            .is_some_and(director::Playback::owns_w_slice);
        if dir != 0.0 && host_owns_w {
            let w_range = self.effective_w_range();
            self.w_slice = (self.w_slice + dir * W_SCRUB_RATE * dt).clamp(-w_range, w_range);
        }

        let t_dir = (self.slider_right_held as i32 - self.slider_left_held as i32) as f32;
        if t_dir != 0.0 {
            self.rot_time = (self.rot_time + t_dir * T_SCRUB_RATE * dt).max(0.0);
            const T_SLIDER_CAP: f32 = 1.0e6;
            if self.rot_time > self.t_slider_max {
                let new_max = (self.rot_time * 2.0).min(T_SLIDER_CAP);
                self.t_slider_max = new_max;
                if self.rot_time > T_SLIDER_CAP {
                    self.rot_time = T_SLIDER_CAP;
                }
            }
            self.recompose_spins_at(self.rot_time);
        }

        let dt_animation = if self.rotate {
            dt * self.rate_scale
        } else {
            0.0
        };
        let omega = match self.rotation_mode {
            RotationMode::Active => Bivector4::ZERO,
            RotationMode::Composer => self.omega_animation(),
        };
        director::step_row_rotation(
            self.playback.as_mut(),
            &mut self.spins,
            &mut self.w_slice,
            &mut self.rot_time,
            dt_animation,
            self.rotation_mode,
            omega,
        );
        const T_SLIDER_CAP: f32 = 1.0e6;
        if self.rot_time > self.t_slider_max {
            let new_max = (self.rot_time * 2.0).min(T_SLIDER_CAP);
            self.t_slider_max = new_max;
            if self.rot_time > T_SLIDER_CAP {
                self.rot_time = T_SLIDER_CAP;
            }
        }
        // Sync before the step, so the tick collides this frame's rotor.
        self.sync_physics_row();
        let bodies_moving = !self.physics.at_rest();
        self.physics.tick(dt);
        self.sdf_upload_pending |= bodies_moving;
    }

    pub(crate) fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        let viewport = {
            let cfg = &ctx.rd.surface_bundle.config;
            (cfg.width, cfg.height)
        };

        self.camera.aspect = viewport.0 as f32 / viewport.1.max(1) as f32;
        let pointer_free = !ctx.ui_capture.pointer && !self.rig.is_flying();
        self.update_gimbal(pointer_free, &ctx.input, viewport);

        if self.sdf_upload_pending || self.spins.rotors_differ_from(&self.uploaded_rotors) {
            self.rebuild_bodies();
        }

        let lift_orbit = self.view_mode == ViewMode::Filmstrip && self.strip_w && self.strip_t;
        self.orbit.target.y = if lift_orbit { BODY_Y } else { 0.0 };
        self.rig.advance(
            loam_app::orbit_on_right(ctx.input),
            ctx.ui_capture,
            &mut self.camera,
            &mut self.orbit,
            ctx.dt,
            ctx.runtime,
        );
        let view = self.camera.view();

        // No flush here: the viewport is only known in `render`, which flushes once.
        let cfg = &ctx.rd.surface_bundle.config;
        {
            let mut changed = false;
            let u = self.node.uniforms_mut();
            changed |= set_if_changed(&mut u.camera_pos, view.position.to_array());
            changed |= set_if_changed(&mut u.camera_forward, view.forward.to_array());
            changed |= set_if_changed(&mut u.camera_right, view.right.to_array());
            changed |= set_if_changed(&mut u.camera_up, view.up.to_array());
            changed |= set_if_changed(&mut u.fov_y_tan, (60.0_f32.to_radians() * 0.5).tan());
            changed |= set_if_changed(&mut u.resolution, [cfg.width as f32, cfg.height as f32]);
            changed |= set_if_changed(&mut u.w_slice, self.w_slice);
            changed |= set_if_changed(
                &mut u.params[0],
                if self.environment.floor_visible {
                    1.0
                } else {
                    0.0
                },
            );
            // Outside the change test: a clock tick must not upload.
            u.time = ctx.time;
            u.tick = ctx.tick as f32;
            self.sdf_upload_pending |= changed;
        }

        // After every reader of the press edge.
        self.left_was_down = ctx.input.buttons.left.down;
    }

    pub(crate) fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        // Keyboard zoom changes PPP under a native-resolution surface and letter-boxes the scene.
        ctx.options_mut(|o| o.zoom_with_keyboard = false);

        const MENU_BAR_PAD: f32 = 24.0;
        const LABEL_INSET: f32 = 14.0;
        let build_label = format!("build: {}{}", env!("BUILD_HASH"), env!("BUILD_DIRTY"),);
        egui::Area::new(egui::Id::new("polytope-playground-build"))
            .anchor(
                egui::Align2::RIGHT_TOP,
                [-LABEL_INSET, MENU_BAR_PAD + LABEL_INSET],
            )
            .show(ctx, |ui| {
                ui.add(egui::Label::new(
                    egui::RichText::new(build_label)
                        .monospace()
                        .size(11.0)
                        .color(egui::Color32::from_gray(140)),
                ));
            });

        if self.show_formula {
            let formula = self.formula_string();
            let name = if self.rotation_mode == RotationMode::Active {
                combo_name(&self.spins.spin().active)
            } else {
                None
            };
            let bivec = self.spins.row_rotor().log();
            let default_pos = formula_popup_seat(ctx);
            let popup_frame = egui::Frame::popup(&ctx.style()).inner_margin(8.0);
            const FORMULA_POPUP_W: f32 = 320.0;
            egui::Window::new("formula")
                .id(egui::Id::new("polytope-playground-formula"))
                .title_bar(false)
                .resizable(false)
                .collapsible(false)
                .movable(true)
                .default_pos(default_pos)
                .default_width(FORMULA_POPUP_W)
                .max_width(FORMULA_POPUP_W)
                .frame(popup_frame)
                .show(ctx, |ui| {
                    ui.set_max_width(FORMULA_POPUP_W);
                    if !formula.is_empty() {
                        ui.add(egui::Label::new(egui::RichText::new(&formula).monospace()).wrap());
                    }
                    if let Some(n) = name {
                        ui.add(egui::Label::new(
                            egui::RichText::new(n).color(egui::Color32::from_rgb(255, 217, 140)),
                        ));
                    }
                    ui.separator();
                    ui.label(egui::RichText::new("log(R) bivector").small().weak());
                    loam_egui::bivector_matrix(ui, &bivec);
                });
        }

        if self.view_mode == ViewMode::Filmstrip {
            self.render_filmstrip_cell_labels(ctx);
        }

        if self.show_controls {
            self.render_overlay(ctx, frame.runtime);
        }

        self.render_help_window(ctx);
        self.render_render_panel(ctx);
        self.render_example_callout(ctx, frame);
        self.render_mode_annotation(ctx, frame);
    }

    fn render_mode_annotation(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        if !self.mode_annotation_open.open {
            return;
        }
        let Some(annotation) = state::mode_annotation(self.wireframe.projection) else {
            return;
        };

        let row_frame = self.row_frame();
        let Some((slot, _entry)) = row_frame
            .row
            .iter()
            .enumerate()
            .find(|(_, e)| e.shape.polytope4().is_some())
        else {
            return;
        };
        let world_pos = row_frame.pose(slot).position_r3();

        let cfg = &frame.rd.surface_bundle.config;
        let ppp = ctx.pixels_per_point();
        let vp_w = (cfg.width as f32 / ppp).round() as u32;
        let vp_h = (cfg.height as f32 / ppp).round() as u32;
        let Some(screen_pos) =
            loam_egui::world_to_screen(&self.camera, world_pos, (vp_w, vp_h), &EuclideanR3)
        else {
            return;
        };

        loam_egui::callout(
            ctx,
            "polytope-playground-mode-annotation",
            screen_pos,
            &mut self.mode_annotation_open,
            annotation.title,
            |ui| {
                ui.label(&annotation.body);
            },
        );
    }

    fn render_example_callout(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        if !self.example_callout.open {
            return;
        }
        let row_frame = self.row_frame();
        let Some((slot, entry)) = row_frame
            .row
            .iter()
            .enumerate()
            .find(|(_, e)| e.shape.polytope4().is_some())
        else {
            return;
        };
        let polytope = entry.shape.polytope4().expect("filter guarantees Some");
        let label = entry.label;
        let world_pos = row_frame.anchor_r3(slot, polytope.topology().vertices[0]);

        let cfg = &frame.rd.surface_bundle.config;
        let ppp = ctx.pixels_per_point();
        let vp_w = (cfg.width as f32 / ppp).round() as u32;
        let vp_h = (cfg.height as f32 / ppp).round() as u32;
        let Some(screen_pos) =
            loam_egui::world_to_screen(&self.camera, world_pos, (vp_w, vp_h), &EuclideanR3)
        else {
            return;
        };

        let title = format!("{label} vertex 0");
        loam_egui::callout(
            ctx,
            "polytope-playground-example-callout",
            screen_pos,
            &mut self.example_callout,
            &title,
            |ui| {
                ui.label(
                    "This line tracks vertex 0 as the shape rotates. Drag the panel to move the label.",
                );
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Anchor coordinates").strong());
                ui.label(format!(
                    "world R³: ({:.2}, {:.2}, {:.2})",
                    world_pos.x, world_pos.y, world_pos.z
                ));
            },
        );
    }

    pub(crate) fn on_key(
        &mut self,
        kc: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    ) {
        use winit::event::ElementState;
        use winit::keyboard::KeyCode;
        let pressed = state == ElementState::Pressed;
        match kc {
            KeyCode::ArrowUp => self.slider_up_held = pressed,
            KeyCode::ArrowDown => self.slider_down_held = pressed,
            KeyCode::ArrowLeft => self.slider_left_held = pressed,
            KeyCode::ArrowRight => self.slider_right_held = pressed,
            KeyCode::KeyH if pressed => self.show_controls = !self.show_controls,
            KeyCode::KeyT if pressed => {
                self.rotate = !self.rotate;
            }
            KeyCode::Space if pressed && !self.rig.is_flying() => {
                self.rotate = !self.rotate;
            }
            KeyCode::AltLeft | KeyCode::AltRight if self.rig.is_flying() => {
                self.rig.freecam.on_alt(pressed, ctx.runtime);
            }
            KeyCode::Digit1 | KeyCode::Numpad1 if pressed => self.toggle_plane(0),
            KeyCode::Digit2 | KeyCode::Numpad2 if pressed => self.toggle_plane(1),
            KeyCode::Digit3 | KeyCode::Numpad3 if pressed => self.toggle_plane(2),
            KeyCode::Digit4 | KeyCode::Numpad4 if pressed => self.toggle_plane(3),
            KeyCode::Digit5 | KeyCode::Numpad5 if pressed => self.toggle_plane(4),
            KeyCode::Digit6 | KeyCode::Numpad6 if pressed => self.toggle_plane(5),
            _ => {}
        }
    }

    pub(crate) fn title(&self, _fps: f32) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("polytope playground")
    }
}

// Not a field on `Demo`: `apply_command` borrows the console and the demo at once.
pub(crate) struct RotateScene {
    demo: Demo,
    console: Console<Demo>,
    text_hud: hud::TextHud,
    hud_seat: hud::HudSeat,
}

const SECTION_ALPHA_MIN_VISIBLE: f32 = 0.05;

fn run_section_alpha(
    layer_name: &str,
    layer: &mut state::SectionLayer,
    args: &[&str],
    out: &mut loam_egui::console::ConsoleWriter,
) -> anyhow::Result<()> {
    match args.first().copied() {
        None => {
            let state = if layer.fill_visible() {
                if layer.surface_alpha >= 1.0 {
                    "opaque"
                } else {
                    "translucent"
                }
            } else {
                "off"
            };
            out.line(format!(
                "section {layer_name}-alpha: {:.3} ({state})",
                layer.surface_alpha
            ));
        }
        Some(token) => {
            let parsed: f32 = token
                .parse()
                .map_err(|e| anyhow!("invalid alpha `{token}`: {e}"))?;
            let valid = parsed == 0.0 || (SECTION_ALPHA_MIN_VISIBLE..=1.0).contains(&parsed);
            if !valid {
                return Err(anyhow!(
                    "section {layer_name}-alpha {parsed} out of range; expected 0 (off) or {SECTION_ALPHA_MIN_VISIBLE}..=1.0"
                ));
            }
            layer.surface_alpha = parsed;
            out.line(format!(
                "section {layer_name}-alpha: set to {parsed:.3}{}",
                if parsed == 0.0 { " (off)" } else { "" }
            ));
        }
    }
    Ok(())
}

impl RotateScene {
    pub(crate) fn new(
        ctx: &mut SetupCtx<'_>,
        control: &loam_app::shell::SceneControl,
    ) -> Result<Self> {
        Ok(Self {
            demo: Demo::new(ctx)?,
            console: Self::build_console(ctx.runtime, control),
            text_hud: hud::TextHud::new(
                &ctx.rd.device,
                &ctx.rd.queue,
                ctx.rd.target_format(),
                ctx.rd.sample_count(),
            )?,
            hud_seat: hud::HudSeat::default(),
        })
    }
}

fn load_director(args: &Args, slots: usize) -> Result<Option<Playback>> {
    if args.has_bare_flag("director") {
        return Err(anyhow!(
            "--director needs its path attached: --director=path/to/timeline.ron"
        ));
    }
    let Some(path) = args.get("director") else {
        return Ok(None);
    };
    let text =
        std::fs::read_to_string(path).map_err(|e| anyhow!("reading timeline `{path}`: {e}"))?;
    let director =
        Director::from_ron(&text).map_err(|e| anyhow!("parsing timeline `{path}`: {e}"))?;
    Ok(Some(Playback::new(director, slots)?))
}

impl loam_app::shell::Scene for RotateScene {
    fn tick(&mut self, dt: f32, _ctx: &mut loam_app::TickCtx) {
        self.demo.tick(dt);
    }

    fn menus(&mut self, ui: &mut egui::Ui) {
        self.demo.menu_contents(ui);
    }

    fn apply_command(
        &mut self,
        cmd: &loam_app::command::CommandLine,
        _ctx: &mut loam_app::command::CommandCtx<'_>,
    ) -> Result<()> {
        self.console
            .dispatch(&cmd.name, &cmd.arg_refs(), &mut self.demo);
        Ok(())
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        self.demo.update(ctx);
    }

    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        self.hud_seat = hud::hud_seat(ctx.available_rect(), ctx.pixels_per_point());
        self.demo.ui(ctx, frame);
        loam_app::log::pump_into(&mut self.console);
        frame.runtime.pump_console(&mut self.console);
        self.console.ui(ctx);
        // After the console UI, so a line typed this frame is drained next frame.
        frame.runtime.forward_console(&mut self.console);
    }

    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    ) {
        if !ctx.ui_capture.keyboard {
            self.demo.on_key(code, state, ctx);
        }
    }

    fn record(&mut self, ctx: &mut loam_app::RenderCtx<'_>) -> Result<()> {
        self.demo.record(ctx.rd, ctx.encoder, ctx.view)?;
        self.text_hud.record(ctx, &self.demo, self.hud_seat);
        Ok(())
    }

    fn title(&self, fps: f32) -> std::borrow::Cow<'static, str> {
        self.demo.title(fps)
    }
}

fn main() -> Result<()> {
    loam_app::run::<loam_app::shell::SceneShell<shell::Playground>>(RunConfig {
        window: WindowAttributes::default()
            .with_title("polytope playground")
            .with_visible(false),
        ..RunConfig::default()
    })
}

#[cfg(test)]
mod color_tests {
    use super::*;

    #[test]
    fn cell_strength_vanishes_at_both_ends_and_for_flat_cells() {
        let cells: [&[u32]; 2] = [&[0, 1], &[2, 3]];
        let vertices = [Vec4::W * -0.5, Vec4::W * 0.5, Vec4::ZERO, Vec4::X];
        let mut strengths = vec![99.0; 4];
        for (slice, expected) in [(-5.0, 0.0), (-0.5, 0.0), (0.0, 1.0), (0.5, 0.0), (5.0, 0.0)] {
            compute_cell_strengths(&cells, &vertices, slice, &mut strengths);
            assert_eq!(strengths, [expected, 0.0], "slice {slice}");
        }
    }
}

#[cfg(test)]
mod section_command_tests {
    use super::*;
    use loam_egui::console::ConsoleWriter;

    #[test]
    fn section_alpha_sets_off_and_visible_rejects_faint_and_bad() {
        let run = |start: f32, args: &[&str]| -> (f32, bool) {
            let mut layer = state::SectionLayer {
                perimeter: true,
                surface_alpha: start,
            };
            let mut out = ConsoleWriter::new();
            let ok = run_section_alpha("cross", &mut layer, args, &mut out).is_ok();
            (layer.surface_alpha, ok)
        };

        assert_eq!(run(1.0, &["0.5"]), (0.5, true), "in-range alpha is set");
        assert_eq!(run(0.5, &["1.0"]), (1.0, true), "opaque alpha is set");
        assert_eq!(run(0.85, &["0"]), (0.0, true), "0 turns the layer off");
        let (val, ok) = run(0.85, &["0.01"]);
        assert!(!ok, "faint (0, MIN) alpha must be rejected");
        assert_eq!(val, 0.85, "rejected faint alpha leaves the field untouched");
        assert_eq!(run(0.85, &["2.0"]).0, 0.85, "over-range alpha is rejected");
        assert_eq!(
            run(0.85, &["notafloat"]).0,
            0.85,
            "unparseable alpha is rejected"
        );
        assert_eq!(run(0.7, &[]), (0.7, true), "bare query leaves the field");
    }
}

#[cfg(test)]
mod director_arg_tests {
    use super::*;

    const DEFAULT_SLOTS: usize = 4;

    #[test]
    fn no_director_argument_leaves_the_row_on_the_ui_clock() {
        assert!(load_director(&Args::default(), DEFAULT_SLOTS)
            .unwrap()
            .is_none());
    }

    #[test]
    fn the_shipped_timeline_loads_through_the_flag_and_leaves_the_row_its_tail() {
        let args = Args::from_pairs([("director", "timelines/row-sweep.ron")]);
        let playback = load_director(&args, DEFAULT_SLOTS)
            .expect("the committed timeline loads")
            .expect("the flag yields a playback");
        assert_eq!(playback.directed(), [true, false, false, false]);
        assert!(playback.owns_w_slice());
    }

    #[test]
    fn the_space_separated_director_form_is_diagnosed_rather_than_ignored() {
        let args = Args::from_argv(["--director", "timelines/row-sweep.ron"]);
        let err =
            load_director(&args, DEFAULT_SLOTS).expect_err("a bare --director is not a default");
        assert!(format!("{err:#}").contains("--director="), "{err:#}");
    }

    #[test]
    fn an_unreadable_timeline_path_fails_setup() {
        let args = Args::from_pairs([("director", "no-such-directory-for-a-timeline/x.ron")]);
        let err = load_director(&args, DEFAULT_SLOTS).expect_err("missing file");
        assert!(format!("{err:#}").contains("x.ron"), "{err:#}");
    }

    #[test]
    fn a_timeline_the_booted_row_cannot_host_fails_setup() {
        let args = Args::from_pairs([("director", "timelines/row-sweep.ron")]);
        assert!(
            load_director(&args, 1).is_ok(),
            "slot 0 fits a one-slot row"
        );
        let err = load_director(&args, 0).expect_err("no slot 0 in an empty row");
        assert!(format!("{err:#}").contains("0-slot row"), "{err:#}");
    }
}

#[cfg(test)]
mod formula_popup_tests {
    use super::*;

    #[test]
    fn formula_popup_seats_below_the_menu_bar_panel() {
        let ctx = egui::Context::default();
        let mut seat = None;
        let mut bar_bottom = 0.0;
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            bar_bottom = egui::TopBottomPanel::top("shell-menu-bar")
                .exact_height(64.0)
                .show(ctx, |ui| {
                    ui.label("bar");
                })
                .response
                .rect
                .bottom();
            seat = Some(formula_popup_seat(ctx));
        });
        let seat = seat.expect("run closure sets the seat");
        assert!(
            seat.y >= bar_bottom,
            "popup seat y {} must clear the menu bar bottom {bar_bottom}",
            seat.y
        );
    }
}

#[cfg(test)]
mod hyperslice_filter_tests {
    use super::*;

    #[test]
    fn slab_overlap_includes_touching_ends_and_clamps_nonpositive_thickness() {
        for (lo, hi, slice, thickness, expected) in [
            (0.8, 0.9, 0.0, 0.2, false),
            (-0.9, -0.8, 0.0, 0.2, false),
            (-0.5, 0.5, 0.0, 0.2, true),
            (-0.05, 0.05, 0.0, 0.2, true),
            (0.45, 0.55, 0.5, 0.2, true),
            (-0.6, -0.5, 0.0, 1.0, true),
            (0.5, 0.6, 0.0, 1.0, true),
            (-0.3, 0.3, 0.0, 0.0, true),
            (0.1, 0.3, 0.0, 0.0, false),
            (0.0, 0.3, 0.0, 0.0, true),
            (-0.3, 0.3, 0.0, -5.0, true),
            (0.1, 0.3, 0.0, -5.0, false),
        ] {
            assert_eq!(
                slab_overlaps(lo, hi, slice, thickness),
                expected,
                "[{lo}, {hi}], {slice}, {thickness}"
            );
        }
    }
}

#[cfg(test)]
mod section_cap_projection_tests {
    use super::*;

    #[test]
    fn section_cap_matches_wireframe_under_perspective4d() {
        let focal = 2.0;
        let w_slice = 0.4;
        let proj = loam_math::Projection::Perspective4D {
            focal_distance: focal,
        };
        let body_pos = Vec3::new(1.3, -0.7, 0.2);
        let scale = perspective_scale_at_w(w_slice, &proj);
        assert!(scale.is_some(), "Perspective4D must take the affine shim");
        for cap_r3 in [[0.5, 0.0, 0.0], [0.0, 0.3, -0.2], [-0.4, 0.1, 0.6]] {
            let via_shim =
                cap_vertex_projected_and_world(cap_r3, w_slice, scale, &proj, body_pos).1;
            let p4 = Vec4::new(cap_r3[0], cap_r3[1], cap_r3[2], w_slice);
            let via_wireframe = (project_to_world(p4, &proj, body_pos)).to_array();
            for k in 0..3 {
                assert!(
                    (via_shim[k] - via_wireframe[k]).abs() < 1e-5,
                    "cap {cap_r3:?} component {k}: shim {} vs wireframe {}",
                    via_shim[k],
                    via_wireframe[k]
                );
            }
        }
    }

    #[test]
    fn section_cap_per_vertex_finite_under_stereographic() {
        let w_slice = 0.0;
        let proj = loam_math::Projection::Stereographic { pole: Vec4::W };
        let body_pos = Vec3::new(0.5, 0.0, -0.3);
        let scale = perspective_scale_at_w(w_slice, &proj);
        assert_eq!(scale, None, "Stereographic must take the per-vertex path");
        for cap_r3 in [
            [0.5, 0.0, 0.0],
            [0.0, -0.4, 0.3],
            [0.95, 0.0, 0.0],
            [0.02, -0.01, 0.015],
        ] {
            let world = cap_vertex_projected_and_world(cap_r3, w_slice, scale, &proj, body_pos).1;
            for (k, c) in world.iter().enumerate() {
                assert!(
                    c.is_finite(),
                    "cap {cap_r3:?} produced non-finite world component {k}: {c}"
                );
            }
        }
    }

    #[test]
    fn affine_wireframe_keeps_single_segment_and_caps_land_on_it() {
        let proj = loam_math::Projection::Perspective4D {
            focal_distance: 2.0,
        };
        let body_pos = Vec3::ZERO;
        let w_slice = 0.0;
        let a = Vec4::new(0.5, 0.4, -0.3, 0.5);
        let b = Vec4::new(0.5, 0.4, -0.3, -0.5);
        let mut mesh = LineMesh::<3>::default();
        let white = [1.0, 1.0, 1.0, 1.0];
        push_blended_edge(
            &mut mesh,
            a,
            b,
            Vec4::ZERO,
            white,
            white,
            1.0,
            0.0,
            &proj,
            body_pos,
            SPACE_TESSELLATION_SAMPLES,
            STEREOGRAPHIC_VIEW_RADIUS,
        );
        assert_eq!(
            mesh.segments.len(),
            1,
            "affine flat edge must stay a single segment"
        );
        let mid = a.lerp(b, 0.5);
        let cap_r3 = [mid.x, mid.y, mid.z];
        let scale = perspective_scale_at_w(w_slice, &proj);
        let cap = Vec3::from_array(
            cap_vertex_projected_and_world(cap_r3, w_slice, scale, &proj, body_pos).1,
        );
        let (s, e) = mesh.segments[0];
        let gap = point_to_segment_distance(cap, Vec3::from_array(s), Vec3::from_array(e));
        assert!(
            gap < 1e-5,
            "affine cap must lie on its single-segment edge, gap {gap}"
        );
    }

    #[test]
    fn honest_section_cap_is_projection_invariant_projected_cap_is_not() {
        let body_pos = Vec3::new(0.7, -0.2, 0.4);
        let w_slice = 0.3;
        let cap_r3 = [0.4, -0.25, 0.15];
        let actives = [
            loam_math::Projection::Identity,
            loam_math::Projection::Perspective4D {
                focal_distance: 2.0,
            },
            loam_math::Projection::Stereographic { pole: Vec4::W },
            loam_math::Projection::schlegel(Vec4::W, 0.5, 0.9),
        ];

        let honest_reference = {
            let proj = state::section_layer_projection(true, loam_math::Projection::Identity);
            let scale = perspective_scale_at_w(w_slice, &proj);
            cap_vertex_projected_and_world(cap_r3, w_slice, scale, &proj, body_pos).1
        };
        let mut projected_caps = Vec::new();
        for active in actives {
            let honest_proj = state::section_layer_projection(true, active);
            assert_eq!(
                honest_proj,
                loam_math::Projection::Identity,
                "honest layer must stay drop-w under {active:?}"
            );
            let honest_scale = perspective_scale_at_w(w_slice, &honest_proj);
            let honest = cap_vertex_projected_and_world(
                cap_r3,
                w_slice,
                honest_scale,
                &honest_proj,
                body_pos,
            )
            .1;
            for k in 0..3 {
                assert!(
                    (honest[k] - honest_reference[k]).abs() < 1e-6,
                    "honest cap drifted under {active:?}: {honest:?} vs {honest_reference:?}"
                );
            }

            let cap_proj = state::section_layer_projection(false, active);
            assert_eq!(cap_proj, active, "projected layer must follow {active:?}");
            let cap_scale = perspective_scale_at_w(w_slice, &cap_proj);
            projected_caps.push(
                cap_vertex_projected_and_world(cap_r3, w_slice, cap_scale, &cap_proj, body_pos).1,
            );
        }

        let moved = projected_caps
            .iter()
            .any(|c| (0..3).any(|k| (c[k] - honest_reference[k]).abs() > 1e-4));
        assert!(
            moved,
            "projected cap must move under at least one active projection; \
             got {projected_caps:?} all equal to honest {honest_reference:?}"
        );
    }

    fn point_to_segment_distance(p: Vec3, s: Vec3, e: Vec3) -> f32 {
        let d = e - s;
        let len_sq = d.length_squared();
        if len_sq < 1e-20 {
            return (p - s).length();
        }
        let t = ((p - s).dot(d) / len_sq).clamp(0.0, 1.0);
        (p - (s + t * d)).length()
    }

    fn cap_projected(cap_r3: [f32; 3], w_slice: f32, proj: &loam_math::Projection<4>) -> Vec3 {
        let scale = perspective_scale_at_w(w_slice, proj);
        cap_vertex_projected_and_world(cap_r3, w_slice, scale, proj, Vec3::ZERO).0
    }

    #[test]
    fn cap_fill_triangle_dropped_near_pole() {
        let r = STEREOGRAPHIC_VIEW_RADIUS;
        let projected = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(r * 2.0, 0.0, 0.0),
        ];
        let mut indices = vec![[0u32, 1, 2], [0u32, 1, 1]];
        retain_in_radius_triangles(&mut indices, 0, 0, &projected, Some(r));
        assert_eq!(
            indices,
            vec![[0u32, 1, 1]],
            "the triangle touching the near-pole vertex must be dropped, the far one kept"
        );

        let all_far = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.0),
        ];
        let mut far_indices = vec![[0u32, 1, 2]];
        retain_in_radius_triangles(&mut far_indices, 0, 0, &all_far, Some(r));
        assert_eq!(
            far_indices,
            vec![[0u32, 1, 2]],
            "a fan entirely within the radius must keep every triangle"
        );

        let mut affine_indices = vec![[0u32, 1, 2]];
        retain_in_radius_triangles(&mut affine_indices, 0, 0, &projected, None);
        assert_eq!(
            affine_indices,
            vec![[0u32, 1, 2]],
            "no clip (affine) must keep every triangle even past the radius"
        );
    }

    #[test]
    fn cap_fill_matches_perimeter_clip() {
        let proj = loam_math::Projection::Stereographic { pole: Vec4::W };
        let clip = stereographic_clip_radius(&proj, STEREOGRAPHIC_VIEW_RADIUS);
        let near = cap_projected([0.05, 0.02, 0.01], 0.999, &proj);
        let far = cap_projected([0.5, 0.0, 0.0], 0.0, &proj);
        let perimeter_keeps_near = sample_in_radius(near, clip);
        let perimeter_keeps_far = sample_in_radius(far, clip);
        assert!(
            !perimeter_keeps_near,
            "near-pole cap projected to {near:?} (|.| = {}) must fail the clip",
            near.length()
        );
        assert!(perimeter_keeps_far, "far cap must pass the clip");
        let projected = [Vec3::ZERO, far, near];
        let mut indices = vec![[0u32, 1, 2]];
        retain_in_radius_triangles(&mut indices, 0, 0, &projected, clip);
        let fill_keeps = !indices.is_empty();
        assert_eq!(
            fill_keeps,
            perimeter_keeps_near && perimeter_keeps_far,
            "fill triangle keep/drop must match the perimeter's endpoint test"
        );
        assert!(!fill_keeps, "the near-pole fan must be dropped");
    }
}
