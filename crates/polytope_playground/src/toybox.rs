//! Floor friction can transfer momentum through w; the demo leaves linear velocity undamped.

use std::borrow::Cow;

use anyhow::{anyhow, Result};
use glam::{Mat4, Vec2, Vec3, Vec4};
use loam_app::{egui, Camera, CameraController, FrameCtx, OrbitController, RenderCtx, SetupCtx};
use loam_camera::Ray;
use loam_egui::{Console, ConsoleUi};
use loam_math::{Bivector4, EuclideanR3, EuclideanR4, Projection, Rotor, Rotor4, WPlane};
use loam_physics::euclidean_r4::{
    halfspace4_body_r4, polytope_body_r4, register_default_narrowphase, regular_polytope4_inertia,
};
use loam_physics::{BodyId, World};
use loam_render::{
    DepthBuffer, DepthMode, LineRasterNode, SkyGroundNode, SkyGroundUniforms, TriangleRasterNode,
    Viewport,
};
use loam_shape::polytope::{polytope_section_faces_append, Polytope4, SectionScratch};
use loam_shape::{LineMesh, TriangleMesh};

use crate::consts::W_SCRUB_RATE;
use crate::physics::ndc_from_pixels;
use crate::projections::WireframeProjection;
use crate::verbs::WireframeControls;
use crate::wireframe_geom::stereographic_view_radius;
use loam_app::environment::{register_floor_command, register_ground_command, Environment};
use loam_shape::projected_edges::push_projected_chord;

const TICK_HZ: u32 = 60;

const TICK_DT: f32 = 1.0 / TICK_HZ as f32;

const BASE_SUBSTEPS: usize = 4;

const MAX_SUBSTEPS: usize = 16;

const MIN_SOLVER_DT: f32 = TICK_DT / MAX_SUBSTEPS as f32;

const GRAVITY: f32 = -9.8;

const FLOOR_Y: f32 = 0.0;

const W_SLICE: f32 = 0.0;

const W_SLICE_RANGE: f32 = 1.5;

const BODY_MASS: f32 = 1.0;

// Circumradius.
const BODY_SIZE: f32 = 0.45;

const RESTITUTION: f32 = 0.05;

const TOYS: [Polytope4; 5] = [
    Polytope4::Cell24,
    Polytope4::Tesseract,
    Polytope4::Pentatope,
    Polytope4::Cell16,
    Polytope4::Tesseract,
];

const SPAWN_SPACING: f32 = 1.4;

const SPAWN_CLEARANCE: f32 = 0.20;

const SETTLED_W_BAND: f32 = 0.06;

// Empirical body-to-body travel budget; this is not a continuous collision bound.
const RESOLVABLE_STEP_TRAVEL: f32 = 0.150;

const TRAVEL_MARGIN: f32 = 0.9;

const MAX_RELEASE_SPEED: f32 = 0.5 * TRAVEL_MARGIN * RESOLVABLE_STEP_TRAVEL / MIN_SOLVER_DT;

const MAX_ANGULAR_SPEED: f32 =
    0.5 * TRAVEL_MARGIN * RESOLVABLE_STEP_TRAVEL / (BODY_SIZE * MIN_SOLVER_DT);

const STEP_TRAVEL_BUDGET: f32 = TRAVEL_MARGIN * RESOLVABLE_STEP_TRAVEL;

const RELEASE_GAIN: f32 = 0.3;

const RELEASE_SPIN_GAIN: f32 = 0.25;

const MAX_CARRY_SPEED: f32 = 20.0;

const MAX_GRAB_ACCEL: f32 = 400.0;

const _: () = assert!(MAX_CARRY_SPEED < MAX_RELEASE_SPEED);

const ANGULAR_DAMPING: f32 = 1.2;

const GRAB_STIFFNESS: f32 = 20.0;

const PICK_TOLERANCE: f32 = 0.1 * BODY_SIZE;

const WIREFRAME_W_FADE: f32 = BODY_SIZE;
const WIREFRAME_MIN_SHADE: f32 = 0.25;

const WIREFRAME_PROJECTION: WireframeProjection = WireframeProjection::WPinhole;

const WAKE_IMPULSE: f32 = 0.05;

const GRAB_TRAIL: usize = 8;

const RELEASE_WINDOW: f32 = 0.08;

const W_PER_RISE: f32 = 1.0;

const PLANE_MIN_COS: f32 = 1e-3;

const FADE_EXTENT: f32 = 0.35 * BODY_SIZE;

const REST_TRAVEL: f32 = 0.02;

const REST_WINDOW: u32 = 30;

const REST_SPEED: f32 = 0.15;
const REST_ANGULAR_SPEED: f32 = 0.3;

const W_LABEL_MIN: f32 = SETTLED_W_BAND;

const ARENA_HALF_EXTENT: f32 = 3.6;

const WALL_RESTITUTION: f32 = 0.4;

const ARENA_OUTLINE_COLOR: [f32; 4] = [0.55, 0.60, 0.68, 1.0];

const ARENA_OUTLINE_WIDTH_PX: f32 = 1.5;

const TOY_COLOR_FALLBACK: [f32; 3] = [0.8, 0.8, 0.8];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum GrabAxis {
    Slice,
    Through,
}

pub(crate) struct ToyBody {
    body: BodyId,
    polytope: Polytope4,
    color: [f32; 3],
    rest_anchor: Vec4,
    rest_time: f32,
}

#[derive(Copy, Clone, Debug)]
struct GrabTrail {
    points: [Vec4; GRAB_TRAIL],
    // Seconds since the preceding pointer sample.
    spans: [f32; GRAB_TRAIL],
    head: usize,
    len: usize,
}

impl GrabTrail {
    fn seeded(point: Vec4) -> Self {
        Self {
            points: [point; GRAB_TRAIL],
            spans: [0.0; GRAB_TRAIL],
            head: 0,
            len: 1,
        }
    }

    fn push(&mut self, point: Vec4, dt: f32) {
        self.head = (self.head + 1) % GRAB_TRAIL;
        self.points[self.head] = point;
        self.spans[self.head] = dt;
        self.len = (self.len + 1).min(GRAB_TRAIL);
    }

    /// Mean velocity over up to [`RELEASE_WINDOW`] seconds of retained samples.
    fn velocity(&self) -> Vec4 {
        let mut span = 0.0;
        let mut index = self.head;
        for _ in 0..self.len - 1 {
            let previous = (index + GRAB_TRAIL - 1) % GRAB_TRAIL;
            span += self.spans[index];
            index = previous;
            if span >= RELEASE_WINDOW {
                break;
            }
        }
        if span <= 0.0 {
            return Vec4::ZERO;
        }
        (self.points[self.head] - self.points[index]) / span
    }
}

struct Grab {
    toy: usize,
    /// Depth along the camera forward axis.
    depth: f32,
    /// Body-local grab point.
    lever_local: Vec4,
    target: Vec4,
    /// Unclamped target for estimating release velocity at arena walls.
    intent: Vec4,
    plane_point: Vec3,
    trail: GrabTrail,
}

/// Held hull bounds in xz and distance from the w slice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DropFootprint {
    min: Vec2,
    max: Vec2,
    from_slice: f32,
}

pub(crate) struct Toybox {
    world: World<EuclideanR4>,
    toys: Vec<ToyBody>,
    grab: Option<Grab>,
    slice: f32,
    tick: u64,
    local_vertices: Vec<Vec4>,
    section_scratch: SectionScratch,
    cap: TriangleMesh<3>,
}

impl Toybox {
    pub(crate) fn new() -> Option<Self> {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        world.gravity = Some(Vec4::new(0.0, GRAVITY, 0.0, 0.0));
        let floor = world.push_body(halfspace4_body_r4(Vec4::Y, FLOOR_Y)?);
        world.bodies[floor].restitution = RESTITUTION;
        for normal in [Vec4::X, -Vec4::X, Vec4::Z, -Vec4::Z] {
            let wall = world.push_body(halfspace4_body_r4(normal, -ARENA_HALF_EXTENT)?);
            world.bodies[wall].restitution = WALL_RESTITUTION;
        }

        let mut toys = Vec::with_capacity(TOYS.len());
        for (index, polytope) in TOYS.into_iter().enumerate() {
            let x = (index as f32 - (TOYS.len() as f32 - 1.0) * 0.5) * SPAWN_SPACING;
            let pose = face_down_pose(polytope, 0);
            let vertices: Vec<Vec4> = polytope
                .topology()
                .vertices
                .iter()
                .map(|v| BODY_SIZE * *v)
                .collect();
            let lowest = (vertices.iter()).fold(f32::INFINITY, |m, v| m.min(pose.apply(*v).y));
            let id = world.push_body(polytope_body_r4(
                Vec4::new(x, SPAWN_CLEARANCE - lowest, 0.0, W_SLICE),
                Vec4::ZERO,
                vertices,
                BODY_MASS,
            )?);
            let body = &mut world.bodies[id];
            body.restitution = RESTITUTION;
            body.orientation.rotation = pose;
            body.inertia = regular_polytope4_inertia(polytope, BODY_MASS, BODY_SIZE);
            toys.push(ToyBody {
                body: id,
                polytope,
                color: toy_color(polytope),
                rest_anchor: world.bodies[id].position,
                rest_time: 0.0,
            });
        }

        Some(Self {
            world,
            toys,
            grab: None,
            slice: W_SLICE,
            tick: 0,
            local_vertices: Vec::new(),
            section_scratch: SectionScratch::default(),
            cap: TriangleMesh::<3>::default(),
        })
    }

    pub(crate) fn slice(&self) -> f32 {
        self.slice
    }

    pub(crate) fn set_slice(&mut self, slice: f32) {
        let reach = self.slice_reach();
        self.slice = slice.clamp(-reach, reach);
    }

    /// Includes the spawn range and every body circumradius.
    pub(crate) fn slice_reach(&self) -> f32 {
        let deepest = (self.toys.iter())
            .map(|toy| self.world.bodies[toy.body].position.w.abs())
            .fold(0.0f32, f32::max);
        (deepest + BODY_SIZE).max(W_SLICE_RANGE)
    }

    pub(crate) fn scrub_slice(&mut self, dir: f32, dt: f32) {
        self.set_slice(self.slice + dir * W_SCRUB_RATE * dt);
    }

    pub(crate) fn position(&self, toy: usize) -> Vec4 {
        self.world.bodies[self.toys[toy].body].position
    }

    pub(crate) fn w_offsets(&self) -> impl Iterator<Item = (usize, f32)> + '_ {
        let slice = self.slice;
        (self.toys.iter().enumerate())
            .map(move |(index, toy)| (index, self.world.bodies[toy.body].position.w - slice))
    }

    pub(crate) fn grab_handle(&self) -> Option<Vec3> {
        let grab = self.grab.as_ref()?;
        let body = &self.world.bodies[self.toys[grab.toy].body];
        Some((body.position + body.orientation.rotation.apply(grab.lever_local)).truncate())
    }

    pub(crate) fn drop_footprint(&self) -> Option<DropFootprint> {
        let grab = self.grab.as_ref()?;
        let toy = &self.toys[grab.toy];
        let body = &self.world.bodies[toy.body];
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for vertex in toy.polytope.topology().vertices.iter() {
            let posed = body.position + BODY_SIZE * body.orientation.rotation.apply(*vertex);
            let xz = Vec2::new(posed.x, posed.z);
            min = min.min(xz);
            max = max.max(xz);
        }
        Some(DropFootprint {
            min,
            max,
            from_slice: (body.position.w - self.slice).abs(),
        })
    }

    pub(crate) fn slice_marks(&self) -> impl Iterator<Item = SliceMark> + '_ {
        self.toys.iter().map(move |toy| SliceMark {
            w: self.world.bodies[toy.body].position.w,
            color: toy.color,
            asleep: self.world.bodies[toy.body].is_sleeping(),
        })
    }

    fn substeps_for_current_speed(&self, dt: f32) -> usize {
        let fastest = self
            .toys
            .iter()
            .map(|toy| {
                let body = &self.world.bodies[toy.body];
                body.velocity.length() + body.angular_velocity.magnitude() * BODY_SIZE
            })
            .fold(0.0f32, f32::max);
        let needed = (fastest * dt / STEP_TRAVEL_BUDGET).ceil() as usize;
        let baseline = (dt / (TICK_DT / BASE_SUBSTEPS as f32)).ceil() as usize;
        let maximum = (dt / MIN_SOLVER_DT).ceil() as usize;
        needed.clamp(baseline.max(1), maximum.max(1))
    }

    pub(crate) fn tick(&mut self, tick_dt: f32) {
        if let Some(grab) = &self.grab {
            let body = &mut self.world.bodies[self.toys[grab.toy].body];
            let desired = clamp_length(
                (grab.target - body.position) * GRAB_STIFFNESS,
                MAX_CARRY_SPEED,
            );
            body.velocity += clamp_length(desired - body.velocity, MAX_GRAB_ACCEL * tick_dt);
        }
        let substeps = self.substeps_for_current_speed(tick_dt);
        let dt = tick_dt / substeps as f32;
        let decay = (-ANGULAR_DAMPING * dt).exp();
        for _ in 0..substeps {
            self.world.step(dt);
            for toy in &self.toys {
                let body = &mut self.world.bodies[toy.body];
                body.angular_velocity = body.angular_velocity * decay;
            }
            // Wake before the next substep so the remaining contact solve includes the struck body.
            self.wake_on_contact();
        }
        self.latch_parked_bodies(tick_dt);
        self.tick += 1;
    }

    fn latch_parked_bodies(&mut self, dt: f32) {
        let held = self.grab.as_ref().map(|g| g.toy);
        for (index, toy) in self.toys.iter_mut().enumerate() {
            if held == Some(index) {
                toy.rest_time = 0.0;
                toy.rest_anchor = self.world.bodies[toy.body].position;
                continue;
            }
            let body = &mut self.world.bodies[toy.body];
            let travelled = (body.position - toy.rest_anchor).length();
            let moving = body.velocity.length() > REST_SPEED
                || body.angular_velocity.magnitude() > REST_ANGULAR_SPEED;
            if travelled > REST_TRAVEL || moving {
                toy.rest_anchor = body.position;
                toy.rest_time = 0.0;
                continue;
            }
            toy.rest_time += dt;
            if toy.rest_time >= REST_WINDOW as f32 * TICK_DT {
                body.sleep();
            }
        }
    }

    fn wake(&mut self, toy: usize) {
        let body = &mut self.world.bodies[self.toys[toy].body];
        body.wake();
        let position = body.position;
        let toy = &mut self.toys[toy];
        toy.rest_anchor = position;
        toy.rest_time = 0.0;
    }

    fn wake_on_contact(&mut self) {
        let mut hit = [false; TOYS.len()];
        for manifold in self.world.manifolds.values() {
            let deepest = (manifold.points.iter())
                .map(|p| p.normal_impulse)
                .fold(0.0f32, f32::max);
            if deepest <= WAKE_IMPULSE {
                continue;
            }
            for (index, toy) in self.toys.iter().enumerate() {
                if manifold.body_a == toy.body || manifold.body_b == toy.body {
                    hit[index] = true;
                }
            }
        }
        for (index, woken) in hit.iter().enumerate() {
            if *woken && self.world.bodies[self.toys[index].body].is_sleeping() {
                self.wake(index);
            }
        }
    }

    pub(crate) fn press(&mut self, ray: &Ray, forward: Vec3) -> bool {
        self.grab = None;
        let Some((toy, hit)) = self.pick(ray) else {
            return false;
        };
        self.wake(toy);
        let body = &mut self.world.bodies[self.toys[toy].body];
        body.velocity = Vec4::ZERO;
        body.angular_velocity = Bivector4::ZERO;
        // Use the body's w coordinate to avoid a spurious grab lever through the slice.
        let grabbed = Vec4::new(hit.x, hit.y, hit.z, body.position.w);
        let lever_local = body
            .orientation
            .rotation
            .inverse()
            .apply(grabbed - body.position);
        let depth = (hit - ray.origin).dot(forward);
        self.grab = Some(Grab {
            toy,
            depth,
            lever_local,
            target: body.position,
            intent: body.position,
            plane_point: hit,
            trail: GrabTrail::seeded(body.position),
        });
        true
    }

    pub(crate) fn hold(&mut self, ray: &Ray, forward: Vec3, axis: GrabAxis, dt: f32) {
        let Some(grab) = self.grab.as_mut() else {
            return;
        };
        if let Some(point) = plane_point(ray, forward, grab.depth) {
            let delta = point - grab.plane_point;
            grab.target = clamp_target_to_arena(advance_target(grab.target, delta, axis));
            grab.intent = advance_target(grab.intent, delta, axis);
            grab.plane_point = point;
        }
        grab.trail.push(grab.intent, dt);
    }

    pub(crate) fn release(&mut self) {
        let Some(grab) = self.grab.take() else {
            return;
        };
        let velocity = clamp_length(grab.trail.velocity() * RELEASE_GAIN, MAX_RELEASE_SPEED);
        self.wake(grab.toy);
        let body = &mut self.world.bodies[self.toys[grab.toy].body];
        // Replace the hold velocity before applying the release impulse.
        body.velocity = Vec4::ZERO;
        let point = body.position + body.orientation.rotation.apply(grab.lever_local);
        let spin_before = body.angular_velocity;
        body.apply_impulse_at_point(&EuclideanR4, velocity * body.mass(), point);
        body.angular_velocity =
            body.angular_velocity * RELEASE_SPIN_GAIN + spin_before * (1.0 - RELEASE_SPIN_GAIN);
        let angular = body.angular_velocity.magnitude();
        if angular > MAX_ANGULAR_SPEED {
            body.angular_velocity = body.angular_velocity * (MAX_ANGULAR_SPEED / angular);
        }
    }

    pub(crate) fn throw(&mut self, toy: usize, velocity: Vec4) {
        if self.grab.as_ref().is_some_and(|grab| grab.toy == toy) {
            self.grab = None;
        }
        self.wake(toy);
        let body = &mut self.world.bodies[self.toys[toy].body];
        body.velocity = clamp_length(velocity, MAX_RELEASE_SPEED);
    }

    /// Picks visible caps with rim tolerance; invisible bodies use their bounding spheres.
    pub(crate) fn pick(&mut self, ray: &Ray) -> Option<(usize, Vec3)> {
        let mut nearest: Option<(usize, f32)> = None;
        let mut invisible = [false; TOYS.len()];
        let mut extents = [0.0f32; TOYS.len()];
        let mut cap = std::mem::take(&mut self.cap);
        let mut local = std::mem::take(&mut self.local_vertices);
        for (toy, hidden) in invisible.iter_mut().enumerate() {
            let extent = append_cap(
                &self.world,
                &self.toys[toy],
                self.slice,
                &mut local,
                &mut self.section_scratch,
                &mut cap,
            );
            *hidden = extent <= 0.0;
            extents[toy] = extent;
            for triangle in &cap.indices {
                let [a, b, c] = triangle.map(|i| Vec3::from_array(cap.vertices[i as usize]));
                let Some(distance) = ray.intersect_triangle(a, b, c) else {
                    continue;
                };
                if nearest.is_none_or(|(_, best)| distance < best) {
                    nearest = Some((toy, distance));
                }
            }
        }
        self.cap = cap;
        self.local_vertices = local;
        if nearest.is_none() {
            for (toy, hidden) in invisible.iter().enumerate() {
                let radius = if *hidden {
                    BODY_SIZE
                } else {
                    extents[toy] + PICK_TOLERANCE
                };
                let centre = self.world.bodies[self.toys[toy].body].position.truncate();
                let Some(distance) = ray.intersect_sphere(centre, radius) else {
                    continue;
                };
                if nearest.is_none_or(|(_, best)| distance < best) {
                    nearest = Some((toy, distance));
                }
            }
        }
        nearest.map(|(toy, distance)| (toy, ray.origin + ray.direction * distance))
    }

    /// Translucent caps must not write depth over the background.
    pub(crate) fn build_frame_meshes(
        &mut self,
        opaque: &mut TriangleMesh<3>,
        faded: &mut TriangleMesh<3>,
    ) {
        clear_mesh(opaque);
        clear_mesh(faded);
        let mut cap = std::mem::take(&mut self.cap);
        let mut local = std::mem::take(&mut self.local_vertices);
        for toy in &self.toys {
            let extent = append_cap(
                &self.world,
                toy,
                self.slice,
                &mut local,
                &mut self.section_scratch,
                &mut cap,
            );
            let alpha = section_alpha(extent);
            for color in &mut cap.colors {
                color[3] = alpha;
            }
            append_mesh(if alpha >= 1.0 { opaque } else { faded }, &cap);
        }
        self.cap = cap;
        self.local_vertices = local;
    }

    fn build_overlay_mesh(&self, overlay: &PhysicsOverlay, mesh: &mut LineMesh<3>) {
        build_physics_overlay_mesh(&self.world, BODY_SIZE, overlay, mesh);
    }
}

#[cfg(test)]
impl Toybox {
    fn toys(&self) -> &[ToyBody] {
        &self.toys
    }

    fn velocity(&self, toy: usize) -> Vec4 {
        self.world.bodies[self.toys[toy].body].velocity
    }

    fn angular_velocity(&self, toy: usize) -> Bivector4 {
        self.world.bodies[self.toys[toy].body].angular_velocity
    }

    fn cap_stats(&mut self, toy: usize) -> (f32, usize) {
        let mut cap = std::mem::take(&mut self.cap);
        let mut local = std::mem::take(&mut self.local_vertices);
        let extent = append_cap(
            &self.world,
            &self.toys[toy],
            self.slice,
            &mut local,
            &mut self.section_scratch,
            &mut cap,
        );
        let stats = (section_alpha(extent), cap.vertices.len());
        self.cap = cap;
        self.local_vertices = local;
        stats
    }

    fn run(&mut self, ticks: usize) {
        for _ in 0..ticks {
            self.tick(TICK_DT);
        }
    }

    fn fastest_step_travel(&self) -> f32 {
        let dt = TICK_DT / self.substeps_for_current_speed(TICK_DT) as f32;
        (self.world.bodies.iter())
            .map(|b| (b.velocity.length() + b.angular_velocity.magnitude() * BODY_SIZE) * dt)
            .fold(0.0, f32::max)
    }

    fn deepest_point(&self) -> f32 {
        let mut deepest = f32::INFINITY;
        for body in self.world.bodies.iter() {
            let loam_physics::Collider::ConvexPolytope4D { vertices } = body.collider() else {
                continue;
            };
            for v in vertices {
                deepest = deepest.min(body.orientation.rotation.apply(*v).y + body.position.y);
            }
        }
        deepest
    }
}

/// Aligns the selected facet outward normal with -y.
fn face_down_pose(polytope: Polytope4, cell: usize) -> Rotor4 {
    let topology = polytope.topology();
    let indices = topology.cells[cell];
    let mut centroid = Vec4::ZERO;
    for index in indices {
        centroid += topology.vertices[*index as usize];
    }
    Rotor4::from_rotation_arc(centroid.normalize(), -Vec4::Y)
}

fn toy_color(polytope: Polytope4) -> [f32; 3] {
    (crate::catalog::SHAPE_CATALOG.iter())
        .find(|entry| entry.shape.polytope4() == Some(polytope))
        .map(|entry| entry.body_color)
        .unwrap_or(TOY_COLOR_FALLBACK)
}

fn clamp_target_to_arena(target: Vec4) -> Vec4 {
    let reach = ARENA_HALF_EXTENT - BODY_SIZE;
    Vec4::new(
        target.x.clamp(-reach, reach),
        target.y.max(FLOOR_Y + BODY_SIZE),
        target.z.clamp(-reach, reach),
        target.w,
    )
}

fn clamp_length(v: Vec4, ceiling: f32) -> Vec4 {
    let length = v.length();
    if length > ceiling {
        v * (ceiling / length)
    } else {
        v
    }
}

pub(crate) struct SliceMark {
    pub(crate) w: f32,
    pub(crate) color: [f32; 3],
    pub(crate) asleep: bool,
}

// Circumradius bound on the hull's w extent.
const IN_SLICE_HALF_WIDTH: f32 = BODY_SIZE;

const RULER_HEIGHT: f32 = 34.0;

/// Returns the dragged slice position; the caller clamps it to the reachable range.
fn draw_slice_ruler(
    ui: &mut egui::Ui,
    slice: f32,
    reach: f32,
    marks: impl Iterator<Item = SliceMark>,
) -> Option<f32> {
    let width = ui.available_width();
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(width, RULER_HEIGHT),
        egui::Sense::click_and_drag(),
    );
    let painter = ui.painter();
    painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(20, 22, 28));

    let x_for = |w: f32| {
        let t = (w / reach).clamp(-1.0, 1.0) * 0.5 + 0.5;
        rect.left() + t * rect.width()
    };
    painter.rect_filled(
        egui::Rect::from_x_y_ranges(
            x_for(slice - IN_SLICE_HALF_WIDTH)..=x_for(slice + IN_SLICE_HALF_WIDTH),
            rect.y_range(),
        ),
        0.0,
        egui::Color32::from_rgb(32, 38, 50),
    );
    let centre = x_for(slice);
    painter.line_segment(
        [
            egui::pos2(centre, rect.top()),
            egui::pos2(centre, rect.bottom()),
        ],
        egui::Stroke::new(1.0, egui::Color32::from_rgb(150, 170, 200)),
    );

    for mark in marks {
        let dim = if mark.asleep { 0.45 } else { 1.0 };
        let channel = |c: f32| (c * dim * 255.0).clamp(0.0, 255.0) as u8;
        let x = x_for(mark.w);
        painter.line_segment(
            [
                egui::pos2(x, rect.top() + 4.0),
                egui::pos2(x, rect.bottom() - 4.0),
            ],
            egui::Stroke::new(
                3.0,
                egui::Color32::from_rgb(
                    channel(mark.color[0]),
                    channel(mark.color[1]),
                    channel(mark.color[2]),
                ),
            ),
        );
    }

    let pointer = response.interact_pointer_pos()?;
    Some(slice_for_ruler_x(
        pointer.x,
        rect.left(),
        rect.width(),
        reach,
    ))
}

fn slice_for_ruler_x(x: f32, left: f32, width: f32, reach: f32) -> f32 {
    let t = ((x - left) / width.max(f32::MIN_POSITIVE)).clamp(0.0, 1.0);
    (t * 2.0 - 1.0) * reach
}

pub(crate) fn plane_point(ray: &Ray, forward: Vec3, depth: f32) -> Option<Vec3> {
    let along = ray.direction.dot(forward);
    if along.abs() < PLANE_MIN_COS {
        return None;
    }
    Some(ray.origin + ray.direction * (depth / along))
}

pub(crate) fn advance_target(target: Vec4, delta: Vec3, axis: GrabAxis) -> Vec4 {
    match axis {
        GrabAxis::Slice => target + Vec4::new(delta.x, delta.y, delta.z, 0.0),
        GrabAxis::Through => target + Vec4::new(delta.x, 0.0, delta.z, delta.y * W_PER_RISE),
    }
}

pub(crate) fn section_alpha(extent: f32) -> f32 {
    (extent / FADE_EXTENT).clamp(0.0, 1.0)
}

fn append_toy_wireframe(
    world: &World<EuclideanR4>,
    toys: &[ToyBody],
    slice: f32,
    controls: &WireframeControls,
    mesh: &mut LineMesh<3>,
    posed: &mut Vec<Vec4>,
) {
    let projection = controls.projection.to_projection();
    for toy in toys {
        let body = &world.bodies[toy.body];
        let topology = toy.polytope.topology();
        let view_radius = stereographic_view_radius(BOOT_ORBIT_DISTANCE);
        let body_pos_r3 = body.position.truncate();
        // Project around each body before translating it into the row.
        posed.clear();
        posed.extend(
            topology
                .vertices
                .iter()
                .map(|v| BODY_SIZE * body.orientation.rotation.apply(*v)),
        );
        let [r, g, b] = toy.color;
        let shade = |local: Vec4| {
            let from_slice = (local.w + body.position.w - slice).abs();
            let near = 1.0 - (from_slice / WIREFRAME_W_FADE).clamp(0.0, 1.0);
            let lit = WIREFRAME_MIN_SHADE + (1.0 - WIREFRAME_MIN_SHADE) * near;
            [r * lit, g * lit, b * lit, controls.alpha]
        };
        for edge in topology.edges {
            let (a, b4) = (posed[edge[0] as usize], posed[edge[1] as usize]);
            push_projected_chord(
                mesh,
                a,
                b4,
                shade(a),
                shade(b4),
                controls.width_px,
                &projection,
                body_pos_r3,
                view_radius,
                crate::consts::SPACE_TESSELLATION_SAMPLES,
            );
        }
    }
}

fn append_arena_outline(mesh: &mut LineMesh<3>) {
    let e = ARENA_HALF_EXTENT;
    let corners = [
        Vec3::new(-e, FLOOR_Y, -e),
        Vec3::new(e, FLOOR_Y, -e),
        Vec3::new(e, FLOOR_Y, e),
        Vec3::new(-e, FLOOR_Y, e),
    ];
    for (index, from) in corners.iter().enumerate() {
        push_overlay_segment(
            mesh,
            *from,
            corners[(index + 1) % corners.len()],
            ARENA_OUTLINE_COLOR,
            ARENA_OUTLINE_COLOR,
            ARENA_OUTLINE_WIDTH_PX,
        );
    }
}

fn append_drop_footprint(mesh: &mut LineMesh<3>, footprint: &DropFootprint) {
    let near = 1.0 - (footprint.from_slice / WIREFRAME_W_FADE).clamp(0.0, 1.0);
    let lit = WIREFRAME_MIN_SHADE + (1.0 - WIREFRAME_MIN_SHADE) * near;
    let [r, g, b, a] = GRAB_HANDLE_COLOR;
    let color = [r * lit, g * lit, b * lit, a];
    let (lo, hi) = (footprint.min, footprint.max);
    let corners = [
        Vec3::new(lo.x, FLOOR_Y, lo.y),
        Vec3::new(hi.x, FLOOR_Y, lo.y),
        Vec3::new(hi.x, FLOOR_Y, hi.y),
        Vec3::new(lo.x, FLOOR_Y, hi.y),
    ];
    for (index, from) in corners.iter().enumerate() {
        push_overlay_segment(
            mesh,
            *from,
            corners[(index + 1) % corners.len()],
            color,
            color,
            ARENA_OUTLINE_WIDTH_PX,
        );
    }
}

fn clear_mesh(mesh: &mut TriangleMesh<3>) {
    mesh.vertices.clear();
    mesh.colors.clear();
    mesh.indices.clear();
}

fn append_mesh(dst: &mut TriangleMesh<3>, src: &TriangleMesh<3>) {
    let base = dst.vertices.len() as u32;
    dst.vertices.extend_from_slice(&src.vertices);
    dst.colors.extend_from_slice(&src.colors);
    dst.indices
        .extend(src.indices.iter().map(|t| t.map(|i| i + base)));
}

/// Returns the cap circumradius about its vertex centroid.
fn append_cap(
    world: &World<EuclideanR4>,
    toy: &ToyBody,
    slice: f32,
    local: &mut Vec<Vec4>,
    scratch: &mut SectionScratch,
    cap: &mut TriangleMesh<3>,
) -> f32 {
    clear_mesh(cap);
    let body = &world.bodies[toy.body];
    let topology = toy.polytope.topology();
    local.clear();
    local.extend(
        (topology.vertices.iter())
            .map(|v| BODY_SIZE * body.orientation.rotation.apply(*v) + Vec4::W * body.position.w),
    );
    let [r, g, b] = toy.color;
    polytope_section_faces_append(
        topology.edges,
        topology.cells,
        local,
        WPlane::new(slice),
        [r, g, b, 1.0],
        scratch,
        cap,
    );
    let extent = cap_extent(&cap.vertices);
    let translate = body.position.truncate();
    for v in &mut cap.vertices {
        v[0] += translate.x;
        v[1] += translate.y;
        v[2] += translate.z;
    }
    extent
}

fn cap_extent(vertices: &[[f32; 3]]) -> f32 {
    if vertices.is_empty() {
        return 0.0;
    }
    let mut centroid = Vec3::ZERO;
    for v in vertices {
        centroid += Vec3::from_array(*v);
    }
    centroid /= vertices.len() as f32;
    (vertices.iter())
        .map(|v| (Vec3::from_array(*v) - centroid).length())
        .fold(0.0, f32::max)
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct PhysicsOverlay {
    contacts: bool,
    normals: bool,
    impulses: bool,
    islands: bool,
    impulse_scale: f32,
    width_px: f32,
}

const DEFAULT_IMPULSE_SCALE: f32 = 0.034;

const CONTACT_CROSS_FRACTION: f32 = 0.15;

const GRAB_HANDLE_FRACTION: f32 = 0.30;
const GRAB_HANDLE_WIDTH: f32 = 2.5;
const GRAB_HANDLE_COLOR: [f32; 4] = [1.0, 0.95, 0.55, 1.0];

const NORMAL_LEN_FRACTION: f32 = 0.9;

const ISLAND_CROSS_FRACTION: f32 = 1.0;

const CONTACT_COLOR: [f32; 4] = [1.00, 0.95, 0.35, 1.0];
const NORMAL_TAIL_COLOR: [f32; 4] = [0.06, 0.24, 0.42, 1.0];
const NORMAL_TIP_COLOR: [f32; 4] = [0.40, 0.95, 1.00, 1.0];
const NORMAL_IMPULSE_COLOR: [f32; 4] = [1.00, 0.30, 0.22, 1.0];
const TANGENT_IMPULSE_COLOR: [f32; 4] = [0.70, 0.40, 1.00, 1.0];

const ISLAND_PALETTE: [[f32; 4]; 6] = [
    [0.35, 0.85, 0.45, 1.0],
    [0.95, 0.55, 0.20, 1.0],
    [0.45, 0.60, 1.00, 1.0],
    [0.95, 0.40, 0.75, 1.0],
    [0.90, 0.90, 0.35, 1.0],
    [0.35, 0.90, 0.90, 1.0],
];

impl Default for PhysicsOverlay {
    fn default() -> Self {
        Self {
            contacts: false,
            normals: false,
            impulses: false,
            islands: false,
            impulse_scale: DEFAULT_IMPULSE_SCALE,
            width_px: 2.0,
        }
    }
}

impl PhysicsOverlay {
    fn any_layer(self) -> bool {
        self.contacts || self.normals || self.impulses || self.islands
    }
}

fn push_overlay_segment(
    mesh: &mut LineMesh<3>,
    from: Vec3,
    to: Vec3,
    from_color: [f32; 4],
    to_color: [f32; 4],
    width: f32,
) {
    mesh.segments.push((from.to_array(), to.to_array()));
    mesh.colors.push((from_color, to_color));
    mesh.widths.push(width);
}

fn push_axis_cross(mesh: &mut LineMesh<3>, centre: Vec3, half: f32, color: [f32; 4], width: f32) {
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        push_overlay_segment(
            mesh,
            centre - axis * half,
            centre + axis * half,
            color,
            color,
            width,
        );
    }
}

fn build_physics_overlay_mesh(
    world: &World<EuclideanR4>,
    radius: f32,
    overlay: &PhysicsOverlay,
    mesh: &mut LineMesh<3>,
) {
    mesh.segments.clear();
    mesh.colors.clear();
    mesh.widths.clear();
    if !overlay.any_layer() {
        return;
    }

    let width = overlay.width_px;

    if overlay.contacts || overlay.normals || overlay.impulses {
        for manifold in world.manifolds.values() {
            for cp in &manifold.points {
                let point = cp.world_point.truncate();
                let normal = cp.normal.truncate();
                if overlay.contacts {
                    push_axis_cross(
                        mesh,
                        point,
                        CONTACT_CROSS_FRACTION * radius,
                        CONTACT_COLOR,
                        width,
                    );
                }
                if overlay.normals {
                    push_overlay_segment(
                        mesh,
                        point,
                        point + normal * (NORMAL_LEN_FRACTION * radius),
                        NORMAL_TAIL_COLOR,
                        NORMAL_TIP_COLOR,
                        width,
                    );
                }
                if overlay.impulses {
                    push_overlay_segment(
                        mesh,
                        point,
                        point + normal * (cp.normal_impulse * overlay.impulse_scale),
                        NORMAL_IMPULSE_COLOR,
                        NORMAL_IMPULSE_COLOR,
                        width,
                    );
                    // The displayed tangent impulse acts on body B.
                    push_overlay_segment(
                        mesh,
                        point,
                        point
                            - cp.tangent_dir.truncate()
                                * (cp.tangent_impulse * overlay.impulse_scale),
                        TANGENT_IMPULSE_COLOR,
                        TANGENT_IMPULSE_COLOR,
                        width,
                    );
                }
            }
        }
    }

    if overlay.islands {
        for (ordinal, island) in world.islands().iter().enumerate() {
            let color = ISLAND_PALETTE[ordinal % ISLAND_PALETTE.len()];
            for &id in &island.bodies {
                push_axis_cross(
                    mesh,
                    world.bodies[id].position.truncate(),
                    ISLAND_CROSS_FRACTION * radius,
                    color,
                    width,
                );
            }
            for &(a, b) in &island.constraints {
                if world.bodies[a].inv_mass() == 0.0 || world.bodies[b].inv_mass() == 0.0 {
                    continue;
                }
                push_overlay_segment(
                    mesh,
                    world.bodies[a].position.truncate(),
                    world.bodies[b].position.truncate(),
                    color,
                    color,
                    width,
                );
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct ToyboxControls {
    overlay: PhysicsOverlay,
    wireframe: WireframeControls,
    w_labels: bool,
    environment: Environment,
    rig: loam_app::camera_rig::CameraRig,
    /// Applied at the next simulation tick.
    pending_throws: Vec<(usize, Vec4)>,
}

impl Default for ToyboxControls {
    fn default() -> Self {
        Self {
            overlay: PhysicsOverlay::default(),
            wireframe: WireframeControls {
                projection: WIREFRAME_PROJECTION,
                ..WireframeControls::default()
            },
            w_labels: false,
            environment: Environment::default(),
            rig: loam_app::camera_rig::CameraRig::default(),
            pending_throws: Vec::new(),
        }
    }
}

fn parse_throw(args: &[&str]) -> Result<(usize, Vec4)> {
    let usage = "usage: throw <toy> <vx> <vy> <vz> [vw]";
    let (toy, rest) = args.split_first().ok_or_else(|| anyhow!("{usage}"))?;
    let toy: usize = toy
        .parse()
        .map_err(|_| anyhow!("throw: `{toy}` is not a toy index ({usage})"))?;
    if toy >= TOYS.len() {
        return Err(anyhow!(
            "throw: toy {toy} does not exist (0..{})",
            TOYS.len()
        ));
    }
    if !(3..=4).contains(&rest.len()) {
        return Err(anyhow!("{usage}"));
    }
    let mut velocity = [0.0f32; 4];
    for (slot, text) in velocity.iter_mut().zip(rest) {
        *slot = text
            .parse()
            .map_err(|_| anyhow!("throw: `{text}` is not a number ({usage})"))?;
    }
    let velocity = Vec4::from_array(velocity);
    if !velocity.is_finite() {
        return Err(anyhow!("throw: velocity must be finite"));
    }
    Ok((toy, velocity))
}

pub(crate) fn register_toybox_commands(
    console: &mut Console<ToyboxControls>,
    runtime: &loam_app::Runtime,
    control: &loam_app::shell::SceneControl,
) {
    loam_app::shell::register_shell_commands::<ToyboxControls, crate::shell::Playground>(
        console,
        loam_app::build_info!(),
        runtime,
        control,
    );
    register_ground_command(console, |c| &mut c.environment);
    register_floor_command(console, |c| &mut c.environment);
    loam_app::camera_rig::register_camera_command(console, |c| &mut c.rig);
    console.register(loam_egui::cmd::<ToyboxControls, _>(
        "throw",
        "throw <toy> <vx> <vy> <vz> [vw]: set a toy's velocity in world units per second",
        |args, controls, out| {
            let (toy, velocity) = parse_throw(args)?;
            controls.pending_throws.push((toy, velocity));
            out.line(format!("throw: toy {toy} at {velocity}"));
            Ok(())
        },
    ));
    console.register(
        loam_egui::cmd::<ToyboxControls, _>(
            "wlabels",
            "per-body w offset labels over each toy (on | off; bare flips)",
            |args, controls, out| {
                let next = match args.first().copied() {
                    None => !controls.w_labels,
                    Some("on") => true,
                    Some("off") => false,
                    Some(other) => {
                        return Err(anyhow!("wlabels: unknown arg `{other}` (try on|off)"));
                    }
                };
                controls.w_labels = next;
                out.line(format!("wlabels: {}", if next { "on" } else { "off" }));
                Ok(())
            },
        )
        .with_args(&[&["on", "off"]]),
    );
    console.register(crate::verbs::wireframe_subcommands::<ToyboxControls>(|c| {
        &mut c.wireframe
    }));
    console.register(
        loam_egui::subcommands::<ToyboxControls>(
            "physics",
            "solver debug overlay (bare flips all four layers)",
        )
        .on_bare(|c: &mut ToyboxControls| {
            let o = &mut c.overlay;
            let on = !o.any_layer();
            o.contacts = on;
            o.normals = on;
            o.impulses = on;
            o.islands = on;
            Ok(())
        })
        .toggle(
            "contacts",
            "axis cross at each contact point (bare flips)",
            |c: &mut ToyboxControls, v| {
                let o = &mut c.overlay;
                o.contacts = v.unwrap_or(!o.contacts);
                Ok(())
            },
        )
        .toggle(
            "normals",
            "contact normal, drawn dark-to-bright along the A-toward-B direction (bare flips)",
            |c: &mut ToyboxControls, v| {
                let o = &mut c.overlay;
                o.normals = v.unwrap_or(!o.normals);
                Ok(())
            },
        )
        .toggle(
            "impulses",
            "accumulated normal + tangent impulse bars (bare flips)",
            |c: &mut ToyboxControls, v| {
                let o = &mut c.overlay;
                o.impulses = v.unwrap_or(!o.impulses);
                Ok(())
            },
        )
        .toggle(
            "islands",
            "colour each island's bodies and coupling constraints (bare flips)",
            |c: &mut ToyboxControls, v| {
                let o = &mut c.overlay;
                o.islands = v.unwrap_or(!o.islands);
                Ok(())
            },
        )
        .custom(
            "impulse-scale",
            "world units of bar length per unit of accumulated impulse (default 0.034)",
            &[&[]],
            &[],
            |c: &mut ToyboxControls, args, out| {
                let o = &mut c.overlay;
                match args.first().copied() {
                    None => out.line(format!("physics impulse-scale: {:.4}", o.impulse_scale)),
                    Some(token) => {
                        let s: f32 = token
                            .parse()
                            .map_err(|e| anyhow!("invalid impulse scale `{token}`: {e}"))?;
                        if !(s.is_finite() && s > 0.0 && s <= 100.0) {
                            return Err(anyhow!(
                                "impulse scale {s} out of range; expected a float in (0, 100]"
                            ));
                        }
                        o.impulse_scale = s;
                        out.line(format!("physics impulse-scale: set to {s:.4}"));
                    }
                }
                Ok(())
            },
        )
        .custom(
            "width",
            "overlay line thickness in pixels (default 2.0)",
            &[&[]],
            &[],
            |c: &mut ToyboxControls, args, out| {
                let o = &mut c.overlay;
                match args.first().copied() {
                    None => out.line(format!("physics width: {:.2} px", o.width_px)),
                    Some(token) => {
                        let w: f32 = token
                            .parse()
                            .map_err(|e| anyhow!("invalid width `{token}`: {e}"))?;
                        if !(w > 0.0 && w <= 16.0) {
                            return Err(anyhow!(
                                "physics width {w} out of range; expected a float in (0, 16]"
                            ));
                        }
                        o.width_px = w;
                        out.line(format!("physics width: set to {w:.2} px"));
                    }
                }
                Ok(())
            },
        ),
    );
}

// Depth24 loses separation between densely stacked 24-cell caps.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const BOOT_ORBIT_DISTANCE: f32 = 9.0;
const BOOT_ORBIT_PITCH: f32 = -0.16;
const BOOT_TARGET_HEIGHT: f32 = 0.7;

const CAMERA_FOV_DEG: f32 = 55.0;
const CAMERA_NEAR: f32 = 0.05;
const CAMERA_FAR: f32 = 200.0;

fn boot_orbit() -> OrbitController<EuclideanR3> {
    let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
    orbit.set_orbit(BOOT_ORBIT_DISTANCE, BOOT_ORBIT_PITCH);
    orbit.target.y = BOOT_TARGET_HEIGHT;
    orbit
}

fn build_caps(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    samples: u32,
    depth: DepthMode,
) -> TriangleRasterNode {
    TriangleRasterNode::new(
        device,
        format,
        depth,
        loam_render::triangle_raster::FragmentShading::FaceNormalLambert,
        samples,
    )
}

enum ToyboxCommand {
    Press(Ray, Vec3),
    Hold(Ray, Vec3, GrabAxis, f32),
    Release,
    Slice(f32),
    Respawn,
}

impl Toybox {
    fn apply_command(&mut self, command: ToyboxCommand) {
        match command {
            ToyboxCommand::Press(ray, forward) => {
                self.press(&ray, forward);
            }
            ToyboxCommand::Hold(ray, forward, axis, dt) => self.hold(&ray, forward, axis, dt),
            ToyboxCommand::Release => self.release(),
            ToyboxCommand::Slice(slice) => self.set_slice(slice),
            ToyboxCommand::Respawn => {
                if let Some(fresh) = Self::new() {
                    *self = fresh;
                } else {
                    tracing::error!("invalid Toybox body configuration");
                }
            }
        }
    }
}

pub(crate) struct ToyboxScene {
    toybox: Toybox,
    pending: Vec<ToyboxCommand>,
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    console: Console<ToyboxControls>,
    caps: TriangleRasterNode,
    faded_caps: TriangleRasterNode,
    sky_ground: SkyGroundNode,
    depth: Option<DepthBuffer>,
    opaque_mesh: TriangleMesh<3>,
    faded_mesh: TriangleMesh<3>,
    controls: ToyboxControls,
    line_node: LineRasterNode,
    /// Arena guides use depth; contact diagnostics remain visible through bodies.
    arena_node: LineRasterNode,
    arena_mesh: LineMesh<3>,
    line_mesh: LineMesh<3>,
    left_was_down: bool,
    pointer_was_captured: bool,
    slice_up_held: bool,
    slice_down_held: bool,
    paused: bool,
}

impl ToyboxScene {
    pub(crate) fn new(
        ctx: &mut SetupCtx<'_>,
        control: &loam_app::shell::SceneControl,
    ) -> Result<Self> {
        let mut console = Console::<ToyboxControls>::new();
        register_toybox_commands(&mut console, ctx.runtime, control);

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 2.0, BOOT_ORBIT_DISTANCE);
        camera.fov_y = CAMERA_FOV_DEG.to_radians();
        camera.near = CAMERA_NEAR;
        camera.far = CAMERA_FAR;
        let orbit = boot_orbit();

        Ok(Self {
            toybox: Toybox::new()
                .ok_or_else(|| anyhow::anyhow!("invalid Toybox body configuration"))?,
            pending: Vec::with_capacity(32),
            camera,
            orbit,
            console,
            caps: build_caps(
                &ctx.rd.device,
                ctx.rd.target_format(),
                ctx.rd.sample_count(),
                DepthMode::ReadWrite {
                    format: DEPTH_FORMAT,
                },
            ),
            faded_caps: build_caps(
                &ctx.rd.device,
                ctx.rd.target_format(),
                ctx.rd.sample_count(),
                DepthMode::ReadOnly {
                    format: DEPTH_FORMAT,
                },
            ),
            sky_ground: SkyGroundNode::new(
                &ctx.rd.device,
                ctx.rd.target_format(),
                DEPTH_FORMAT,
                ctx.rd.sample_count(),
            ),
            depth: None,
            opaque_mesh: TriangleMesh::<3>::default(),
            faded_mesh: TriangleMesh::<3>::default(),
            controls: ToyboxControls::default(),
            line_node: LineRasterNode::new(
                &ctx.rd.device,
                ctx.rd.target_format(),
                DepthMode::Off,
                ctx.rd.sample_count(),
            ),
            arena_node: LineRasterNode::new(
                &ctx.rd.device,
                ctx.rd.target_format(),
                DepthMode::ReadOnly {
                    format: DEPTH_FORMAT,
                },
                ctx.rd.sample_count(),
            ),
            arena_mesh: LineMesh::<3>::default(),
            line_mesh: LineMesh::<3>::default(),
            left_was_down: false,
            pointer_was_captured: false,
            slice_up_held: false,
            slice_down_held: false,
            paused: false,
        })
    }

    fn respawn(&mut self) {
        self.pending.push(ToyboxCommand::Respawn);
    }

    fn panel(&mut self, ctx: &egui::Context) {
        let mut respawn = false;
        egui::Window::new("Toybox")
            .id(egui::Id::new("toybox-scene-controls"))
            .default_pos(egui::pos2(16.0, 48.0))
            .resizable(false)
            .show(ctx, |ui| {
                let slice = self.toybox.slice();
                let reach = self.toybox.slice_reach();
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("w slice").strong());
                    ui.label(
                        egui::RichText::new(format!("{slice:+.2}"))
                            .monospace()
                            .strong(),
                    );
                    let reachable = self
                        .toybox
                        .slice_marks()
                        .filter(|m| (m.w - slice).abs() < IN_SLICE_HALF_WIDTH)
                        .count();
                    ui.label(
                        egui::RichText::new(format!("{reachable}/{} in reach", TOYS.len())).weak(),
                    );
                });
                if let Some(dragged) = draw_slice_ruler(ui, slice, reach, self.toybox.slice_marks())
                {
                    self.pending.push(ToyboxCommand::Slice(dragged));
                }

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("respawn (R)").clicked() {
                        respawn = true;
                    }
                    ui.checkbox(&mut self.paused, "pause (Space)");
                });

                ui.collapsing("controls", |ui| {
                    ui.label("left-drag a shape to carry it; let go to throw it");
                    ui.label("hold Shift while dragging to pull it through w");
                    ui.label("right-drag to orbit the camera");
                    ui.separator();
                    ui.checkbox(&mut self.controls.w_labels, "wlabels");
                    ui.checkbox(&mut self.controls.wireframe.enabled, "wireframe");
                    ui.label(
                        egui::RichText::new("` for the console: physics, ground, wireframe")
                            .small()
                            .weak(),
                    );
                });
            });
        if respawn {
            self.respawn();
        }
    }

    /// A second write to the same upload buffer would change both recorded draws.
    fn record_lines(&mut self, ctx: &mut RenderCtx<'_>, view_proj: Mat4) {
        let rd = &ctx.rd;
        let cfg = &rd.surface_bundle.config;
        let mut mesh = std::mem::take(&mut self.line_mesh);
        self.toybox
            .build_overlay_mesh(&self.controls.overlay, &mut mesh);
        if self.controls.wireframe.enabled {
            append_toy_wireframe(
                &self.toybox.world,
                &self.toybox.toys,
                self.toybox.slice(),
                &self.controls.wireframe,
                &mut mesh,
                &mut self.toybox.local_vertices,
            );
        }
        if let Some(handle) = self.toybox.grab_handle() {
            push_axis_cross(
                &mut mesh,
                handle,
                GRAB_HANDLE_FRACTION * BODY_SIZE,
                GRAB_HANDLE_COLOR,
                GRAB_HANDLE_WIDTH,
            );
        }
        self.line_node.set_camera(
            &rd.queue,
            view_proj,
            Vec2::new(cfg.width as f32, cfg.height as f32),
        );
        self.line_node.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &mesh,
            &Projection::Identity,
            1,
        );
        self.line_mesh = mesh;
        self.line_node.record(ctx.encoder, ctx.view, None, None);

        let mut arena = std::mem::take(&mut self.arena_mesh);
        arena.segments.clear();
        arena.colors.clear();
        arena.widths.clear();
        append_arena_outline(&mut arena);
        if let Some(footprint) = self.toybox.drop_footprint() {
            append_drop_footprint(&mut arena, &footprint);
        }
        self.arena_node.set_camera(
            &rd.queue,
            view_proj,
            Vec2::new(cfg.width as f32, cfg.height as f32),
        );
        self.arena_node.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &arena,
            &Projection::Identity,
            1,
        );
        self.arena_mesh = arena;
        let depth = self.depth.as_ref().expect("ensure() guarantees Some");
        self.arena_node
            .record(ctx.encoder, ctx.view, Some(&depth.view), None);
    }

    fn w_readouts(&self, ctx: &egui::Context, frame: &FrameCtx<'_>) {
        if !self.controls.w_labels {
            return;
        }
        let ppp = ctx.pixels_per_point();
        let cfg = &frame.rd.surface_bundle.config;
        let viewport = (
            (cfg.width as f32 / ppp).round() as u32,
            (cfg.height as f32 / ppp).round() as u32,
        );
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Background,
            egui::Id::new("toybox-w-readout"),
        ));
        for (index, w) in self.toybox.w_offsets() {
            if w.abs() < W_LABEL_MIN {
                continue;
            }
            let world = self.toybox.position(index).truncate();
            let Some(anchor) =
                loam_egui::world_to_screen(&self.camera, world, viewport, &EuclideanR3)
            else {
                continue;
            };
            painter.text(
                anchor,
                egui::Align2::CENTER_CENTER,
                format!("w {w:+.2}"),
                egui::FontId::monospace(12.0),
                egui::Color32::from_rgb(232, 198, 120),
            );
        }
    }
}

impl loam_app::shell::Scene for ToyboxScene {
    fn apply_command(
        &mut self,
        cmd: &loam_app::command::CommandLine,
        _ctx: &mut loam_app::command::CommandCtx<'_>,
    ) -> Result<()> {
        self.console
            .dispatch(&cmd.name, &cmd.arg_refs(), &mut self.controls);
        Ok(())
    }

    fn tick(&mut self, dt: f32, _ctx: &mut loam_app::TickCtx) {
        for command in self.pending.drain(..) {
            self.toybox.apply_command(command);
        }
        for (toy, velocity) in self.controls.pending_throws.drain(..) {
            self.toybox.throw(toy, velocity);
        }
        let dir = (self.slice_up_held as i32 - self.slice_down_held as i32) as f32;
        if dir != 0.0 {
            self.toybox.scrub_slice(dir, dt);
        }
        if !self.paused {
            self.toybox.tick(dt);
        }
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        let cfg = &ctx.rd.surface_bundle.config;
        let viewport = (cfg.width, cfg.height);
        self.camera.aspect = viewport.0 as f32 / viewport.1.max(1) as f32;

        let down = ctx.input.buttons.left.down;
        let pressed = down && !self.left_was_down;
        let released = !down && self.left_was_down;
        self.left_was_down = down;

        let forward = self.camera.view().forward;
        let grabbing = !ctx.ui_capture.pointer;
        if !grabbing {
            if !self.pointer_was_captured {
                self.pending.push(ToyboxCommand::Release);
            }
        } else if pressed {
            if let Some(px) = ctx.input.buttons.left.press_pos {
                let ray = self.camera.ray_from_ndc(ndc_from_pixels(px, viewport));
                self.pending.push(ToyboxCommand::Press(ray, forward));
            }
        } else if released {
            self.pending.push(ToyboxCommand::Release);
        } else if let Some(px) = ctx.input.cursor_pos.filter(|_| down) {
            let axis = if ctx.input.modifiers.shift {
                GrabAxis::Through
            } else {
                GrabAxis::Slice
            };
            let ray = self.camera.ray_from_ndc(ndc_from_pixels(px, viewport));
            match self.pending.last_mut() {
                Some(ToyboxCommand::Hold(previous, view, held_axis, elapsed))
                    if *held_axis == axis =>
                {
                    *previous = ray;
                    *view = forward;
                    *elapsed += ctx.dt;
                }
                _ => self
                    .pending
                    .push(ToyboxCommand::Hold(ray, forward, axis, ctx.dt)),
            }
        }
        self.pointer_was_captured = ctx.ui_capture.pointer;

        if !ctx.ui_capture.pointer {
            self.orbit.advance(
                loam_app::orbit_on_right(ctx.input),
                &mut self.camera,
                &EuclideanR3,
                ctx.dt,
            );
        }
    }

    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        self.panel(ctx);
        self.w_readouts(ctx, frame);
        loam_app::log::pump_into(&mut self.console);
        frame.runtime.pump_console(&mut self.console);
        self.console.ui(ctx);
        frame.runtime.forward_console(&mut self.console);
    }

    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    ) {
        use winit::event::ElementState;
        use winit::keyboard::KeyCode;
        // Release must clear held keys even when the console captures input.
        let pressed = state == ElementState::Pressed && !ctx.ui_capture.keyboard;
        match code {
            KeyCode::ArrowUp => self.slice_up_held = pressed,
            KeyCode::ArrowDown => self.slice_down_held = pressed,
            KeyCode::Space if pressed => self.paused = !self.paused,
            KeyCode::KeyR if pressed => self.respawn(),
            _ => {}
        }
    }

    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> Result<()> {
        let rd = &ctx.rd;
        let cfg = &rd.surface_bundle.config;

        DepthBuffer::ensure(
            &mut self.depth,
            &rd.device,
            DEPTH_FORMAT,
            (cfg.width, cfg.height),
            rd.sample_count(),
        );
        let depth = self.depth.as_ref().expect("ensure() guarantees Some");

        let view = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height.max(1) as f32;
        let view_mat = Mat4::look_to_rh(view.position, view.forward, view.up);
        let proj_mat =
            Mat4::perspective_rh(self.camera.fov_y, aspect, self.camera.near, self.camera.far);
        let view_proj = proj_mat * view_mat;

        // Clear before either cap pass loads the attachments.
        self.sky_ground.set_uniforms(
            &rd.queue,
            &SkyGroundUniforms::new(
                view_proj,
                Viewport::full([cfg.width, cfg.height]),
                self.controls
                    .environment
                    .ground(FLOOR_Y, self.controls.environment.floor_visible),
            ),
        );
        self.sky_ground
            .record(ctx.encoder, ctx.view, &depth.view, None);

        self.toybox
            .build_frame_meshes(&mut self.opaque_mesh, &mut self.faded_mesh);
        for (node, mesh) in [
            (&mut self.caps, &self.opaque_mesh),
            (&mut self.faded_caps, &self.faded_mesh),
        ] {
            node.upload::<EuclideanR3, 3>(&rd.device, &rd.queue, mesh, &Projection::Identity);
            node.set_camera(&rd.queue, view_proj);
            node.record(ctx.encoder, ctx.view, Some(&depth.view), None);
        }
        self.record_lines(ctx, view_proj);
        Ok(())
    }

    fn title(&self, _fps: f32) -> Cow<'static, str> {
        Cow::Borrowed("polytope playground - Toybox")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc_probe;
    use loam_math::{Bivector, Plane4};
    use loam_physics::euclidean_r4::sphere_body_r4;
    use loam_physics::manifold::PENETRATION_SLOP;

    const SETTLE_TICKS: usize = 120;

    const FLICK_TICKS: usize = 10;

    const EYE: Vec3 = Vec3::new(0.0, 1.0, 8.0);
    const FORWARD: Vec3 = Vec3::new(0.0, 0.0, -1.0);

    fn scene() -> Toybox {
        Toybox::new().unwrap()
    }

    fn settled_awake() -> Toybox {
        let mut toybox = settled();
        for toy in 0..TOYS.len() {
            toybox.wake(toy);
        }
        toybox.tick(TICK_DT);
        toybox
    }

    fn settled() -> Toybox {
        let mut toybox = scene();
        toybox.run(SETTLE_TICKS);
        toybox
    }

    fn ray_through(target: Vec3) -> Ray {
        Ray {
            origin: EYE,
            direction: (target - EYE).normalize(),
        }
    }

    fn drag_frame(toybox: &mut Toybox, target: Vec3, axis: GrabAxis) {
        toybox.hold(&ray_through(target), FORWARD, axis, 1.0 / TICK_HZ as f32);
        toybox.tick(TICK_DT);
    }

    fn grab_centre(toybox: &mut Toybox, toy: usize) -> bool {
        let centre = toybox.position(toy).truncate();
        toybox.press(&ray_through(centre), FORWARD)
    }

    fn peak_w(toybox: &Toybox) -> f32 {
        toybox.w_offsets().map(|(_, w)| w.abs()).fold(0.0, f32::max)
    }

    fn boot_view_proj(aspect: f32) -> Mat4 {
        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.aspect = aspect;
        boot_orbit().advance(
            loam_app::Input::default(),
            &mut camera,
            &EuclideanR3,
            1.0 / 60.0,
        );
        let view = camera.view();
        Mat4::perspective_rh(CAMERA_FOV_DEG.to_radians(), aspect, CAMERA_NEAR, CAMERA_FAR)
            * Mat4::look_to_rh(view.position, view.forward, view.up)
    }

    fn hull_reach(toybox: &Toybox) -> (f32, f32) {
        let (mut x, mut z) = (0.0_f32, 0.0_f32);
        for body in toybox.world.bodies.iter() {
            let loam_physics::Collider::ConvexPolytope4D { vertices } = body.collider() else {
                continue;
            };
            for v in vertices {
                let world = body.orientation.rotation.apply(*v) + body.position;
                x = x.max(world.x.abs());
                z = z.max(world.z.abs());
            }
        }
        (x, z)
    }

    #[test]
    fn pointer_samples_do_not_apply_extra_tick_forces() {
        let mut one_sample = settled();
        let mut many_samples = settled();
        assert!(grab_centre(&mut one_sample, 0));
        assert!(grab_centre(&mut many_samples, 0));
        let ray = ray_through(one_sample.position(0).truncate() + Vec3::new(0.5, 0.5, 0.0));
        one_sample.hold(&ray, FORWARD, GrabAxis::Slice, TICK_DT);
        for _ in 0..4 {
            many_samples.hold(&ray, FORWARD, GrabAxis::Slice, TICK_DT / 4.0);
        }
        assert_eq!(many_samples.velocity(0), Vec4::ZERO);
        one_sample.tick(TICK_DT);
        many_samples.tick(TICK_DT);
        assert_eq!(one_sample.position(0), many_samples.position(0));
        assert_eq!(one_sample.velocity(0), many_samples.velocity(0));
    }

    #[test]
    fn tick_uses_the_supplied_duration() {
        let mut toybox = settled();
        toybox.throw(0, Vec4::Y * 3.0);
        let start = toybox.position(0).y;
        toybox.tick(1.0 / 120.0);
        let rise = toybox.position(0).y - start;
        assert!((0.023..0.025).contains(&rise), "rise {rise}");
    }

    #[test]
    fn slice_scrub_clamps_at_reachable_limits() {
        let mut toybox = scene();
        assert_eq!(toybox.slice(), W_SLICE, "the scene boots off its own slice");

        for _ in 0..TICK_HZ {
            toybox.scrub_slice(1.0, 1.0 / TICK_HZ as f32);
        }
        assert!(
            (toybox.slice() - W_SCRUB_RATE).abs() < 1e-4,
            "a second of held Up moved the slice to {}",
            toybox.slice()
        );

        for _ in 0..TICK_HZ * 10 {
            toybox.scrub_slice(1.0, 1.0 / TICK_HZ as f32);
        }
        assert_eq!(
            toybox.slice(),
            W_SLICE_RANGE,
            "the scrub ran past its range"
        );
        for _ in 0..TICK_HZ * 20 {
            toybox.scrub_slice(-1.0, 1.0 / TICK_HZ as f32);
        }
        assert_eq!(toybox.slice(), -W_SLICE_RANGE);

        toybox.set_slice(100.0);
        assert_eq!(toybox.slice(), W_SLICE_RANGE);
        toybox.set_slice(-100.0);
        assert_eq!(toybox.slice(), -W_SLICE_RANGE);
        toybox.set_slice(0.25);
        assert_eq!(toybox.slice(), 0.25);
    }

    #[test]
    fn slice_outside_hulls_clears_caps() {
        let mut toybox = settled();
        let (mut opaque, mut faded) = (TriangleMesh::default(), TriangleMesh::default());
        toybox.build_frame_meshes(&mut opaque, &mut faded);
        let at_home = opaque.vertices.len();
        assert!(
            at_home > 0,
            "the settled pile drew nothing at its own slice"
        );

        toybox.set_slice(W_SLICE_RANGE);
        toybox.build_frame_meshes(&mut opaque, &mut faded);
        assert!(
            opaque.vertices.is_empty() && faded.vertices.is_empty(),
            "a slice {W_SLICE_RANGE} away from the pile still cut it"
        );
        for toy in 0..toybox.toys().len() {
            assert_eq!(toybox.cap_stats(toy), (0.0, 0), "toy {toy} still has a cap");
        }

        toybox.set_slice(W_SLICE);
        toybox.build_frame_meshes(&mut opaque, &mut faded);
        assert_eq!(
            opaque.vertices.len(),
            at_home,
            "the slice came home to a different pile"
        );
    }

    #[test]
    fn substeps_follow_fastest_body() {
        let mut toybox = settled();
        assert_eq!(
            toybox.substeps_for_current_speed(TICK_DT),
            BASE_SUBSTEPS,
            "a settled pile should cost the settling rate and nothing more"
        );

        let thrown = toybox.toys[0].body;
        toybox.world.bodies[thrown].velocity = Vec4::new(MAX_RELEASE_SPEED, 0.0, 0.0, 0.0);
        let fast = toybox.substeps_for_current_speed(TICK_DT);
        assert!(
            fast > BASE_SUBSTEPS && fast <= MAX_SUBSTEPS,
            "a body at the ceiling asked for {fast} substeps"
        );
        assert!(
            toybox.fastest_step_travel() <= STEP_TRAVEL_BUDGET,
            "the chosen substep count still lets a step outrun the narrowphase"
        );
    }

    #[test]
    fn grab_target_respects_arena_bounds() {
        let reach = ARENA_HALF_EXTENT - BODY_SIZE;
        for corner in [
            Vec4::new(99.0, -99.0, 99.0, 0.0),
            Vec4::new(-99.0, -5.0, -99.0, 2.0),
        ] {
            let held = clamp_target_to_arena(corner);
            assert!(held.x.abs() <= reach + 1e-6 && held.z.abs() <= reach + 1e-6);
            assert!(held.y >= FLOOR_Y + BODY_SIZE - 1e-6);
            assert_eq!(held.w, corner.w, "the clamp must leave w alone");
        }
    }

    #[test]
    fn wall_clamp_preserves_release_velocity() {
        let mut toybox = settled();
        assert!(grab_centre(&mut toybox, 0));
        let mut cursor = toybox.position(0).truncate();
        for _ in 0..12 {
            cursor += Vec3::new(3.0, 0.0, 0.0);
            drag_frame(&mut toybox, cursor, GrabAxis::Slice);
        }
        toybox.release();
        assert!(
            toybox.velocity(0).length() > 1.0,
            "a hard flick past the wall threw {}",
            toybox.velocity(0).length()
        );
    }

    #[test]
    fn held_body_does_not_sleep() {
        let mut toybox = settled();
        assert!(grab_centre(&mut toybox, 0));
        let start = toybox.position(0);
        let mut cursor = start.truncate();
        for _ in 0..(REST_WINDOW as usize * 3) {
            cursor += Vec3::new(0.004, 0.0, 0.0);
            drag_frame(&mut toybox, cursor, GrabAxis::Slice);
        }
        let carried = toybox.position(0) - start;
        assert!(
            carried.truncate().length() > 0.05,
            "a held body moved {carried} while being carried, so the latch pulled it back to its anchor"
        );
    }

    #[test]
    fn the_draw_reads_the_console_controls_rather_than_a_scene_constant() {
        let toybox = settled();
        let draw = |controls: &WireframeControls| {
            let mut mesh = LineMesh::<3>::default();
            append_toy_wireframe(
                &toybox.world,
                &toybox.toys,
                toybox.slice(),
                controls,
                &mut mesh,
                &mut Vec::new(),
            );
            mesh
        };
        let shipped = ToyboxControls::default().wireframe;
        let widened = draw(&WireframeControls {
            width_px: shipped.width_px * 2.0,
            ..shipped
        });
        let faded = draw(&WireframeControls {
            alpha: 0.5,
            ..shipped
        });
        let reprojected = draw(&WireframeControls {
            projection: WireframeProjection::Shadow,
            ..shipped
        });
        let base = draw(&shipped);
        assert_ne!(widened.widths, base.widths, "`wireframe width` is inert");
        assert!(
            faded.colors.iter().all(|(a, b)| a[3] == 0.5 && b[3] == 0.5),
            "`wireframe alpha` is inert"
        );
        assert_ne!(
            reprojected.segments, base.segments,
            "`wireframe perspective` is inert"
        );
    }

    #[test]
    fn wireframe_reuses_its_mesh_and_pose_buffers() {
        let toybox = settled();
        let mut mesh = LineMesh::default();
        let mut posed = Vec::new();
        let controls = ToyboxControls::default().wireframe;
        append_toy_wireframe(
            &toybox.world,
            &toybox.toys,
            toybox.slice(),
            &controls,
            &mut mesh,
            &mut posed,
        );
        mesh.segments.clear();
        mesh.colors.clear();
        mesh.widths.clear();
        let bytes = alloc_probe::bytes_allocated_by(|| {
            append_toy_wireframe(
                &toybox.world,
                &toybox.toys,
                toybox.slice(),
                &controls,
                &mut mesh,
                &mut posed,
            );
        });
        assert_eq!(bytes, 0);
    }

    #[test]
    fn the_wireframe_shade_follows_a_vertex_distance_from_the_slice() {
        let mut toybox = settled();
        let mut near = LineMesh::<3>::default();
        append_toy_wireframe(
            &toybox.world,
            &toybox.toys,
            toybox.slice(),
            &ToyboxControls::default().wireframe,
            &mut near,
            &mut Vec::new(),
        );
        toybox.set_slice(W_SLICE_RANGE);
        let mut far = LineMesh::<3>::default();
        append_toy_wireframe(
            &toybox.world,
            &toybox.toys,
            toybox.slice(),
            &ToyboxControls::default().wireframe,
            &mut far,
            &mut Vec::new(),
        );
        let brightness =
            |m: &LineMesh<3>| -> f32 { m.colors.iter().map(|(a, _)| a[0] + a[1] + a[2]).sum() };
        assert!(
            brightness(&far) < brightness(&near),
            "moving the slice off the pile did not dim the wireframe, so the shade is not reading w at all"
        );
    }

    #[test]
    fn collision_wakes_sleeping_body() {
        let mut toybox = settled();
        assert!(
            toybox.world.bodies[toybox.toys[1].body].is_sleeping(),
            "the pile never slept, so this pin is vacuous"
        );
        let start = toybox.position(1);
        let (from, to) = (toybox.position(0), toybox.position(1));
        toybox.wake(0);
        let thrown = toybox.toys[0].body;
        toybox.world.bodies[thrown].velocity = (to - from).normalize() * MAX_CARRY_SPEED;
        let mut woke = false;
        for _ in 0..60 {
            toybox.tick(TICK_DT);
            woke |= !toybox.world.bodies[toybox.toys[1].body].is_sleeping();
        }
        assert!(woke, "the struck toy never woke, so it is still static");
        assert!(
            (toybox.position(1) - start).length() > 0.01,
            "the struck toy never moved"
        );
    }

    #[test]
    fn release_speed_tracks_pointer_speed() {
        fn thrown_at(units_per_frame: f32) -> f32 {
            let mut toybox = settled();
            assert!(grab_centre(&mut toybox, 0));
            let mut cursor = toybox.position(0).truncate();
            for _ in 0..12 {
                cursor += Vec3::new(units_per_frame, 0.0, 0.0);
                drag_frame(&mut toybox, cursor, GrabAxis::Slice);
            }
            toybox.release();
            assert!(toybox.fastest_step_travel() <= STEP_TRAVEL_BUDGET + 1e-5);
            toybox.velocity(0).length()
        }

        let gentle = thrown_at(0.05);
        let medium = thrown_at(0.5);
        let hard = thrown_at(4.0);
        assert!(
            gentle < medium && medium < hard,
            "throws did not order with cursor speed: {gentle}, {medium}, {hard}"
        );
        assert!(
            hard > 4.0 * medium,
            "a ten-times-faster drag threw only {hard} against {medium}, so the top of the range is still flattened"
        );
        assert!(
            hard <= MAX_RELEASE_SPEED,
            "a throw at {hard} passed the narrowphase ceiling {MAX_RELEASE_SPEED}"
        );
    }
    #[test]
    fn maximum_throw_stays_inside_walls() {
        let reach = ARENA_HALF_EXTENT + 16.0 * PENETRATION_SLOP;
        for direction in [
            Vec4::X,
            -Vec4::X,
            Vec4::Z,
            -Vec4::Z,
            Vec4::new(1.0, 0.4, 1.0, 0.0).normalize(),
            Vec4::new(-1.0, 0.6, 0.7, 0.0).normalize(),
        ] {
            let mut toybox = settled();
            let thrown = toybox.toys[0].body;
            toybox.wake(0);
            toybox.world.bodies[thrown].velocity = direction * MAX_RELEASE_SPEED;
            for tick in 0..600 {
                toybox.tick(TICK_DT);
                let (x, z) = hull_reach(&toybox);
                assert!(
                    x <= reach && z <= reach,
                    "a throw along {direction} reached x {x}, z {z} at tick {tick}, \
                     outside the container's {ARENA_HALF_EXTENT}"
                );
            }
        }
    }

    #[test]
    fn the_boot_camera_frames_the_whole_arena() {
        let e = ARENA_HALF_EXTENT;
        for aspect in [4.0 / 3.0, 16.0 / 9.0] {
            let view_proj = boot_view_proj(aspect);
            for corner in [
                Vec3::new(-e, FLOOR_Y, -e),
                Vec3::new(e, FLOOR_Y, -e),
                Vec3::new(e, FLOOR_Y, e),
                Vec3::new(-e, FLOOR_Y, e),
            ] {
                let clip = view_proj * corner.extend(1.0);
                assert!(clip.w > 0.0, "corner {corner} is behind the boot camera");
                let ndc = clip.truncate() / clip.w;
                assert!(
                    ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0 && (0.0..=1.0).contains(&ndc.z),
                    "aspect {aspect}: arena corner {corner} lands at ndc {ndc}, \
                     outside the boot frame"
                );
            }
        }
    }

    #[test]
    fn invisible_body_remains_pickable() {
        let mut toybox = settled();
        let ray = ray_through(toybox.position(0).truncate());
        assert_eq!(toybox.pick(&ray).map(|(toy, _)| toy), Some(0));

        let id = toybox.toys[0].body;
        toybox.world.bodies[id].position.w += 4.0 * BODY_SIZE;
        assert_eq!(toybox.cap_stats(0), (0.0, 0), "the toy still has a cap");

        let (picked, hit) = toybox
            .pick(&ray)
            .expect("the off-slice body is unreachable");
        assert_eq!(picked, 0, "the fallback grabbed the wrong body");
        assert!(
            (hit - toybox.position(0).truncate()).length() <= BODY_SIZE + 1e-4,
            "the fallback reported a hit {hit} away from the body's ball"
        );
        assert!(
            toybox.press(&ray, FORWARD),
            "the off-slice body cannot be grabbed"
        );
        assert_eq!(toybox.grab.as_ref().map(|g| g.toy), Some(0));
    }

    #[test]
    fn rim_pick_has_bounded_tolerance() {
        let mut toybox = settled();
        let centre = toybox.position(0).truncate();
        let mut grazing = None;
        for step in 1..400 {
            let out = 3.0 * BODY_SIZE * step as f32 / 400.0;
            let probe = centre + Vec3::new(out, 0.0, 0.0);
            if toybox.pick(&ray_through(probe)).is_none() {
                grazing = Some(out);
                break;
            }
        }
        let edge = grazing.expect("the pick never stopped, so it has no bound at all");
        assert!(
            edge > PICK_TOLERANCE && edge < 2.0 * BODY_SIZE,
            "the pick died {edge} from the centre, outside the section-plus-rim band"
        );
    }

    #[test]
    fn angular_damping_decays_free_flight_spin() {
        let mut toybox = settled();
        let id = toybox.toys[0].body;
        toybox.world.bodies[id].position += Vec4::Y * 6.0;
        toybox.wake(0);
        toybox.world.bodies[id].angular_velocity = Bivector4::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.6);
        let launched = toybox.angular_velocity(0).magnitude();

        let mut last = launched;
        for _ in 0..TICK_HZ {
            toybox.tick(TICK_DT);
            let now = toybox.angular_velocity(0).magnitude();
            assert!(now <= last + 1e-6, "the spin grew in free flight");
            last = now;
        }
        let expected = launched * (-ANGULAR_DAMPING).exp();
        assert!(
            (last - expected).abs() < 1e-3 * launched,
            "a second of free flight left {last} of spin, not the {expected} the \
             damping coefficient prescribes"
        );
        assert!(
            last < 0.4 * launched,
            "a knocked body still tumbles at {} of its launch spin after a second",
            last / launched
        );
    }

    #[test]
    fn spawned_hulls_settle_without_creeping() {
        let mut toybox = scene();
        assert!(
            toybox.deepest_point() > SPAWN_CLEARANCE - 1e-5,
            "a toy spawned inside the floor"
        );
        let spawned: Vec<f32> = (0..toybox.toys().len())
            .map(|i| toybox.position(i).y)
            .collect();

        for _ in 0..SETTLE_TICKS {
            toybox.tick(TICK_DT);
            assert!(toybox.deepest_point() > -8.0 * PENETRATION_SLOP);
            assert!(toybox.fastest_step_travel() < RESOLVABLE_STEP_TRAVEL);
        }
        for (index, w) in toybox.w_offsets() {
            assert!(
                w.abs() < SETTLED_W_BAND,
                "floor moved toy {index} through w"
            );
        }
        for toy in 0..toybox.toys().len() {
            let (alpha, vertices) = toybox.cap_stats(toy);
            assert!(
                vertices > 0 && alpha == 1.0,
                "toy {toy} settled outside the slice"
            );
        }
        for (toy, start) in spawned.iter().enumerate() {
            assert!(toybox.position(toy).y < *start, "toy {toy} never fell");
            assert_eq!(
                toybox.velocity(toy),
                Vec4::ZERO,
                "toy {toy} never came to rest"
            );
        }
        let deepest = toybox.deepest_point();
        assert!(
            deepest > -2.0 * PENETRATION_SLOP,
            "a toy sank to {deepest} through the floor"
        );
        assert!(
            deepest < 0.02,
            "the pile came to rest {deepest} above the floor rather than on it"
        );

        let resting: Vec<Vec4> = (0..toybox.toys().len())
            .map(|i| toybox.position(i))
            .collect();
        toybox.run(2);
        let after: Vec<Vec4> = (0..toybox.toys().len())
            .map(|i| toybox.position(i))
            .collect();
        assert_eq!(resting, after, "a settled pile kept creeping");
    }

    #[test]
    fn the_spawn_pose_lands_a_whole_cell_on_the_floor() {
        for polytope in [
            Polytope4::Pentatope,
            Polytope4::Tesseract,
            Polytope4::Cell16,
            Polytope4::Cell24,
        ] {
            let topology = polytope.topology();
            for cell in [0, topology.cells.len() / 2, topology.cells.len() - 1] {
                let pose = face_down_pose(polytope, cell);
                let posed: Vec<Vec4> = (topology.vertices.iter()).map(|v| pose.apply(*v)).collect();
                let lowest = posed.iter().fold(f32::INFINITY, |m, v| m.min(v.y));
                for index in topology.cells[cell] {
                    let y = posed[*index as usize].y;
                    assert!(
                        (y - lowest).abs() < 1e-4,
                        "{polytope:?} cell {cell} vertex {index} sits {} above the \
                         lowest point, so the toy is dropped on a corner",
                        y - lowest
                    );
                }
            }
        }
    }

    #[test]
    fn grab_tracks_pointer_motion() {
        let mut toybox = settled();
        assert!(grab_centre(&mut toybox, 0), "the grab missed toy 0");
        let start = toybox.position(0);
        let mut cursor = start.truncate();
        for _ in 0..30 {
            cursor += Vec3::new(0.02, 0.02, 0.0);
            drag_frame(&mut toybox, cursor, GrabAxis::Slice);
        }
        let held = toybox.position(0);
        assert!(
            (held.truncate() - cursor).length() < 0.15,
            "the body sat {} from the cursor it was dragged by",
            (held.truncate() - cursor).length()
        );
        assert!(
            (held - start).length() > 0.3,
            "the body never left the place it was grabbed at"
        );
    }

    #[test]
    fn release_ignores_stale_pointer_motion() {
        let throw = |stall: usize| {
            let mut toybox = settled();
            assert!(grab_centre(&mut toybox, 0));
            let mut cursor = toybox.position(0).truncate();
            for _ in 0..30 {
                cursor += Vec3::new(0.05, 0.0, 0.0);
                drag_frame(&mut toybox, cursor, GrabAxis::Slice);
            }
            for _ in 0..stall {
                drag_frame(&mut toybox, cursor, GrabAxis::Slice);
            }
            toybox.release();
            toybox.velocity(0)
        };
        let flicked = throw(0);
        let stalled = throw(12);
        assert!(
            flicked.x > 0.5 * 3.0 * RELEASE_GAIN,
            "a 3 u/s drag released in motion threw at {flicked}"
        );
        assert!(
            stalled.length() < 0.1 * flicked.length(),
            "a drag that stopped before the release still threw at {stalled}: the \
             release is reading the total drag, not the recent motion"
        );
    }

    #[test]
    fn off_centre_release_applies_torque() {
        let mut toybox = settled();
        let centre = toybox.position(0).truncate();
        let handle = centre + Vec3::new(0.0, 0.22, 0.0);
        assert!(
            toybox.press(&ray_through(handle), FORWARD),
            "the grab missed"
        );
        let id = toybox.toys[0].body;
        let lever_local = toybox.grab.as_ref().expect("held").lever_local;
        assert!(
            lever_local.length() > 0.05,
            "the grab stored a handle at the centre of mass"
        );

        let mut cursor = handle;
        for _ in 0..20 {
            cursor += Vec3::new(0.03, 0.0, 0.0);
            drag_frame(&mut toybox, cursor, GrabAxis::Slice);
        }
        let rotation = toybox.world.bodies[id].orientation.rotation;
        let inertia = toybox.world.bodies[id].inertia;
        let before = toybox.angular_velocity(0);
        toybox.release();

        let after = toybox.angular_velocity(0);
        assert!(
            after.magnitude() > 0.5,
            "an off-centre grab left the body spinning at {}",
            after.magnitude()
        );
        let lever = rotation.apply(lever_local);
        let expected =
            Bivector4::wedge(lever, toybox.velocity(0) * BODY_MASS) * (RELEASE_SPIN_GAIN / inertia);
        let got = after + before * -1.0;
        assert!(
            (got + expected * -1.0).magnitude() < 1e-3 * expected.magnitude().max(1.0),
            "the release torqued by {got:?}, not RELEASE_SPIN_GAIN of the grabbed point's {expected:?}"
        );
    }

    #[test]
    fn through_modifier_controls_w_velocity() {
        let throw = |axis: GrabAxis| {
            let mut toybox = settled();
            let parked = toybox.position(0).w;
            assert!(grab_centre(&mut toybox, 0));
            let mut cursor = toybox.position(0).truncate();
            for _ in 0..24 {
                cursor += Vec3::new(0.0, 0.04, 0.0);
                drag_frame(&mut toybox, cursor, axis);
            }
            toybox.release();
            (parked, toybox.position(0), toybox.velocity(0))
        };
        let (parked, position, velocity) = throw(GrabAxis::Slice);
        assert_eq!(velocity.w, 0.0, "an unmodified grab threw off the slice");
        assert_eq!(
            position.w, parked,
            "an unmodified grab carried the body off the slice"
        );

        let (parked, position, velocity) = throw(GrabAxis::Through);
        assert!(
            velocity.w > 0.5,
            "the modifier released at w velocity {}",
            velocity.w
        );
        assert!(
            position.w - parked > 0.5,
            "the modifier never carried the body off the slice (w {})",
            position.w - parked
        );
        assert!(
            velocity.y.abs() < 1e-5,
            "the modifier kept the screen rise as well as trading it for w"
        );
    }

    #[test]
    fn body_collision_transfers_w_momentum() {
        let mut toybox = settled();
        assert!(peak_w(&toybox) < SETTLED_W_BAND);
        assert!(grab_centre(&mut toybox, 0));
        let target = toybox.position(1).truncate();
        let mut cursor = toybox.position(0).truncate();
        let step = (target - cursor) / 12.0;
        for _ in 0..12 {
            cursor += step;
            drag_frame(&mut toybox, cursor, GrabAxis::Slice);
        }
        toybox.release();
        let mut worst = 0.0_f32;
        for _ in 0..180 {
            toybox.tick(TICK_DT);
            worst = worst.max(peak_w(&toybox));
        }
        assert!(
            worst > SETTLED_W_BAND,
            "hull against hull left every body inside the band the floor alone \
             keeps them in (best |w| = {worst})"
        );
    }

    #[test]
    fn shrinking_caps_leave_opaque_pass() {
        let mut toybox = settled();
        let (mut opaque, mut faded) = (TriangleMesh::default(), TriangleMesh::default());
        toybox.build_frame_meshes(&mut opaque, &mut faded);
        let solid = opaque.vertices.len();
        assert!(solid > 0, "nothing was drawn in the slice");
        assert!(faded.vertices.is_empty(), "a settled pile drew a faded cap");
        for color in &opaque.colors {
            assert_eq!(color[3], 1.0, "the opaque pass carries a translucent cap");
        }

        assert!(grab_centre(&mut toybox, 0));
        let mut cursor = toybox.position(0).truncate();
        let mut alphas = Vec::new();
        for _ in 0..24 {
            cursor += Vec3::new(0.0, 0.02, 0.0);
            drag_frame(&mut toybox, cursor, GrabAxis::Through);
            alphas.push(toybox.cap_stats(0).0);
        }
        assert_eq!(alphas[0], 1.0, "the body started already faded");
        assert!(
            alphas.windows(2).all(|w| w[1] <= w[0] + 1e-6),
            "the fade did not fall as the cap shrank: {alphas:?}"
        );
        assert!(
            *alphas.last().expect("non-empty") < 0.5,
            "the cap was still at alpha {} when it left the slice",
            alphas.last().expect("non-empty")
        );

        toybox.build_frame_meshes(&mut opaque, &mut faded);
        assert!(
            opaque.vertices.len() < solid,
            "the drifting body is still drawn in the opaque pass"
        );
        let expected: usize = (0..toybox.toys().len())
            .map(|toy| {
                let (alpha, vertices) = toybox.cap_stats(toy);
                if alpha >= 1.0 {
                    vertices
                } else {
                    0
                }
            })
            .sum();
        assert_eq!(
            opaque.vertices.len(),
            expected,
            "a cap landed in the wrong pass for its alpha"
        );
    }

    #[test]
    fn visible_cap_pick_rejects_bounding_sphere_excess() {
        let mut toybox = settled();
        let centre = toybox.position(0).truncate();
        assert!(
            toybox.pick(&ray_through(centre)).is_some(),
            "a ray at the centre missed the body"
        );

        let far = centre + Vec3::new(BODY_SIZE * 0.95, BODY_SIZE * 0.95, 0.0);
        assert!(
            (far - centre).length() < BODY_SIZE * 1.5,
            "the probe left the ball, so the pin is vacuous"
        );
        assert_eq!(
            toybox.pick(&ray_through(far)),
            None,
            "a ray far outside the section still grabbed, so the fallback is behaving like a bounding-ball pick"
        );
    }

    #[test]
    fn pick_selects_nearest_cap() {
        let mut toybox = settled();
        let ray = ray_through(toybox.position(0).truncate());
        assert_eq!(toybox.pick(&ray).map(|(toy, _)| toy), Some(0));

        let blocker = toybox.toys[1].body;
        let far = toybox.position(0);
        toybox.world.bodies[blocker].position =
            Vec4::from((ray.origin + ray.direction * 2.0, far.w));
        assert_eq!(
            toybox.pick(&ray).map(|(toy, _)| toy),
            Some(1),
            "the pick reached past the nearer body"
        );
    }

    #[test]
    fn missed_grab_does_not_throw() {
        let mut toybox = settled();
        let sky = Ray {
            origin: EYE,
            direction: Vec3::new(0.0, 1.0, 0.0),
        };
        assert_eq!(toybox.pick(&sky), None);
        assert!(!toybox.press(&sky, FORWARD));
        assert!(toybox.grab.is_none());
        toybox.release();
        for toy in 0..toybox.toys().len() {
            assert_eq!(
                toybox.velocity(toy),
                Vec4::ZERO,
                "a missed press threw {toy}"
            );
        }
    }

    #[test]
    fn the_grab_plane_holds_the_pick_depth_across_the_view() {
        let depth = 6.0;
        for ndc in [(0.0_f32, 0.0_f32), (0.7, 0.4), (-0.9, -0.6)] {
            let direction = (FORWARD + Vec3::X * ndc.0 * 0.8 + Vec3::Y * ndc.1 * 0.5).normalize();
            let ray = Ray {
                origin: EYE,
                direction,
            };
            let point = plane_point(&ray, FORWARD, depth).expect("a camera ray meets the plane");
            assert!(
                ((point - EYE).dot(FORWARD) - depth).abs() < 1e-4,
                "ndc {ndc:?} landed at depth {}",
                (point - EYE).dot(FORWARD)
            );
        }
        let parallel = Ray {
            origin: EYE,
            direction: Vec3::X,
        };
        assert_eq!(plane_point(&parallel, FORWARD, depth), None);
    }

    #[test]
    fn the_slice_can_always_be_scrubbed_out_to_the_deepest_toy() {
        let mut toybox = settled();
        assert_eq!(
            toybox.slice_reach(),
            W_SLICE_RANGE,
            "a settled pile should not widen the reach past the spawn range"
        );

        let rolled = W_SLICE_RANGE * 3.0;
        let id = toybox.toys[0].body;
        toybox.world.bodies[id].position.w = rolled;

        toybox.set_slice(rolled);
        assert!(
            (toybox.slice() - rolled).abs() < IN_SLICE_HALF_WIDTH,
            "the slice clamped at {} and could not reach a toy at {rolled}",
            toybox.slice()
        );
        assert!(
            toybox
                .slice_marks()
                .any(|m| (m.w - toybox.slice()).abs() < IN_SLICE_HALF_WIDTH),
            "the scrub arrived where no toy was"
        );
    }

    #[test]
    fn a_click_on_the_ruler_lands_on_the_w_it_points_at() {
        const LEFT: f32 = 40.0;
        const WIDTH: f32 = 200.0;
        let reach = 4.5;
        for (x, want) in [
            (LEFT, -reach),
            (LEFT + WIDTH * 0.5, 0.0),
            (LEFT + WIDTH, reach),
            (LEFT + WIDTH * 0.75, reach * 0.5),
        ] {
            let got = slice_for_ruler_x(x, LEFT, WIDTH, reach);
            assert!(
                (got - want).abs() < 1e-4,
                "a click at {x} read as w {got}, not {want}"
            );
        }
        assert_eq!(slice_for_ruler_x(LEFT - 500.0, LEFT, WIDTH, reach), -reach);
        assert_eq!(slice_for_ruler_x(LEFT + 500.0, LEFT, WIDTH, reach), reach);
    }

    #[test]
    fn the_drawn_handle_rides_the_caught_point_rather_than_the_body_centre() {
        let mut toybox = settled();
        assert!(
            toybox.grab_handle().is_none(),
            "a handle drew with nothing held"
        );
        let centre = toybox.position(0).truncate();
        assert!(toybox.press(&ray_through(centre + Vec3::new(0.0, 0.2, 0.0)), FORWARD));

        let held = toybox.grab_handle().expect("held");
        assert!(
            (held - toybox.position(0).truncate()).length() > 0.05,
            "the handle drew at the centre of mass, which is exactly the off-centre catch the marker exists to show"
        );

        let id = toybox.toys[0].body;
        toybox.world.bodies[id].orientation.rotation =
            (Plane4::Xy.unit_bivector() * 0.9).exp().normalize();
        assert!(
            (toybox.grab_handle().expect("still held") - held).length() > 0.05,
            "the handle stayed put while the hull turned under it"
        );

        toybox.release();
        assert!(
            toybox.grab_handle().is_none(),
            "the handle outlived the grab"
        );
    }

    const TRANSLATE_TOL: f32 = 1e-5;

    const FIXTURE_DT: f32 = 1.0 / 60.0;

    const FIXTURE_OVERLAP: f32 = 0.2;

    const FIXTURE_GROUP_GAP: f32 = 20.0;

    const FIXTURE_SLIDE_SPEED: f32 = 3.0;

    fn overlapping_pairs(pairs: usize) -> World<EuclideanR4> {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        for pair in 0..pairs {
            let base = Vec4::X * (FIXTURE_GROUP_GAP * pair as f32);
            world.push_body(sphere_body_r4(base, Vec4::ZERO, BODY_SIZE, BODY_MASS).unwrap());
            world.push_body(
                sphere_body_r4(
                    base + Vec4::X * (2.0 * BODY_SIZE - FIXTURE_OVERLAP),
                    Vec4::ZERO,
                    BODY_SIZE,
                    BODY_MASS,
                )
                .unwrap(),
            );
        }
        world.step(FIXTURE_DT);
        assert_eq!(
            world.manifolds.len(),
            pairs,
            "the fixture layout did not produce one manifold per pair"
        );
        world
    }

    fn only(layer: fn(&mut PhysicsOverlay)) -> PhysicsOverlay {
        let mut overlay = PhysicsOverlay::default();
        layer(&mut overlay);
        overlay
    }

    fn overlay_mesh(world: &World<EuclideanR4>, overlay: &PhysicsOverlay) -> LineMesh<3> {
        let mut mesh = LineMesh::<3>::default();
        build_physics_overlay_mesh(world, BODY_SIZE, overlay, &mut mesh);
        assert_eq!(mesh.colors.len(), mesh.segments.len());
        assert_eq!(mesh.widths.len(), mesh.segments.len());
        mesh
    }

    fn contact_count(world: &World<EuclideanR4>) -> usize {
        world.manifolds.values().map(|m| m.points.len()).sum()
    }

    #[test]
    fn a_landing_fills_the_manifolds_the_overlay_draws_from() {
        let toybox = settled_awake();
        assert!(
            !toybox.world.manifolds.is_empty(),
            "the pile settled with no manifold, so every layer would draw nothing"
        );
        let peak = (toybox.world.manifolds.values())
            .flat_map(|m| m.points.iter())
            .fold(0.0f32, |m, cp| m.max(cp.normal_impulse));
        assert!(peak > 0.0, "a resting pile accumulated no normal impulse");

        let mut mesh = LineMesh::<3>::default();
        for (name, overlay) in [
            ("contacts", only(|o| o.contacts = true)),
            ("normals", only(|o| o.normals = true)),
            ("impulses", only(|o| o.impulses = true)),
            ("islands", only(|o| o.islands = true)),
        ] {
            toybox.build_overlay_mesh(&overlay, &mut mesh);
            assert!(
                !mesh.segments.is_empty(),
                "the {name} layer drew nothing over a settled pile"
            );
        }
    }

    #[test]
    fn the_islands_layer_draws_no_coupling_to_the_static_floor() {
        let toybox = settled_awake();
        let islands = toybox.world.islands();
        let floor_pairs: usize = (islands.iter())
            .flat_map(|i| i.constraints.iter())
            .filter(|&&(a, b)| {
                toybox.world.bodies[a].inv_mass() == 0.0 || toybox.world.bodies[b].inv_mass() == 0.0
            })
            .count();
        assert!(
            floor_pairs > 0,
            "no toy is resting on the floor, so the skip is vacuous"
        );

        let mut mesh = LineMesh::<3>::default();
        toybox.build_overlay_mesh(&only(|o| o.islands = true), &mut mesh);
        let crosses: usize = islands.iter().map(|i| 3 * i.bodies.len()).sum();
        let couplings: usize = (islands.iter())
            .flat_map(|i| i.constraints.iter())
            .filter(|&&(a, b)| {
                toybox.world.bodies[a].inv_mass() != 0.0 && toybox.world.bodies[b].inv_mass() != 0.0
            })
            .count();
        assert_eq!(
            mesh.segments.len(),
            crosses + couplings,
            "the islands layer drew a bar for a pair the solver couples nothing through"
        );
    }

    #[test]
    fn every_contact_emits_one_normal_along_the_stored_direction() {
        let world = overlapping_pairs(2);
        let mesh = overlay_mesh(&world, &only(|o| o.normals = true));

        let contacts = contact_count(&world);
        assert!(contacts > 0, "fixture produced no contacts");
        assert_eq!(
            mesh.segments.len(),
            contacts,
            "the normals layer emitted {} segments for {contacts} contacts",
            mesh.segments.len()
        );

        let mut segments = mesh.segments.iter();
        for (&(a, b), manifold) in world.manifolds.iter() {
            let separation = (world.bodies[b].position - world.bodies[a].position).truncate();
            for cp in &manifold.points {
                let &(from, to) = segments.next().expect("one segment per contact");
                let point = cp.world_point.truncate();
                assert!(
                    (Vec3::from_array(from) - point).length() < TRANSLATE_TOL,
                    "normal starts at {from:?}, not at the contact point {point:?}"
                );
                let drawn = Vec3::from_array(to) - Vec3::from_array(from);
                let expected = cp.normal.truncate() * (NORMAL_LEN_FRACTION * BODY_SIZE);
                assert!(
                    (drawn - expected).length() < TRANSLATE_TOL,
                    "normal drawn as {drawn:?}, not {expected:?}"
                );
                assert!(
                    drawn.dot(separation) > 0.0,
                    "normal runs from A toward B; the layer would misreport a flipped normal"
                );
            }
        }
    }

    #[test]
    fn impulse_bar_length_tracks_the_accumulated_impulse() {
        let world = overlapping_pairs(1);
        let contacts = contact_count(&world);
        let solved: f32 = (world.manifolds.values())
            .flat_map(|m| m.points.iter())
            .map(|cp| cp.normal_impulse)
            .sum();
        assert!(
            solved > 0.0,
            "fixture accumulated no normal impulse, so the length pin is vacuous"
        );

        let base = only(|o| o.impulses = true);
        let mesh = overlay_mesh(&world, &base);
        assert_eq!(
            mesh.segments.len(),
            2 * contacts,
            "the impulses layer emits a normal and a tangent bar per contact"
        );

        let doubled = overlay_mesh(
            &world,
            &PhysicsOverlay {
                impulse_scale: base.impulse_scale * 2.0,
                ..base
            },
        );
        for (i, (&(from, to), &(from2, to2))) in
            mesh.segments.iter().zip(&doubled.segments).enumerate()
        {
            let short = Vec3::from_array(to) - Vec3::from_array(from);
            let long = Vec3::from_array(to2) - Vec3::from_array(from2);
            assert!(
                (long - short * 2.0).length() < TRANSLATE_TOL,
                "bar {i} did not scale with impulse_scale: {short:?} then {long:?}"
            );
        }

        let points = world.manifolds.values().flat_map(|m| m.points.iter());
        for (cp, chunk) in points.zip(mesh.segments.chunks_exact(2)) {
            let bar = Vec3::from_array(chunk[0].1) - Vec3::from_array(chunk[0].0);
            let expected = cp.normal.truncate() * (cp.normal_impulse * base.impulse_scale);
            assert!(
                (bar - expected).length() < TRANSLATE_TOL,
                "normal-impulse bar drawn as {bar:?}, not {expected:?}"
            );
        }
    }

    #[test]
    fn a_sliding_pair_draws_its_friction_bar_against_the_slide() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_body(
            sphere_body_r4(
                Vec4::ZERO,
                Vec4::Y * FIXTURE_SLIDE_SPEED,
                BODY_SIZE,
                BODY_MASS,
            )
            .unwrap(),
        );
        world.push_body(
            sphere_body_r4(
                Vec4::X * (2.0 * BODY_SIZE - FIXTURE_OVERLAP),
                Vec4::ZERO,
                BODY_SIZE,
                BODY_MASS,
            )
            .unwrap(),
        );
        world.step(FIXTURE_DT);

        let overlay = only(|o| o.impulses = true);
        let mesh = overlay_mesh(&world, &overlay);
        let contacts = contact_count(&world);
        assert_eq!(mesh.segments.len(), 2 * contacts);

        let mut chunks = mesh.segments.chunks_exact(2);
        let mut braked = 0;
        for (&(a, b), manifold) in world.manifolds.iter() {
            let lead = (world.bodies[a].velocity - world.bodies[b].velocity).truncate();
            for cp in &manifold.points {
                let chunk = chunks.next().expect("two bars per contact");
                let normal_bar = Vec3::from_array(chunk[0].1) - Vec3::from_array(chunk[0].0);
                assert!(
                    normal_bar.dot(world.bodies[b].velocity.truncate()) > 0.0,
                    "normal impulse arrow opposes body B's response"
                );
                let bar = Vec3::from_array(chunk[1].1) - Vec3::from_array(chunk[1].0);
                let expected = cp.tangent_impulse * overlay.impulse_scale;
                assert!(
                    (bar.length() - expected).abs() < TRANSLATE_TOL,
                    "friction bar runs {} world units for a {} accumulator",
                    bar.length(),
                    cp.tangent_impulse
                );
                if cp.tangent_impulse > 0.0 {
                    braked += 1;
                    assert!(
                        bar.dot(lead) > 0.0,
                        "friction bar {bar:?} runs with the slide {lead:?} it should brake"
                    );
                }
            }
        }
        assert!(
            braked > 0,
            "no contact accumulated friction, so the sign pin is vacuous"
        );
    }

    #[test]
    fn one_island_marks_its_bodies_in_one_colour() {
        let world = overlapping_pairs(2);
        let islands = world.islands();
        assert_eq!(islands.len(), 2, "fixture did not split into two islands");

        let mesh = overlay_mesh(&world, &only(|o| o.islands = true));
        let expected: usize = islands
            .iter()
            .map(|i| 3 * i.bodies.len() + i.constraints.len())
            .sum();
        assert_eq!(mesh.segments.len(), expected);

        let mut per_island: Vec<[u32; 4]> = Vec::new();
        for island in &islands {
            let mut colors = std::collections::BTreeSet::new();
            for &id in &island.bodies {
                let centre = world.bodies[id].position.truncate();
                let mut arms = 0;
                for (&(from, to), &(color, _)) in mesh.segments.iter().zip(&mesh.colors) {
                    let mid = (Vec3::from_array(from) + Vec3::from_array(to)) * 0.5;
                    if (mid - centre).length() < TRANSLATE_TOL
                        && Vec3::from_array(from) != Vec3::from_array(to)
                    {
                        arms += 1;
                        colors.insert(color.map(f32::to_bits));
                    }
                }
                assert_eq!(arms, 3, "body {id:?} is not marked by a three-arm cross");
            }
            assert_eq!(
                colors.len(),
                1,
                "island {:?} marked its bodies in {} colours",
                island.id,
                colors.len()
            );
            let color = *colors.iter().next().expect("one colour");
            assert!(
                !per_island.contains(&color),
                "two islands share a colour, so the partition cannot be read off the overlay"
            );
            per_island.push(color);
        }
    }

    #[test]
    fn the_contact_overlay_layers_reach_the_allocator_zero_times() {
        let world = overlapping_pairs(2);
        let overlay = PhysicsOverlay {
            contacts: true,
            normals: true,
            impulses: true,
            ..PhysicsOverlay::default()
        };
        let mut mesh = LineMesh::<3>::default();
        build_physics_overlay_mesh(&world, BODY_SIZE, &overlay, &mut mesh);
        assert!(!mesh.segments.is_empty(), "the fixture emitted nothing");

        let warm = alloc_probe::bytes_allocated_by(|| {
            build_physics_overlay_mesh(&world, BODY_SIZE, &overlay, &mut mesh)
        });
        assert_eq!(
            warm, 0,
            "a warm contact overlay asked the allocator for {warm} bytes"
        );
    }

    #[test]
    fn a_held_toy_marks_its_floor_footprint_under_itself() {
        let mut toybox = settled();
        assert!(toybox.drop_footprint().is_none());
        assert!(grab_centre(&mut toybox, 0));
        let footprint = toybox.drop_footprint().expect("held");
        let centre = toybox.position(0);
        let mid = (footprint.min + footprint.max) * 0.5;
        assert!((mid - Vec2::new(centre.x, centre.z)).length() < 1e-4);
        let half = (footprint.max - footprint.min) * 0.5;
        assert!(
            half.x > 0.5 * BODY_SIZE && half.x <= BODY_SIZE + 1e-4,
            "{half}"
        );
        assert!(
            half.y > 0.5 * BODY_SIZE && half.y <= BODY_SIZE + 1e-4,
            "{half}"
        );

        let target = centre.truncate() + Vec3::new(1.0, 0.0, 0.0);
        for _ in 0..FLICK_TICKS * 6 {
            drag_frame(&mut toybox, target, GrabAxis::Slice);
        }
        let after = toybox.drop_footprint().expect("still held");
        let moved = (after.min + after.max) * 0.5 - mid;
        assert!(
            moved.x > 0.3,
            "the footprint stayed put while the toy moved: {moved}"
        );
    }

    #[test]
    fn a_throw_wakes_a_sleeping_toy_and_sets_its_velocity() {
        let mut toybox = settled();
        assert!(toybox.world.bodies[toybox.toys[1].body].is_sleeping());
        toybox.throw(1, Vec4::new(0.0, 3.0, 0.0, 0.0));
        assert!(!toybox.world.bodies[toybox.toys[1].body].is_sleeping());
        let start = toybox.position(1);
        toybox.tick(TICK_DT);
        assert!(toybox.position(1).y > start.y);
    }

    #[test]
    fn throw_parses_an_optional_w_and_rejects_a_missing_toy() {
        assert_eq!(
            parse_throw(&["2", "1", "2", "3"]).expect("three components"),
            (2, Vec4::new(1.0, 2.0, 3.0, 0.0))
        );
        assert_eq!(
            parse_throw(&["0", "1", "2", "3", "4"])
                .expect("four components")
                .1,
            Vec4::new(1.0, 2.0, 3.0, 4.0)
        );
        assert!(parse_throw(&["9", "0", "0", "0"]).is_err());
        assert!(parse_throw(&["0", "0", "0"]).is_err());
        assert!(parse_throw(&["0", "NaN", "0", "0"]).is_err());
        assert!(parse_throw(&["0", "0", "inf", "0"]).is_err());
    }
}
