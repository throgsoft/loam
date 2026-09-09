use std::f32::consts::FRAC_PI_2;

use glam::Vec4;
use loam_app::session::run;
use loam_math::{Bivector, EuclideanR4, Iso4Flat, IsometryGroup, Plane4, Rotor, Rotor4};
use loam_runtime::host::{HostConfig, HostError};
use loam_runtime::{
    Access, ActionId, AppCommand, Bindings, Command, Commands, Ctx, Dispatch, DomainBuilder,
    DomainHandle, Domains, Entity, Eye, Input, Instance, Key, LogCapacity, Material, MaterialId,
    Outcome, Phase, Pose, PreparedGeometry, Projection4, Rejection, Session, SimConfig,
    SpawnBundle, TypedDomain, Value, ViewSpec,
};

const TWIST: ActionId = ActionId(0);
const SCRAMBLE: ActionId = ActionId(1);
const UNDO: ActionId = ActionId(2);
const RESET: ActionId = ActionId(3);
const TWIST_TICKS: u32 = 20;
const SCRAMBLE_TWISTS: usize = 12;
const PIECE_SPACING: f32 = 0.7;
const STICKER_HALF: f32 = 0.3;
const FOCAL_DISTANCE: f32 = 4.0;
const CELL_COLORS: [[f32; 4]; 8] = [
    [0.9, 0.1, 0.1, 1.0],
    [1.0, 0.5, 0.1, 1.0],
    [0.1, 0.7, 0.2, 1.0],
    [0.2, 0.4, 0.9, 1.0],
    [0.95, 0.9, 0.2, 1.0],
    [0.95, 0.95, 0.95, 1.0],
    [0.6, 0.2, 0.8, 1.0],
    [0.2, 0.8, 0.8, 1.0],
];

#[derive(Clone, Copy, PartialEq, Eq)]
struct Cell {
    axis: usize,
    sign: i8,
}

impl Cell {
    fn direction(self) -> Vec4 {
        let mut direction = Vec4::ZERO;
        direction[self.axis] = f32::from(self.sign);
        direction
    }

    fn color(self) -> usize {
        self.axis * 2 + usize::from(self.sign > 0)
    }

    fn twist_plane(self) -> Plane4 {
        Plane4::ALL
            .into_iter()
            .find(|plane| !plane_axes(*plane).contains(&self.axis))
            .unwrap_or(Plane4::Xy)
    }
}

fn plane_axes(plane: Plane4) -> [usize; 2] {
    match plane {
        Plane4::Xy => [0, 1],
        Plane4::Xz => [0, 2],
        Plane4::Xw => [0, 3],
        Plane4::Yz => [1, 2],
        Plane4::Yw => [1, 3],
        Plane4::Zw => [2, 3],
    }
}

fn grid_point(grid: [i8; 4]) -> Vec4 {
    Vec4::new(
        f32::from(grid[0]),
        f32::from(grid[1]),
        f32::from(grid[2]),
        f32::from(grid[3]),
    )
}

#[derive(Clone, Copy)]
struct Piece {
    grid: [i8; 4],
    orientation: Rotor4,
}

impl Piece {
    fn iso(&self) -> Iso4Flat {
        Iso4Flat {
            rotation: self.orientation,
            translation: grid_point(self.grid) * PIECE_SPACING,
        }
    }

    fn in_cell(&self, cell: Cell) -> bool {
        self.grid[cell.axis] == cell.sign
    }
}

#[derive(Clone, Copy)]
struct Sticker {
    cell: Cell,
}

#[derive(Clone, Copy)]
struct Slot {
    local: Cell,
}

impl Slot {
    fn iso(self) -> Iso4Flat {
        let direction = self.local.direction();
        Iso4Flat {
            rotation: Rotor4::from_rotation_arc(Vec4::W, direction),
            translation: direction * STICKER_HALF,
        }
    }
}

#[derive(Clone, Copy)]
struct Twist {
    cell: Cell,
    plane: Plane4,
    quarter_turns: i8,
}

impl Twist {
    fn rotor(&self, fraction: f32) -> Rotor4 {
        let angle = f32::from(self.quarter_turns) * FRAC_PI_2 * fraction;
        (self.plane.unit_bivector() * angle).exp().normalize()
    }

    fn inverse(self) -> Self {
        Self {
            quarter_turns: -self.quarter_turns,
            ..self
        }
    }
}

#[derive(Clone, Copy)]
struct Turning {
    twist: Twist,
    ticks_left: u32,
}

loam_runtime::stores! {
    #[derive(Default)]
    pub struct CubeStores {
        pieces: Store<Piece>,
        stickers: Store<Sticker>,
        slots: Relation<Slot>,
        history: Value<Vec<Twist>>,
        turning: Value<Option<Turning>>,
        selected: Value<Option<Entity>>,
        rng: Value<u64>,
    }
}

#[derive(Clone, Copy)]
struct Puzzle {
    domain: DomainHandle<EuclideanR4>,
    highlight: MaterialId,
    cell_materials: [MaterialId; 8],
}

fn place_cell(app: &CubeStores, r4: &mut TypedDomain<EuclideanR4>, rotor: Rotor4, cell: Cell) {
    for (piece_entity, piece) in app.pieces.iter() {
        if !piece.in_cell(cell) {
            continue;
        }
        let iso = EuclideanR4.iso_compose(Iso4Flat::from_rotation(rotor), piece.iso());
        if let Some(pose) = r4.poses.get_mut(piece_entity) {
            *pose = Pose::from(iso);
        }
        for id in app.slots.outgoing(piece_entity) {
            let Some(link) = app.slots.get(id) else {
                continue;
            };
            if let Some(pose) = r4.poses.get_mut(link.to) {
                *pose = Pose::from(EuclideanR4.iso_compose(iso, link.data.iso()));
            }
        }
    }
}

fn commit(app: &mut CubeStores, r4: &mut TypedDomain<EuclideanR4>, twist: Twist) {
    let rotor = twist.rotor(1.0);
    for (_, piece) in app.pieces.iter_mut() {
        if !piece.in_cell(twist.cell) {
            continue;
        }
        let turned = rotor.apply(grid_point(piece.grid)).round();
        piece.grid = [
            turned.x as i8,
            turned.y as i8,
            turned.z as i8,
            turned.w as i8,
        ];
        piece.orientation = (rotor * piece.orientation).normalize();
    }
    place_cell(app, r4, Rotor4::IDENTITY, twist.cell);
}

// Knuth, TAOCP vol. 2, 3.3.4, the MMIX linear congruential constants.
fn random_twist(rng: &mut u64) -> Twist {
    *rng = rng
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = *rng >> 33;
    let cell = Cell {
        axis: (bits % 4) as usize,
        sign: if bits & 4 == 0 { -1 } else { 1 },
    };
    Twist {
        cell,
        plane: cell.twist_plane(),
        quarter_turns: if bits & 8 == 0 { 1 } else { -1 },
    }
}

struct TwistCommand {
    twist: Twist,
}

impl AppCommand<CubeStores> for TwistCommand {
    fn name(&self) -> &'static str {
        "twist"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, CubeStores>) -> Result<Outcome, Rejection> {
        if dispatch.app.turning.get().is_some() {
            return Err(Rejection::Unsupported("a twist is in progress"));
        }
        dispatch.app.turning.set(Some(Turning {
            twist: self.twist,
            ticks_left: TWIST_TICKS,
        }));
        dispatch.app.history.get_mut().push(self.twist);
        Ok(Outcome::Done)
    }
}

struct Scramble {
    puzzle: Puzzle,
}

impl AppCommand<CubeStores> for Scramble {
    fn name(&self) -> &'static str {
        "scramble"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, CubeStores>) -> Result<Outcome, Rejection> {
        let r4 = dispatch.domains.typed(self.puzzle.domain)?;
        for _ in 0..SCRAMBLE_TWISTS {
            let twist = random_twist(dispatch.app.rng.get_mut());
            commit(dispatch.app, r4, twist);
            dispatch.app.history.get_mut().push(twist);
        }
        Ok(Outcome::Done)
    }
}

struct Undo {
    puzzle: Puzzle,
}

impl AppCommand<CubeStores> for Undo {
    fn name(&self) -> &'static str {
        "undo"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, CubeStores>) -> Result<Outcome, Rejection> {
        let Some(twist) = dispatch.app.history.get_mut().pop() else {
            return Ok(Outcome::Done);
        };
        let r4 = dispatch.domains.typed(self.puzzle.domain)?;
        commit(dispatch.app, r4, twist.inverse());
        Ok(Outcome::Done)
    }
}

fn sticker_cube() -> PreparedGeometry {
    let corner = |bits: u32| {
        let sign = |bit: u32| {
            if bits & bit == 0 {
                -STICKER_HALF
            } else {
                STICKER_HALF
            }
        };
        [sign(1), sign(2), sign(4), 0.0]
    };
    let mut segments = Vec::new();
    for a in 0..8_u32 {
        for b in (a + 1)..8 {
            if (a ^ b).count_ones() == 1 {
                segments.push([corner(a), corner(b)]);
            }
        }
    }
    PreparedGeometry::Lines4 { segments }
}

fn grid_points() -> impl Iterator<Item = [i8; 4]> {
    (0..81_i32)
        .map(|index| {
            [
                (index % 3 - 1) as i8,
                (index / 3 % 3 - 1) as i8,
                (index / 9 % 3 - 1) as i8,
                (index / 27 - 1) as i8,
            ]
        })
        .filter(|grid| *grid != [0; 4])
}

fn selected_twist(app: &CubeStores) -> Option<Twist> {
    let selected = (*app.selected.get())?;
    let sticker = app.stickers.get(selected)?;
    Some(Twist {
        cell: sticker.cell,
        plane: sticker.cell.twist_plane(),
        quarter_turns: 1,
    })
}

fn main() -> Result<(), HostError> {
    let config = SimConfig::default();
    let stores = CubeStores {
        rng: Value::new(config.seed),
        ..CubeStores::default()
    };
    let mut session = Session::new(stores, config);
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let sticker_geometry = session.prepare(sticker_cube());
    let cell_materials = CELL_COLORS.map(|color| session.add_material(Material::flat(color)));
    let highlight = session.add_material(Material::flat([1.0; 4]));
    let root = session.views().root();
    let puzzle = Puzzle {
        domain: r4,
        highlight,
        cell_materials,
    };

    session.dispatch(|d| -> Result<(), Rejection> {
        for grid in grid_points() {
            let piece = Piece {
                grid,
                orientation: Rotor4::IDENTITY,
            };
            let piece_entity = d.spawn(
                SpawnBundle::new()
                    .at(r4, Pose::from(piece.iso()))
                    .row(piece),
            )?;
            for (axis, &sign) in grid.iter().enumerate() {
                if sign == 0 {
                    continue;
                }
                let cell = Cell { axis, sign };
                let slot = Slot { local: cell };
                let sticker = d.spawn(
                    SpawnBundle::new()
                        .at(
                            r4,
                            Pose::from(EuclideanR4.iso_compose(piece.iso(), slot.iso())),
                        )
                        .instance(Instance::new(
                            sticker_geometry,
                            cell_materials[cell.color()],
                        ))
                        .row(Sticker { cell }),
                )?;
                d.app.slots.link(piece_entity, sticker, slot)?;
            }
        }
        let eye = Pose::at(Vec4::W * FOCAL_DISTANCE);
        let eye = d.spawn(SpawnBundle::new().at(r4, eye))?;
        let projection = Projection4 {
            focal: FOCAL_DISTANCE,
        };
        d.domains
            .typed(r4)?
            .add_view(ViewSpec::new(root, eye, projection));
        Ok(())
    })?;
    session.views_mut().root_mut().eye =
        Eye::looking_at([0.0, 2.0, 6.0], [0.0; 3], [0.0, 1.0, 0.0]);

    session.system(
        Phase::Dispatch,
        "select",
        Access::new().writes::<Option<Entity>>().domain(r4.id()),
        move |ctx: Ctx<'_, CubeStores>| {
            let Some(pointer) = ctx.input.began() else {
                return;
            };
            let Some(pick) = ctx.pick(pointer.ndc) else {
                return;
            };
            if !ctx.app.stickers.contains(pick.entity) {
                return;
            }
            let Ok(r4) = ctx.domains.typed(puzzle.domain) else {
                return;
            };
            if let Some(previous) = *ctx.app.selected.get() {
                if let (Some(instance), Some(sticker)) = (
                    r4.instances.get_mut(previous),
                    ctx.app.stickers.get(previous),
                ) {
                    instance.material = puzzle.cell_materials[sticker.cell.color()];
                }
            }
            if let Some(instance) = r4.instances.get_mut(pick.entity) {
                instance.material = puzzle.highlight;
            }
            ctx.app.selected.set(Some(pick.entity));
        },
    );

    session.system(
        Phase::Dispatch,
        "actions",
        Access::new().reads::<Sticker>().commands(),
        move |app: &mut CubeStores, input: &Input, commands: &mut Commands<CubeStores>| {
            if input.pressed(TWIST) {
                if let Some(twist) = selected_twist(app) {
                    commands.app(TwistCommand { twist });
                }
            }
            if input.pressed(SCRAMBLE) {
                commands.app(Scramble { puzzle });
            }
            if input.pressed(UNDO) {
                commands.app(Undo { puzzle });
            }
            if input.pressed(RESET) {
                commands.submit(Command::Reset);
            }
        },
    );

    session.system(
        Phase::Simulation,
        "turn",
        Access::new().writes::<Piece>().domain(r4.id()),
        move |app: &mut CubeStores, domains: &mut Domains| {
            let Some(mut turning) = *app.turning.get() else {
                return;
            };
            let Ok(r4) = domains.typed(puzzle.domain) else {
                return;
            };
            turning.ticks_left = turning.ticks_left.saturating_sub(1);
            let fraction = 1.0 - turning.ticks_left as f32 / TWIST_TICKS as f32;
            place_cell(app, r4, turning.twist.rotor(fraction), turning.twist.cell);
            if turning.ticks_left == 0 {
                commit(app, r4, turning.twist);
                app.turning.set(None);
            } else {
                app.turning.set(Some(turning));
            }
        },
    );

    session.set_initial()?;
    let bindings = Bindings::new()
        .key(Key::Space, TWIST)
        .key(Key::Letter('s'), SCRAMBLE)
        .key(Key::Letter('u'), UNDO)
        .key(Key::Letter('r'), RESET);
    run(session, HostConfig::new("rubiks4d", bindings))
}
