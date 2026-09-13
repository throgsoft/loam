use std::f32::consts::FRAC_PI_2;

use glam::Vec4;
use loam::app::session::{launch, SessionApp};
use loam::math::{Bivector, EuclideanR4, Iso4Flat, IsometryGroup, Plane4, Rotor, Rotor4};
use loam::runtime::host::{HostConfig, HostError};
use loam::runtime::{
    ActionId, AppCommand, Bindings, Command, Ctx, Dispatch, DomainBuilder, DomainError,
    DomainHandle, Eye, Instance, Key, LogCapacity, Material, MaterialId, Outcome, Phase, Pose,
    PreparedGeometry, Projection4, Rejection, Session, SimConfig, SpawnBundle, TypedDomain, Value,
    ViewSpec,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    selected: bool,
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

loam::runtime::stores! {
    #[derive(Default)]
    pub struct CubeStores {
        pieces: Store<Piece>,
        stickers: Store<Sticker>,
        slots: Relation<Slot>,
        history: Value<Vec<Twist>>,
        turning: Value<Option<Turning>>,
        rng: Value<u64>,
    }
}

#[derive(Clone, Copy)]
struct Puzzle {
    domain: DomainHandle<EuclideanR4>,
    highlight: MaterialId,
    cell_materials: [MaterialId; 8],
}

fn place_cell(
    app: &CubeStores,
    r4: &mut TypedDomain<EuclideanR4>,
    rotor: Rotor4,
    cell: Cell,
) -> Result<(), DomainError> {
    for (piece_entity, piece) in app.pieces.iter() {
        if !piece.in_cell(cell) {
            continue;
        }
        let iso = EuclideanR4.iso_compose(Iso4Flat::from_rotation(rotor), piece.iso());
        r4.set_pose(piece_entity, Pose::from(iso))?;
        for id in app.slots.outgoing(piece_entity) {
            let Some(link) = app.slots.get(id) else {
                continue;
            };
            let pose = Pose::from(EuclideanR4.iso_compose(iso, link.data.iso()));
            r4.set_pose(link.to(), pose)?;
        }
    }
    Ok(())
}

fn commit(
    app: &mut CubeStores,
    r4: &mut TypedDomain<EuclideanR4>,
    twist: Twist,
) -> Result<(), DomainError> {
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
    place_cell(app, r4, Rotor4::IDENTITY, twist.cell)
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
        if dispatch.app.turning.get().is_some() {
            return Err(Rejection::Unsupported("a twist is in progress"));
        }
        let r4 = dispatch.domains.typed(self.puzzle.domain)?;
        for _ in 0..SCRAMBLE_TWISTS {
            let twist = random_twist(dispatch.app.rng.get_mut());
            commit(dispatch.app, r4, twist)?;
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
        if dispatch.app.turning.get().is_some() {
            return Err(Rejection::Unsupported("a twist is in progress"));
        }
        let Some(twist) = dispatch.app.history.get_mut().pop() else {
            return Ok(Outcome::Done);
        };
        let r4 = dispatch.domains.typed(self.puzzle.domain)?;
        commit(dispatch.app, r4, twist.inverse())?;
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
    let (_, sticker) = app.stickers.iter().find(|(_, sticker)| sticker.selected)?;
    Some(Twist {
        cell: sticker.cell,
        plane: sticker.cell.twist_plane(),
        quarter_turns: 1,
    })
}

fn build(
    args: loam::app::args::Args,
) -> Result<(Session<CubeStores>, SessionApp<CubeStores>), HostError> {
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
                        .row(Sticker {
                            cell,
                            selected: false,
                        }),
                )?;
                d.link(piece_entity, sticker, slot)?;
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
        move |ctx: Ctx<'_, CubeStores>| -> Result<(), DomainError> {
            let Some(pointer) = ctx.input.began() else {
                return Ok(());
            };
            let Some(pick) = ctx.pick(pointer.ndc) else {
                return Ok(());
            };
            if !ctx.app.stickers.contains(pick.entity) {
                return Ok(());
            }
            let previous = ctx
                .app
                .stickers
                .iter()
                .find_map(|(entity, sticker)| sticker.selected.then_some(entity));
            if previous == Some(pick.entity) {
                return Ok(());
            };
            let r4 = ctx.domains.typed(puzzle.domain)?;
            if let Some(previous) = previous {
                let sticker = ctx
                    .app
                    .stickers
                    .get_mut(previous)
                    .ok_or(DomainError::Stale(previous))?;
                let material = puzzle.cell_materials[sticker.cell.color()];
                sticker.selected = false;
                let instance = r4
                    .instance_mut(previous)
                    .ok_or(DomainError::Stale(previous))?;
                instance.material = material;
            }
            let sticker = ctx
                .app
                .stickers
                .get_mut(pick.entity)
                .ok_or(DomainError::Stale(pick.entity))?;
            sticker.selected = true;
            let instance = r4
                .instance_mut(pick.entity)
                .ok_or(DomainError::Stale(pick.entity))?;
            instance.material = puzzle.highlight;
            Ok(())
        },
    );

    session.system(
        Phase::Dispatch,
        "actions",
        move |ctx: Ctx<'_, CubeStores>| {
            if ctx.input.pressed(TWIST) {
                if let Some(twist) = selected_twist(ctx.app) {
                    ctx.commands.app(TwistCommand { twist });
                }
            }
            if ctx.input.pressed(SCRAMBLE) {
                ctx.commands.app(Scramble { puzzle });
            }
            if ctx.input.pressed(UNDO) {
                ctx.commands.app(Undo { puzzle });
            }
            if ctx.input.pressed(RESET) {
                ctx.commands.submit(Command::Reset);
            }
            Ok(())
        },
    );

    session.system(
        Phase::Simulation,
        "turn",
        move |ctx: Ctx<'_, CubeStores>| -> Result<(), DomainError> {
            let Some(mut turning) = *ctx.app.turning.get() else {
                return Ok(());
            };
            let r4 = ctx.domains.typed(puzzle.domain)?;
            turning.ticks_left = turning.ticks_left.saturating_sub(1);
            let fraction = 1.0 - turning.ticks_left as f32 / TWIST_TICKS as f32;
            place_cell(
                ctx.app,
                r4,
                turning.twist.rotor(fraction),
                turning.twist.cell,
            )?;
            if turning.ticks_left == 0 {
                commit(ctx.app, r4, turning.twist)?;
                ctx.app.turning.set(None);
            } else {
                ctx.app.turning.set(Some(turning));
            }
            Ok(())
        },
    );

    session.set_initial()?;
    let bindings = Bindings::new()
        .key(Key::Space, TWIST)
        .key(Key::Letter('s'), SCRAMBLE)
        .key(Key::Letter('u'), UNDO)
        .key(Key::Letter('r'), RESET);
    let app =
        SessionApp::with_args(HostConfig::new("rubiks4d", bindings), args).recover_on_fault(RESET);
    Ok((session, app))
}

fn main() -> Result<(), HostError> {
    launch(build)
}

#[cfg(test)]
mod tests {
    use loam::app::args::Args;
    use loam::runtime::{ActionEvent, Input, Pointer, PointerButton, PointerPhase, Publication};

    use super::*;

    #[test]
    fn restored_selection_keeps_its_highlight_and_drives_a_twist() {
        let (mut session, _) = build(Args::default()).unwrap();
        let mut publication = Publication::default();
        session.publish(&mut publication).unwrap();

        let (ndc, picked) = publication
            .views
            .iter()
            .flat_map(|view| {
                view.records
                    .instances
                    .rows()
                    .iter()
                    .map(|record| (view.placement.apply(record.image_point), record.entity))
            })
            .filter_map(|(point, entity)| {
                session
                    .views()
                    .ndc(point)
                    .and_then(|ndc| session.pick(ndc).map(|pick| (ndc, entity, pick)))
            })
            .find_map(|(ndc, entity, pick)| (entity == pick.entity).then_some((ndc, pick)))
            .expect("a published sticker was visible to the root view");
        let selected_cell = session
            .app
            .stickers
            .get(picked.entity)
            .expect("the pick was a sticker")
            .cell;
        session
            .boundary(Input {
                pointers: vec![Pointer {
                    id: 0,
                    button: Some(PointerButton::Primary),
                    ndc,
                    delta: [0.0; 2],
                    phase: PointerPhase::Began,
                    time: 0.0,
                }],
                ..Input::default()
            })
            .unwrap();
        assert_eq!(
            session
                .app
                .stickers
                .get(picked.entity)
                .map(|sticker| (sticker.cell, sticker.selected)),
            Some((selected_cell, true))
        );
        session.publish(&mut publication).unwrap();
        let highlight = publication
            .views
            .iter()
            .flat_map(|view| view.records.instances.rows())
            .find(|record| record.entity == picked.entity)
            .map(|record| record.material)
            .expect("the selected sticker was published");

        let snapshot = session.snapshot().unwrap();
        session.restore(&snapshot).unwrap();
        let (restored, sticker) = session
            .app
            .stickers
            .iter()
            .find(|(_, sticker)| sticker.selected)
            .expect("restore kept the selected sticker");
        assert_ne!(restored, picked.entity);
        assert_eq!(sticker.cell, selected_cell);
        session.publish(&mut publication).unwrap();
        let restored_highlight = publication
            .views
            .iter()
            .flat_map(|view| view.records.instances.rows())
            .find(|record| record.entity == restored)
            .map(|record| record.material)
            .expect("the restored selected sticker was published");
        assert_eq!(restored_highlight, highlight);
        session
            .boundary(Input {
                actions: vec![ActionEvent {
                    action: TWIST,
                    pressed: true,
                }],
                ..Input::default()
            })
            .unwrap();
        assert_eq!(
            (*session.app.turning.get()).map(|turning| turning.twist.cell),
            Some(selected_cell)
        );
    }

    #[test]
    fn busy_history_commands_leave_state_unchanged_until_twist_finishes() {
        let (mut session, _) = build(Args::default()).unwrap();
        let grids = |session: &Session<CubeStores>| {
            let mut grids: Vec<_> = session
                .app
                .pieces
                .iter()
                .map(|(entity, piece)| (entity.key(), piece.grid))
                .collect();
            grids.sort_unstable_by_key(|(key, _)| *key);
            grids
        };
        let history = |session: &Session<CubeStores>| {
            session
                .app
                .history
                .get()
                .iter()
                .map(|twist| (twist.cell, twist.plane, twist.quarter_turns))
                .collect::<Vec<_>>()
        };
        let (cubie, initial_grid) = session
            .app
            .pieces
            .iter()
            .find(|(_, piece)| piece.grid == [1, 1, 0, 1])
            .map(|(entity, piece)| (entity, piece.grid))
            .expect("the known cubie exists");
        let initial_grids = grids(&session);
        let initial_history = history(&session);
        let twist = Twist {
            cell: Cell { axis: 3, sign: 1 },
            plane: Plane4::Xy,
            quarter_turns: 1,
        };

        session
            .dispatch(|dispatch| TwistCommand { twist }.apply(dispatch))
            .unwrap();
        let busy_grids = grids(&session);
        let busy_history = history(&session);
        let busy_rng = *session.app.rng.get();
        session
            .boundary(Input {
                actions: vec![
                    ActionEvent {
                        action: SCRAMBLE,
                        pressed: true,
                    },
                    ActionEvent {
                        action: UNDO,
                        pressed: true,
                    },
                ],
                ..Input::default()
            })
            .unwrap();
        assert_eq!(session.results().len(), 2);
        assert!(session.results().iter().all(|result| {
            result.outcome == Err(Rejection::Unsupported("a twist is in progress"))
        }));
        assert_eq!(grids(&session), busy_grids);
        assert_eq!(history(&session), busy_history);
        assert_eq!(*session.app.rng.get(), busy_rng);

        for _ in 0..TWIST_TICKS {
            session.tick().unwrap();
        }
        assert_eq!(
            session
                .app
                .pieces
                .get(cubie)
                .expect("the cubie exists")
                .grid,
            [-1, 1, 0, 1]
        );

        session
            .boundary(Input {
                actions: vec![ActionEvent {
                    action: UNDO,
                    pressed: true,
                }],
                ..Input::default()
            })
            .unwrap();
        assert_eq!(grids(&session), initial_grids);
        assert_eq!(history(&session), initial_history);
        assert_eq!(
            session
                .app
                .pieces
                .get(cubie)
                .expect("the cubie exists")
                .grid,
            initial_grid
        );
    }
}
