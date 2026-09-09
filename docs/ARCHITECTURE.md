# Architecture

## Crate boundaries

The manifests define the dependency graph. Stable crates never depend on volatile
ones: math, shape, scene, and time sit below physics and the runtime, the renderer above
the runtime, and the hosts above everything. Platform code lives only in `loam-app`.

| Crate | Owns | Loam dependencies |
|---|---|---|
| `loam-math` | Spaces, isometries, rotors, projections, geometry WGSL | none |
| `loam-shape` | Shapes, 4-polytope topology, cross-sections, isovolumes, the distance-field trait | `loam-math` |
| `loam-scene` | CSG scenes, CPU evaluation, loading, edits, WGSL emission | `loam-math`, `loam-shape` |
| `loam-time` | Fixed timestep, frame traces, replay data, timelines, the parallel shim | `loam-math` |
| `loam-input` | Mouse and keyboard accumulator | none |
| `loam-console` | Console model | none |
| `loam-physics` | Bodies, integration, collision, contacts, constraints, field contacts, certified broadphase, per-space features, persistent saves | `loam-math`, `loam-shape`, `loam-time` |
| `loam-runtime` | Entities, typed stores, domains, views and bridges, phases, commands, work orders and bulk stores, field programs, an optional physics world per domain | `loam-math`, `loam-shape`, and `loam-physics` behind the `physics` feature |
| `loam-render` | The GPU context, records, view maps and projective depth, materials, passes and the schedule, the presenter, raster and raymarch nodes, the field interpreter and specialization, work items and readbacks | `loam-math`, `loam-runtime`, `loam-shape`, `loam-time` |
| `loam-text` | Glyph overlay, extruded letter solids, and the text pass | `loam-shape`, `loam-render` |
| `loam-egui` | The debug layer's winit input and console panel | `loam-console`, `loam-math` |
| `loam-app` | The session hosts (winit loop and browser worker), args, console, capture, pacing, traces, the frame script | `loam-console`, `loam-egui`, `loam-input`, `loam-math`, `loam-render`, `loam-runtime`, `loam-time` |
| `loam` | Re-exports | `loam-math`, `loam-render`, `loam-time` |
| `polytope_playground` | The flagship demo on the session host | `loam-app`, `loam-math`, `loam-physics`, `loam-render`, `loam-runtime`, `loam-scene`, `loam-shape` |
| `examples` | The `hero`, `tesseract`, `rubiks4d`, and `twospace` bins | `loam-app`, `loam-math`, `loam-physics`, `loam-render`, `loam-runtime`, `loam-shape`, `loam-text`, `loam-time` |
| `xtask` | Web bundling and the local server | none |

`Space` is the geometry abstraction; `IsometryGroup`, `WgslSpace`, `RasterizableSpace`, and
`PhysicsSpace` express separate requirements; the CPU crates' WGSL is a shader ABI with `loam-render`.

## The session

`Session<A>` owns the application's stores, domains, views, bridges, commands, and work
orders. `A` is declared with the `stores!` macro from `Store<T>`, `Value<T>`, and
relation fields, which derives its snapshot, publication, and change logs. The phases
run in order: `Phase::Dispatch`, `Simulation`, `Publication`, `Presentation`. A system
registers in one phase with an `Access` naming its stores, commands, domains, and
awaited work; `system_at` places it before or after an entry such as `DOMAIN_STEP`.
`Session::boundary` commits deferred commands, then runs each Dispatch entry and the
commands it submitted; `Session::tick` runs the Simulation entries once per fixed step
with the domain step among them; `Session::publish` runs the Publication entries,
stamps the record buffers, then runs the Presentation entries.

Commands are `Command` values and `AppCommand<A>` implementations applied through
`Dispatch` at the boundary, each returning an `Outcome` or a `Rejection` that
`Session::results` keeps for the frame. `Session::snapshot` captures stores, entities,
domains, views, and bridges. `Session::restore` validates the domain count, the
checkpoints, and every domain through `Domain::check_restore` before it changes
anything; then it cancels pending commands and work, replaces the state, rebases the
entity references the engine owns, such as a field's operands, and plans the bulk
stores `apply_restore` refills; `set_initial` records the snapshot `Command::Reset`
returns to. `DomainBuilder::new(name, space)` erases the space once, and `Domains::typed`
downcasts a handle back to `TypedDomain<S>`. `ChartPose`, `ChartPoint`, and
`ChartTangent` cross the erasure; `DomainSpace::pose_from_chart` and `tangent_from_chart`
convert them, and `Space::chart_envelope` and `valid_point` bound them.
`ChartCommand::Place`, `Attach`, `Move`, and `Walk` reach a domain's facilities first; a
facility that claims one applies it in its own world, and the domain applies the rest.

## Views and image spaces

A view is a `ViewSpec` of a root image space, an eye entity, and an eye-relative
`ViewMapping` into an image space: `Section4` cuts an R⁴ domain at a `w` plane,
`Projection4` projects it with a focal length, `Identity3` passes an R³ domain through,
and `Klein` takes an H³ domain through the Klein model so geodesics stay straight. Every
mapping reports one projective depth, `projective_depth`, compared under
`DepthConvention::ReversedZ` in a `Depth32Float` attachment, so passes from different
mappings share a depth test. `Views::to_root` composes `Rigid` or `Nonlinear`
placements from an image space to the root, `None` for an unplaced one; a `Bridge` from
`Session::bridge` places a view's image space at an anchor entity of another domain and
unplaces it when either end despawns. `Domains::pick` casts the ray `Views::ray` pulls
into each image space and takes the nearest hit by root-eye projective depth;
`pick_lifted` keeps only views whose mapping has a `ray_lift`, and `Session::grab`,
`drag`, and `release` run a drag on that pick through a plane facing the root eye and
`ChartCommand::Move`, returning a `DragRelease` with the release velocity.

## Presentation

`Session::publish` fills a `Publication` with one `PublishedView` per view that
reaches the root, each carrying `ViewRecords` of instances, segments, triangles, and
points stamped with the tick and a sequence. An `Instance` with `section` set on a
`PreparedGeometry::Polytope4` also publishes the perimeter and fan-filled faces of its
cut at the map's `SectionCut`, and its `EdgeShading` reads segment colours from a
palette or a w-depth ramp instead of the material. `Records<A>` lends and releases those
buffers; `ViewRecords::built` names the publication that built a view, and
`Presenter::upload` skips a slot whose domain, view, and stamp it already uploaded.
`MaterialSpec` describes a pipeline, and `Material` names what an instance draws with.

A `FramePass` declares a name, the resources it reads and writes (`SCENE_COLOR`,
`SCENE_DEPTH`, or its own), a `PassOrder` of `BeforeScene` or `AfterScene`, and an
optional `DepthConvention`. `PassSchedule::register` appends the pass and recomputes a
stable topological order, registration order breaking ties, and refuses as a `PassError`
a true cycle, a scene output written before the scene, or a depth convention other than
the frame's. `Presenter::record` clears colour and depth in `present-clear`, records the
passes before the scene, draws the views in `present-draw`, then records the passes
after the scene; every section reports CPU time and a `GpuTime`. `Presenter::attach`
runs at startup and after a device loss and attaches every pass with a `FrameFormat`.
The wrappers `SkyGroundPass`, `RaymarchPass`, `FieldPass`, `HyperslicePass`,
`LinePass`, `PointPass`, and `loam_text::TextPass` each keep the last `publish` from the
frame hook and share one state across clones; `TriangleFeed` uploads its mesh once per
`edit` and after a device loss, and the presenter draws the section fills through one.

## Physics

`World<S>` validates edits: `set_pose`, `set_mass_properties`, and `set_collider` return
an `EditError` for a stale handle or an invalid value and drop the body's contacts;
`drain_dirty` yields each body spawned, integrated, edited, or restored since the last
drain. The `Physics` facility gives a domain one world, writes every dirty body's pose
into its entity's row each step, and claims `Place`, `Move`, and `Walk` for entities
with a body. `BroadphaseBound::Certified` lets a space prune pairs by chart distance;
`Unknown` tests every pair that passes the static and mask filters. The step solves
contact islands through `loam_time::par::for_each_chunk` in chunks of
`ISLANDS_PER_SOLVE_WORKER` once `par::install` has installed an executor.
`FieldNarrowphase::sphere_against_field` refuses, with a `FieldRefusal`, a bound or a
gradient too small to give a normal.
`World::snapshot` returns a `WorldState` of the bodies with their field bindings and
anchor ids, while gravity, solver iterations, narrowphase functions, and field objects
stay with the world; `World::check_restore` refuses a state whose registrations or
anchors differ, `World::restore` takes one back, and `World::save` and `World::load`
persist a world as text under `PERSIST_VERSION`.

## Fields

A `FieldProgram` is a postfix program over `FieldPrimitive`s, with opcodes such as
`OP_SPHERE`, `OP_UNION`, `OP_SMOOTH_UNION`, and `OP_PUSH_POSE`, a bounded stack, and
the `FieldNode` ball tree over its primitives. `FieldCompiler::compile` builds it from
a domain's `Field` rows and returns a `FieldCost` of changed inputs, affected
dependencies, program layout, index maintenance, and whether it rebuilt in full.
`FieldOp::result_kind` marks intersection, subtraction, and smooth union as
`ConservativeBound`, which the contact query refuses, and keeps union exact outside the
solid; a `Sphere` or `Box`, which evaluates only x, y, and z, is unbounded along a
fourth chart axis, so its ball never culls it. `evaluate` runs the program on the CPU;
`evaluate_bounded` skips a subtree whose ball cannot beat the best distance found, with
the same hits as the unculled walk. `FieldMarchNode` interprets the same program on the
GPU and, after `DEFAULT_SPECIALIZE_AFTER` frames of an unchanged program, asks its
`SpecializationBuilder` for a pipeline with the program inlined; `InlineBuilder` builds
on the calling thread; [PERF.md](PERF.md) records the thousand-sphere costs, 21000
evaluations per ray unculled against 784 through the hierarchy.

## GPU work

`Session::register_bulk` takes a `BulkSpec` naming a store's element size, count,
`Readback`, `SnapshotPolicy`, and `Schedule`. A `WorkItem` registered with
`Session::work` names the bulk stores it writes; each tick orders a `WorkOrder` the
host takes with `issue_work`, marks `submitted`, and lands with `land_readback`, which
reports `Applied`, `Discarded`, or `Failed`. `Readback::Required` holds the entry that
awaits the item from the tick the order is planned for until the rows land, and the
suspended call resumes there; `Optional` never holds; `None` returns nothing.
`Schedule::InStep` orders an item inside the step and `Ahead` orders it at publication
for the tick that follows. `cancel_work` drops planned orders, orders in flight, and
the wait. `checkpoint` stores a bulk store's rows for a tick; a restore plans
`BulkAction::Replace` from the checkpoint for an `Authoritative` store or
`Reinitialize` for a `Reinitializable` one, which the host replays through
`apply_restore`, and leaves a `Derived` store to its next tick. `loam-render` holds the
GPU side in `BulkBuffers`, `ComputeWork`, and `Readbacks`.

## Hosts

`loam_app::session::run`, `run_with_work`, and `launch` run a session on the native
winit loop or in a browser worker with an `OffscreenCanvas`; `launch` takes a
`SessionApp` carrying the passes, frame hook, console verbs, work recorder, pacing,
capture requests, and debug layer. Each frame the host polls readbacks and lands them;
if the session waits on a readback that has not landed, the frame yields, resetting the
clock and running nothing else until it lands. Otherwise it runs the boundary with the
frame's input (empty when resuming), the ticks the `FixedTimestep` owes, and the
publication; the `FrameHook` then runs with the session, the previous frame's sections,
the egui context, the target size, and a `CaptureControl`; the host reads the root eye
after the hook, drains the bulk restore plan through `apply_restore` before it issues
work, and records the presenter, with the `DebugLayer` after every pass that writes the
scene. On device loss the host rebuilds its device objects and restores a coherent
session and checkpoint pair, or stops with an error naming the store it cannot pair; a
recovered browser worker resumes its animation loop. `SessionConsole` holds the verbs
from `SessionApp::command`; typed and scripted lines run at `dispatch_pending`, and a
verb's `Submit` pushes `AppCommand`s to the next boundary. `CaptureControl` queues
`start` and `stop` requests drained at the end of the same frame, `Pacer` applies
`--fps` and `--vsync`, and `--script=path` loads a `Script` whose lines
`ScriptDriver::advance_console` queues per frame; a finished script leaves the window
open. `loam_runtime::host::run_headless` runs boundaries, ticks, and publication with no
GPU; `cargo xtask web` bundles a bin for the browser and `cargo xtask serve` serves it.

## Verification

Hooks reject em dashes, arrows, deferred-work markers, and rustfmt drift at edit time;
each row runs these gates on its crates before review, and CI runs them on the workspace.

| Gate | What it checks |
|---|---|
| `cargo fmt --all --check` | Formatting |
| `cargo clippy --workspace --all-targets -- -D warnings` | Lints, with undocumented unsafe blocks denied |
| `cargo test -p <crate>`; on CI the workspace, then the `gpu_probe` tests under `--include-ignored` on a software Vulkan adapter | Tests, including the GPU probes |
| `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` | Rustdoc warnings and broken links |
| `cargo build --target wasm32-unknown-unknown` for the playground without default features and for the CPU crates with `examples` | The browser build |
| A grep for `target_arch = "wasm32"` outside `loam-app` and `loam-console` | Platform code stays in the hosts |

[PERF.md](PERF.md) holds stamped measurements: the trace capture method, the physics
island solve and broadphase sweep, the presentation edit-to-result path, and the field
compile, traversal, hierarchy, and contact figures. The headless bins print lines a
person checks by hand while their tests hold the analytic values: `twospace` prints the
pick through the bridge, the dragged position, each domain's landmark pick, and the
ball's height; `hero` prints the first letter's height; `polytope_playground` prints the
composer probe, the active polytope and its edges, the segment, fill, and filmstrip
counts, and the frame's section list.

## Open items

Field binding into a domain's physics waits on a replace operation in `loam-physics`.
A threaded `SpecializationBuilder` is not installed and the host has no install point
for one. Device recovery with an authoritative store needs `Session::checkpoint_tick`
so the gate compares ticks without cloning the session, and an in-place pair capture so
a frame with landings stays allocation-free; `Facility::check_restore` defaults to
accept. No exit channel exists
for a finished script or a stopped capture. The playground still lacks the Active colour
mode, the director timelines, composer drag-and-drop, arrow-key shortcuts (`Key` has no
arrows), the gimbal's translate shafts, three simultaneous projections, conformal caps
for the stereographic and Schlegel maps, a schedule-level viewport grid (the filmstrip
lives in the hyperslice pass), the `alpha`, `wide`, `width`, and `wireframe` verbs, and
throwing on release; scrubbing boxes a new mapping per changed frame because
`ViewSpec::mapping` cannot be mutated in place. `cargo xtask web` builds a debug profile
unless `--release` is passed, skipping `wasm-opt`; the debug bundle's size is not
recorded.
