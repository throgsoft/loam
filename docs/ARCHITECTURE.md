# Architecture

## Crate boundaries

The manifests define the dependency graph. Stable crates never depend on volatile
ones: math, shape, scene, and time sit below physics and the runtime, the renderer
above the runtime, and the hosts above everything. Platform code lives only in
`loam-app`.

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
| `loam-text` | Glyph overlay and extruded letter solids | `loam-shape` |
| `loam-egui` | The debug layer's winit input and console panel | `loam-console`, `loam-math` |
| `loam-app` | The session hosts (winit loop and browser worker), args, console, capture, pacing, traces, the frame script | `loam-console`, `loam-egui`, `loam-input`, `loam-math`, `loam-render`, `loam-runtime`, `loam-time` |
| `loam` | Re-exports | `loam-math`, `loam-render`, `loam-time` |
| `polytope_playground` | The flagship demo on the session host | `loam-app`, `loam-math`, `loam-physics`, `loam-render`, `loam-runtime`, `loam-scene`, `loam-shape` |
| `examples` | The `hero`, `tesseract`, `rubiks4d`, and `twospace` bins | `loam-app`, `loam-math`, `loam-physics`, `loam-render`, `loam-runtime`, `loam-shape`, `loam-text`, `loam-time` |
| `xtask` | Web bundling and the local server | none |

`Space` is the geometry abstraction; `IsometryGroup`, `WgslSpace`, `RasterizableSpace`,
and `PhysicsSpace` express separate requirements, so rendering support does not imply
rigid-body support. WGSL emitted by the CPU crates is a shader ABI shared with
`loam-render`; the renderer owns no simulation state.

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
domains, views, and bridges; `Session::restore` cancels pending commands and work,
replaces the state, and plans the bulk stores `apply_restore` refills; `set_initial`
records the snapshot that `Command::Reset` returns to. `DomainBuilder::new(name, space)`
registers a domain and erases its space once; `Domains::typed` downcasts a handle back
to `TypedDomain<S>`, and everything else sees `Domain`. `ChartPose`, `ChartPoint`, and
`ChartTangent` cross the erasure; `DomainSpace::pose_from_chart` and
`tangent_from_chart` convert them, and `Space::chart_envelope` and `valid_point` bound
them. `ChartCommand::Place`, `Move`, and `Walk` reach a domain's facilities first, and
a facility that claims one applies it in its own world.

## Views and image spaces

A view is a `ViewSpec` of a root image space, an eye entity, and an eye-relative
`ViewMapping` into an image space: `Section4` cuts an R⁴ domain at a `w` plane,
`Projection4` projects it with a focal length, `Identity3` passes an R³ domain
through, and `Klein` takes an H³ domain through the Klein model so geodesics stay
straight. Every mapping reports one projective depth, `projective_depth`, compared
under `DepthConvention::ReversedZ` in a `Depth32Float` attachment, so passes from
different mappings share a depth test. A `Placement` is `Rigid` or `Nonlinear`;
`Views::to_root` composes the placements from an image space to the root, `None` for
an unplaced one. A `Bridge` from `Session::bridge` places a view's image space at an
anchor entity of another domain and unplaces it when either end despawns.
`Views::ray` pulls a pointer ray into an image space; `Domains::pick` takes the nearest
hit by root-eye projective depth across every view that reaches the root, and
`pick_lifted` keeps only views whose mapping has a `ray_lift`. `Session::grab` picks
among lifted views and records a drag plane through the hit facing the root eye,
`Session::drag` moves the entity to where the ray meets that plane through
`ChartCommand::Move`, and `Session::release` returns a `DragRelease` with the release
velocity.

## Presentation

`Session::publish` fills a `Publication` with one `PublishedView` per view that
reaches the root, each carrying `ViewRecords` of instances, segments, and points
stamped with the tick and a sequence. `Records<A>` lends and releases those buffers;
`ViewRecords::built` names the publication that built a view, and `Presenter::upload`
skips one it already uploaded. `MaterialSpec` describes a pipeline, and `Material`
values from `add_material` name what an instance draws with.

A `FramePass` declares a name, the resources it reads and writes (`SCENE_COLOR`,
`SCENE_DEPTH`, or its own), a `PassOrder` of `BeforeScene` or `AfterScene`, and an
optional `DepthConvention`. `PassSchedule::register` orders passes by those edges and
refuses, as a `PassError`, a cycle, a scene output written before the scene, or a
depth convention other than the frame's. `Presenter::record` clears colour and depth
in `present-clear`, records the passes before the scene, draws the views in
`present-draw`, then records the passes after the scene. Every section reports CPU
time and a `GpuTime`, `Unavailable` when the adapter has no timestamp queries.
`Presenter::attach` runs at startup and after a device loss: it drops device objects,
installs the timer, and hands every pass a `FrameFormat` through `FramePass::attach`.
An application publishes into the pass wrappers from its frame hook: `SkyGroundPass`,
`RaymarchPass`, `FieldPass`, `HyperslicePass`, `LinePass`, and `PointPass` each keep
the last `publish` and share one state across clones. `TriangleFeed` holds a mesh,
view, and optional ground for the pass it builds; `edit` bumps a revision, and the
pass uploads once per revision and again after a device loss.

## Physics

`World<S>` validates edits: `set_pose`, `set_mass_properties`, and `set_collider`
return an `EditError` for a stale handle or an invalid value and drop the body's
contacts; `drain_dirty` yields each body spawned, integrated, edited, or restored since
the last drain. The `Physics` facility gives a domain one world, writes every dirty
body's pose into its entity's row each step, and claims `Place`, `Move`, and `Walk`
for entities with a body. `BroadphaseBound::Certified` lets a space prune pairs by
chart distance, seams included; `BroadphaseBound::Unknown` tests every pair that
passes the static and mask filters. The step solves contact islands through
`loam_time::par::for_each_chunk` in chunks of `ISLANDS_PER_SOLVE_WORKER` once
`par::install` has installed an executor, and `SolveReport` counts what ran.
`FieldNarrowphase::sphere_against_field` tests a body against a `DistanceField` and
refuses with a `FieldRefusal` where the gradient is too small to give a normal.
`World::snapshot` returns a `WorldState` of the bodies with their field bindings and
anchor ids, while gravity, solver iterations, narrowphase functions, and field objects
stay with the world; `World::restore` takes it back, and `World::save` and
`World::load` persist a world as text under `PERSIST_VERSION`.

## Fields

A `FieldProgram` is a postfix program over `FieldPrimitive`s, with opcodes such as
`OP_SPHERE`, `OP_UNION`, `OP_SMOOTH_UNION`, and `OP_PUSH_POSE`, a bounded stack, and
the `FieldNode` ball tree over its primitives. `FieldCompiler::compile` builds it from
a domain's `Field` rows and returns a `FieldCost` of changed inputs, affected
dependencies, program layout, index maintenance, and whether it rebuilt in full.
`evaluate` runs it on the CPU; `evaluate_bounded` skips a subtree whose ball cannot
beat the best distance found, with the same hits as the unculled walk.
`FieldMarchNode` interprets the same program on the GPU and, after
`DEFAULT_SPECIALIZE_AFTER` frames of an unchanged program, asks its
`SpecializationBuilder` for a kernel with the program inlined; `InlineBuilder` builds
on the calling thread. [PERF.md](PERF.md) records a thousand spheres: 21000
evaluations per ray unculled against 784 through the hierarchy, a full compile of
1999 nodes reporting 2999 program writes and 3997 index writes, and a pose-only
compile rewriting every node.

## GPU work

`Session::register_bulk` takes a `BulkSpec` naming a store's element size, count,
`Readback`, `SnapshotPolicy`, and `Schedule`. A `WorkItem` registered with
`Session::work` names the bulk stores it writes; each tick orders a `WorkOrder` the
host takes with `issue_work`, marks `submitted`, and lands with `land_readback`, which
reports `Applied`, `Discarded`, or `Failed`. `Readback::Required` suspends the entry
that awaits the item until the rows land, and the suspended call resumes there;
`Optional` never suspends; `None` returns nothing. `Schedule::InStep` orders an item
inside the step and `Ahead` orders it at publication for the tick that follows.
`cancel_work` drops every order in flight and the wait. `checkpoint` stores a bulk
store's rows for a tick; a restore plans `BulkAction::Replace` from the checkpoint for
an `Authoritative` store or `Reinitialize` for a `Reinitializable` one, which
`apply_restore` replays, and leaves a `Derived` store to its next tick. In
`loam-render`, `BulkBuffers` holds the GPU side, `ComputeWork` records the items, and
`Readbacks::poll` maps the staging buffers.

## Hosts

`loam_app::session::run`, `run_with_work`, and `launch` run a session on the native
winit loop or in a browser worker with an `OffscreenCanvas`; `launch` takes a
`SessionApp` carrying the passes, frame hook, console verbs, work recorder, pacing,
capture requests, and debug layer. Each frame the host polls readbacks and lands
them; if the session waits on a readback that has not landed, the frame yields,
resetting the clock and running nothing else until it lands. Otherwise it runs the
boundary with the frame's input (empty when resuming), the ticks the `FixedTimestep`
owes, and the publication; the `FrameHook` then runs with the session, the previous
frame's sections, the egui context, the target size, and a `CaptureControl`; the
presenter records, and the `DebugLayer` records after every pass that writes the
scene. `SessionConsole` holds the verbs from `SessionApp::command`; typed and scripted
lines queue and run at `dispatch_pending`, and a verb's `Submit` pushes `AppCommand`s
to the next boundary. `CaptureControl` queues `start` and `stop` requests the host
drains at the end of the same frame. `Pacer` applies `--fps` and `--vsync`.
`--script=path` loads a `Script` of `frame command` lines; `ScriptDriver::advance_console`
queues each frame's lines on the console, and the host ignores the returned
`ScriptStatus`, so a finished script leaves the window open.
`loam_runtime::host::run_headless` runs boundaries, ticks, and publication with no
GPU. `cargo xtask web` bundles a bin for the browser and `cargo xtask serve` serves it.

## Verification

Repository hooks reject em dashes, arrows, deferred-work markers, and rustfmt drift at
edit time. Each row runs these gates on its crates before review, and CI runs them on
the workspace.

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
compile, traversal, hierarchy, and contact figures; a claim about frame time or
hot-path cost points at a block there. The headless bins print lines a person checks
by hand while their tests hold the analytic values: `twospace --headless N` prints the
pick through the bridge, the dragged position, each domain's landmark pick, and the
ball's height after `N` ticks; `hero --headless N` prints the first letter's height;
`polytope_playground --headless` prints the active polytope with its edge count, the
published segment count, and the frame's section list.

## Open items

Field binding into a domain's physics waits on a replace operation in `loam-physics`.
A threaded `SpecializationBuilder` is not installed; `InlineBuilder` compiles on the
calling thread. No exit channel exists for a finished script or a stopped capture, so
the host runs until a person closes the window. The playground's move onto the session
host dropped its HUD, the hypergimbal drag control, filmstrip capture, the director
timelines and composer, projection modes, colour schemes, spin presets, the wider verb
table (`alpha`, `perspective`, `wide`, `width`, `wireframe`), and throwing on release;
a later row restores them. `EuclideanR4` leaves `pose_from_chart` and
`tangent_from_chart` unimplemented. `cargo xtask web` builds the playground in a debug
profile unless `--release` is passed, skipping `wasm-opt`, and the debug bundle's size
is not recorded.
