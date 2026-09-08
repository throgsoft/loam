# Scripted Playground implementation plan

Status: proposed implementation sequence, based on the `cleanup` working tree.

The first deliverable is a scripted Toybox that runs natively and in the
browser. The final deliverable is Polytope Playground with its game rules,
scene setup, authoring controls, UI definitions, and console commands in
Rhai. Rust owns geometry, simulation, input mechanics, rendering, and platform
integration. The binary retains startup and binding registration.

Keep the [project thesis](THESIS.md) as the project's thesis. Record implemented
contracts in [Architecture](ARCHITECTURE.md). Update that document as each
change lands; do not describe this plan as existing architecture.

## 1. Scope and order

Each row is a reviewable implementation step. Split larger steps into
smaller changes with the same acceptance criteria.

| Unit | Work | Depends on | Exit condition |
|---|---|---|---|
| 1 | Coherent world mutations, scoped handles, canonical shape and pose | Current cleanup | Rust callers cannot leave the affected collision caches stale through the new operations |
| 2 | Headless Playground session, ordered Toybox commands, host configuration and lifecycle parity | Unit 1 for mutation routing | The same typed Toybox command sequence can run without a window or GPU |
| 3 | Optional Rhai adapter, source bundles, callback and failure contracts | Unit 2 | A script can query and control existing toys on native and browser |
| 4 | Scripted Toybox setup, interaction policy, and one retained UI panel | Unit 3 | Spawn, grab, throw, pause, and respawn work from the same scripts on both hosts |
| 5 | Shared presentation operations and script UI support | Unit 4; extract earlier where Unit 4 needs it | Rotate and Toybox use the same reusable native operations for their common layers |
| 6 | Scripted Rotate, catalogue, Composer, and timeline binding | Units 1, 3, 5 | Their current behavior comes from scripts and native geometry operations |
| 7 | Remaining UI, filmstrip, HUD, console, and deletion of Rust game policy | Units 4, 6 | The complete Playground uses scripts without a parallel Rust implementation |
| 8 | Device capabilities, offscreen rendering, and additional space bindings | After the first scripting deliverable | Separate follow-up changes driven by the next concrete app |

Units 1 and 2 contain the prerequisites for bindings. Do not make full UI
extraction, a new renderer, an editor, audio, or an atlas system prerequisites
for Unit 3. Route each feature through typed operations before moving that
feature into scripts.

Use these file boundaries. New names are proposed module locations.

| Unit | Main files |
|---|---|
| 1 | `loam-physics/src/{world,body,euclidean_r4}.rs`; the required `loam-shape` helpers; current Playground and Hero callers |
| 2 | Playground `Cargo.toml`, new `src/{lib,session,command,model}.rs`, existing `main.rs` and `toybox.rs`; `loam-app/src/lib.rs` and `wasm/worker.rs` |
| 3 | Workspace manifest and lockfile; new `loam-rhai` package; `loam-app` startup and worker messages; Playground frontend registration |
| 4 | `polytope_playground/assets/game/{game.ron,toybox.rhai}`; initial frontend panel adapter; Trunk asset declarations |
| 5 | Reusable mesh and presentation modules in `loam-render`; native widgets in `loam-egui`; Playground frontend adapter |
| 6 | `assets/game/{catalog,rotate,composer}.rhai`; existing timeline assets; their native adapters |
| 7 | Remaining script UI modules; `loam-app/src/shell.rs`; `loam-console/src/lib.rs`; obsolete Playground modules listed in section 7 |
| All | Focused tests beside their owning operations, `.github/workflows/ci.yml`, API docs, and `docs/ARCHITECTURE.md` |

## 2. Ownership and package boundaries

Add one package, `crates/loam-rhai`. Keep reusable operations in their current
owners. Do not add a universal game runtime, entity framework, scheduler, or
asset server for this migration.

| Owner | Planned responsibility | Dependency rule |
|---|---|---|
| `loam-math` | Space operations, points, tangents, rotations, numerical kernels | No Rhai, app, UI, or GPU dependency |
| `loam-shape` | Canonical geometry, sections, projections, shape preparation | No Rhai, app, UI, or GPU dependency |
| `loam-physics` | World mutation, body references, mass properties, integration, queries | No Rhai, app, UI, or GPU dependency |
| `loam-scene` | CSG and field evaluation | Remains a field scene, not the game object store |
| `loam-time` | Fixed clocks, sampled animation, replay data | No script or presentation ownership |
| `loam-rhai` | VM lifecycle, source resolver, explicit bindings, diagnostics | CPU dependencies only; no dependency on the `loam` umbrella |
| Playground library | Concrete session composition, command dispatch, binding installation | CPU build works with frontend dependencies disabled |
| `loam-render` | Native presentation operations, GPU resources, upload and pass order | Consumes committed state; does not own the simulation |
| `loam-egui` | Native widgets used by the frontend's script UI interpreter | Does not expose `egui::Ui` to scripts |
| `loam-app` | Native/browser hosts, loading, input translation, clocks, host tools | Owns platform and presentation lifetimes |
| Playground scripts | Game state, scene definitions, policies, UI descriptions | Use registered operations and opaque resources |

Make Playground's graphics dependencies optional under `frontend`. Add a CPU
library target and mark its binary `required-features = ["frontend"]`.
Keep normal desktop defaults as `frontend` plus `capture`. Make `capture`
enable the frontend. Rhai is part of the Playground library; it needs no
separate demo feature switch.

Gate `loam-app`, `loam-camera`, `loam-render`, `loam-text`, `loam-egui`,
`wgpu`, `winit`, and the font dependency with the frontend. Check the resolved
dependency graph. A new `lib.rs` alone does not remove those dependencies.
`loam-camera` currently brings in `winit` through `loam-input`; keep screen
input and camera conversion in the frontend for the first headless session.

The session owns the world, game object records, script instance, ordered
commands, and retained query storage. The frontend owns cameras, GPU nodes,
mesh scratch, and widget state. Game object records connect authored object
keys, geometry, optional bodies, and visual settings. They do not become a
general ECS. A game without rigid bodies must be able to use `loam-rhai`
without constructing `World<S>`.

Keep the existing runner-local `Runtime` as the host control handle. Do not
put the world or VM inside it. Keep `App` as a platform adapter during the
migration. Its GPU callbacks must not become the headless game interface.

## 3. Contracts to settle before binding

### World changes and references

Change `loam-physics/src/world.rs` and `body.rs` first. Add world-owned
operations for teleporting a body, replacing collision geometry and mass
properties, changing velocity, applying impulses, and removing a body.
Validate before changing state. Clear affected persistent contacts when their
anchors or cached impulses become invalid. Wake bodies where the operation
requires it.

Changing mass must keep mass, inverse mass, and inertia consistent. Replacing
a collider must not leave inertia from the previous shape. Preserve geometry,
mass properties, and contacts when validation fails. Prepare expensive
geometry before the tick that commits it.

Review the visibility of `World::bodies`, `World::manifolds`, and mutable body
access. Keep low-level Rust access only where an algorithm needs it. The
game-facing operations must not expose arena removal or mutable manifolds.
Use `World::despawn_body` so removal clears contacts.

`BodyId` is local to an arena. A reset can recreate its slot and generation.
Add an opaque reference containing world ownership, world lifetime, and
`BodyId`. Reject foreign-world and expired references through checked lookup.
Two sessions must not accept each other's references even when both are at
their first reset epoch. Do not use a process-global mutable ID registry.

Keep authored object keys separate from live references. Save files and
script source use authored keys. A runtime handle cannot be serialized and
restored as authority over a later world. Reset and successful scene reload
invalidate old body, grab, and scene-owned resource references.

### Shape and pose

Give each object one canonical shape description, scale, and authoritative
pose. Derive picking and display geometry from that description. Record an
intentional collision approximation separately when it differs from the
visual shape. Do not infer Toybox geometry from `BODY_SIZE` and `Polytope4`
after a script can change its collider.

Expose local geometry plus a pose in the first bindings. Do not bind the
shared `Shape` enum wholesale. Sphere centers currently mean different things
in field evaluation and rigid-body placement. Clarify those existing Rust
contracts and bind only constructors with explicit local-shape semantics.

Flat body state currently has `position` and an isometry-valued `orientation`.
The isometry can contain a second translation. Settle that invariant in the
body API. The first binding accepts a point and a validated rotation; it does
not expose the raw isometry fields. Retain the generic `Space` and
`PhysicsSpace` contracts when fixing the representation.

Replace `PlaygroundPhysics::sync`'s repeated baking of authored rotation into
hull vertices with canonical geometry and explicit pose composition. Define
authored rotation edits as pose edits. Do not turn a UI edit into physical
angular velocity. Raster sections, SDF inputs, wireframes, picking, and labels
must read the same composed pose.

### Commands, queries, and ticks

Use one ordered simulation queue for pointer actions, shortcuts, UI, console,
and scripts. Replace Toybox's separate pointer queue and `pending_throws`.
Retain source sequence, target tick, and pointer sample duration. Coalescing
drag samples must preserve elapsed input time and release ordering.

Use this initial lifecycle:

1. The host collects events and assigns their order. Host controls and
   presentation controls still run while simulation is paused.
2. The session delivers input and prior command results to `on_events` as an
   ordered batch. Script changes to simulation append typed commands.
3. Before a fixed step, `fixed_update` reads the last committed tick state and
   appends commands after already queued commands for that tick.
4. Apply that tick's commands in order. Integrate physics once with the
   configured fixed step and its explicit substep policy.
5. Refresh retained queries and presentation state. Render from that state.

Queries do not observe pending commands. State this in the scripting API.
Each command is atomic; a failed command does not roll back earlier successful
commands in the batch. Return a typed result with its request ID. Do not add a
whole-world transaction mechanism to implement this rule.

Spawn returns a request token. Deliver the live object/body reference in its
success result before the next script callback. A request token is not a body
handle. During startup, commit setup commands before the first simulation
step, then deliver their results. This lets scripts create an arena without
fabricating arena indices or reserving half-constructed bodies.

Keep presentation changes on a separate documented commit path. Camera,
section display, windows, and pause controls remain usable with no fixed
steps. They cannot bypass simulation commands. Reset and manual step are
explicit simulation controls serviced at a boundary while paused.

Use seeded session randomness only where a game requests it. Simulation
callbacks receive tick time and fixed `dt`. Replay records ordered commands
and the sampled input needed by game policy. Do not promise bitwise parity
across native and browser targets. Parallel kernels remain allowed; a replay
mode must define when their results join the command stream.
Review built-in time and random functions when selecting Rhai packages.
A session that requests reproducibility must control those sources too.

## 4. Rhai adapter

Start `loam-rhai` with `session`, `source`, `diagnostic`, and explicit binding
modules for the operations used by Toybox. The crate owns no mandatory physics
world. Register concrete capabilities when constructing a game's session.
Keep errors typed in engine owners; convert them to script diagnostics at
the adapter boundary.

Pin one tested Rhai release in the workspace and lockfile. Use the same
language and number configuration on native and browser. Start with normal
Rhai integer and floating-point types; validate range and finiteness when
converting to Loam's `f32` values. Do not enable a different integer width
just for WASM. Add the target-specific WASM support required by the selected
release. Rhai's feature flags can change language behavior, including numeric
types and safety checks. [Rhai feature reference](https://rhai.rs/book/start/features.html).

Keep a VM, compiled program, retained scope/state, source revision, and
diagnostics per script session. Compile and initialize once. Use explicit
callback invocation options so top-level initialization does not repeat on
every tick. Test imported functions and persistent state with the chosen
invocation path. Rhai's default `call_fn` evaluates the AST before invoking
the function. [Rhai callback options](https://rhai.rs/book/engine/call-fn.html).

The first callback set is `init`, `on_events`, and `fixed_update`. Script UI
descriptions are built during initialization or structural changes. Native
presentation runs every frame. Add another callback only when a migrated
feature needs a different timing contract.

Bindings expose copied query values, resource references, and a command sink.
Keep `World` exclusively owned outside the VM. A small session-local shared
bridge may hold query storage and queued commands; it must not hold a mutable
world or GPU device. Query results identify their committed tick. Scripts
cannot retain references into reused query storage. A saved query result keeps
its value; a later query reads the current committed state. Reuse buffers
without cloning the world or topology into a new map each tick.

Keep the VM on its owning host thread initially. Native geometry and
simulation work can still run in parallel. Do not wrap the engine in a global
mutex or force graphics resources to implement `Send`. Revisit Rhai's `sync`
feature only for a measured need to move or share a script session.

Register point, tangent, bivector, and rotation operations explicitly. The
first physics module supports Euclidean R⁴. Name that capability explicitly
in its module and signatures. Later H³ or S³
bindings must carry their coordinate and space identity. Do not flatten all
points into interchangeable arrays. Bind physics only for supported
`PhysicsSpace` implementations.

Give scripts batch operations for geometry and presentation. Keep integration,
collision, section generation, projection, topology traversal, interpolation,
and mesh upload in Rust. A script should request an operation on a shape or
collection, not run a callback for each contact, vertex, or ray sample.

### Failure and reload behavior

Set finite operation, recursion, module, and collection limits. Select their
values from the shipped scripts and measured headroom. Keep `unchecked`
disabled. Bound expensive native requests separately: Rhai counts a native
call as one operation and cannot use that count to limit its internal work.
[Rhai operation limits](https://rhai.rs/book/safety/max-operations.html).

On a callback error, discard commands appended by that callback, pause the
affected game, and report module, location, callback, and message. Keep host
input, diagnostics, reset, and reload usable. Script-local state may already
be partly changed; do not retry it as though the call rolled back. The first
recovery path is reset or reload, not deep-cloning the VM on each call.

Freeze earlier unapplied simulation commands and undelivered results while
the script is faulted. Ordinary resume cannot restart it. Successful reset
or reload discards that pending work and starts a new world lifetime. Do not
replay it against the replacement script. Diagnostics may retain its request
metadata.

Load and compile a complete candidate source bundle before replacement.
Initialize it in a candidate session with isolated commands and resources.
Swap sessions only after setup succeeds. Keep the current session on failure.
Successful reload restarts the active scripted scene and invalidates its old
handles. Preserve host window and GPU lifetime. State-preserving hot reload
and arbitrary VM-state serialization are later features.

### Allocation policy

Keep existing allocation-free warm paths for rendering and geometry assembly.
Retain command buffers, query storage, UI trees, text caches, and mesh scratch.
Measure VM allocation separately from native simulation and presentation.
Rhai execution is not an established zero-allocation path.

Before enabling recurring script callbacks, measure the actual scripts and
resolve the repository's allocation rule. If it covers every allocation made
inside a callback, the implementation must satisfy that rule or the project
owner must explicitly approve a narrower policy for script execution. Moving
an allocating callback out of `update` does not resolve the cost. This plan
does not change `AGENTS.md` or assume that exception is approved.

## 5. Native and browser hosts

Use one startup simulation configuration for fixed rate, catch-up policy,
initial scene, and seed. Separate it from window settings. Pass it to both
runners. The browser worker currently drops `RunConfig`, fixes the clock at
60 Hz, and initializes MSAA at 1. Pass the chosen configuration to both hosts.

Change `loam-app/src/lib.rs`, `wasm/launch.rs`, `wasm/main_launcher.rs`, and
`wasm/worker.rs`. Transfer serializable settings with worker initialization.
Create window/device objects on their owning host. Keep GPU options separate
from simulation settings, and report unsupported requests explicitly.

Make native resume idempotent while an app is active or initializing. A repeat
host event must not construct another game. Route browser startup failures
back to the page so missing assets, rejected GPU setup, and script errors end
the loader state. Handle recoverable worker surface errors within configured
budgets. Keep script faults separate from surface and fatal device failures.
Full device-loss reconstruction can follow later.

Add `crates/polytope_playground/assets/game/game.ron` as a small source
catalogue. It names the entry module and the modules needed at startup. This
tracked directory avoids the repository's ignored `scripts/` paths. Both
hosts supply the same module IDs and texts, keyed relative to the game root.
Native development reads it from a configured project root and uses the
existing `FileWatcher`. Browser startup fetches the declared files before
starting the scene. Resolve asset URLs against the page's asset base, not the
worker's Blob URL. Package the files with the native release and copy them
into the Trunk distribution.

Pass an owned, backend-neutral startup payload through host setup and worker
initialization. The Playground frontend converts its module texts into
`loam-rhai::SourceBundle` for the CPU constructor. This keeps Rhai out of
`loam-app`. Keep binary assets separate from the module-text map.

Use one in-memory module resolver on both targets. Normalize module IDs and
validate the catalogue and startup imports before replacing a session.
Reject out-of-project resolution. Later or computed imports can still fail;
apply the callback failure contract to those errors. Keep
filesystem paths, URLs, asynchronous fetch, and file watching outside Rhai
execution. Rhai supports WASM, but its filesystem script loading is not
available there. [Rhai WASM support](https://rhai.rs/book/start/builds/wasm.html).

Define native and browser loaders as concrete host code. Do not introduce a
loader trait or general asset pipeline for two small source-loading paths.
The resolver receives ready data; imports must not block waiting for browser
fetch. Module resolvers return synchronously.
[Rhai module resolver API](https://docs.rs/rhai/latest/rhai/module_resolvers/trait.ModuleResolver.html).

Make binding availability explicit at startup. Portable Playground scripts
use the common capability set. Native-only services register additional
operations. An unsupported required capability produces a load error; it
must not silently succeed through a no-op. A web-capable script must have a
native host with the same common operations and input semantics.

Keep the existing WebGPU, OffscreenCanvas, and module-worker path as the first
browser implementation. Browser/device coverage and a possible WebGL fallback
remain release decisions. Do not cap native graphics features to settle them.
Native files, worker setup, visibility changes, focus, and capture stay in
their host adapters. WASM support does not imply native filesystem or thread
APIs are available in a browser.
[Rust browser target limitations](https://doc.rust-lang.org/rustc/platform-support/wasm32-unknown-unknown.html).

With the new Playground library, update the browser build to select
`--no-default-features --features frontend --bin polytope_playground`.
Set `data-cargo-features="frontend"` and `data-bin="polytope_playground"` in
its Trunk Rust link. Retain `data-cargo-no-default-features`.
[Trunk target and feature selection](https://trunk-rs.github.io/trunk/guide/assets/index.html).

## 6. First scripted deliverable: Toybox

Use the existing Rust Toybox as the first caller of the corrected operations.
Then register bindings against that same seam.

1. Bind queries and commands for existing named toys. Prove callback order,
   pause behavior, errors, and native/browser execution before script spawning.
2. Add prepared geometry, body creation, removal, and asynchronous command
   results. Move the five initial toys, arena layout, material choices, and
   spawn settings into `assets/game/toybox.rhai`.
3. Keep pointer unprojection, picking math, and collision kernels native.
   Move pick preference, grab/release decisions, pointer history, throw gain,
   speed caps, arena rules, sleep tuning, and respawn policy into scripts.
   Bind reusable mechanics rather than the whole `Toybox` struct.
4. Build one script-defined panel for pause, respawn, and throw controls.
   Native widgets retain focus and capture state. Events use the common
   command path.
5. Run the same scripted interaction sequence headlessly, in the native
   frontend, and in the browser worker. Remove the replaced Rust policy once
   the scripted path covers its boundaries.

Extract only the shared native work needed here:

| Existing code | Destination and change |
|---|---|
| Toybox and Hero polytope body construction | `loam-physics::euclidean_r4`: validated size, pose, mass, collider, and matching inertia |
| `PlaygroundPhysics::sync` | Coherent world operations; no independent collider and inertia assignments |
| `Toybox::throw` | World velocity/impulse operations with validation and wake behavior; script retains throw policy |
| `toybox.rs::face_down_pose` | `loam-shape`: facet-normal alignment with caller-supplied target direction |
| Toybox cap generation and `BodyPose::body_local` | Posed section assembly over canonical geometry and caller-owned scratch |
| Toybox contact diagnostic traversal | Read-only body/contact queries plus reusable native overlay geometry |
| Toybox drag mechanics | Keep the small implementation at the game seam until another caller shares it; extract the repeated mathematical operation when justified |

Do not move `GrabTrail`, arena clamps, or Toybox sleep thresholds wholesale
into an engine crate. If scripted pointer history proves expensive, move a
parameterized sampler into Rust and keep its policy in the script.

## 7. Presentation and the rest of Playground

Split `state.rs::Demo` into game state and frontend resources. Extract
operations from demo modules under engine contracts. Do not move whole demo
files into `loam-render` with their current catalogue and layout dependencies.

Provide retained renderable instances with geometry, pose, material, and
visibility. Views select camera, projection, section, viewport, and instances.
Script commands update those records. Native code builds cached meshes,
manages GPU nodes, and records passes into the runner's encoder.

The presentation service must cover the current features as they migrate:

- Physical sections and projected caps with separate fill and perimeter styles.
- Wireframes, pole clipping, slab culling, vertex markers, and cell centers.
- SDF and raster shape rendering, background, depth, and translucent layers.
- Gizmos, world anchors, text, arena guides, and contact diagnostics.
- Multi-view samples with their own pose, slice, camera, and viewport data.

Keep graphics implementation extensible from Rust. The scripting API should
select registered shaders, materials, and passes through resources. It should
not receive `wgpu` objects. Do not replace existing custom-pass access with
a fixed list of Playground render modes.

Start script UI with a retained control description and typed events.
Descriptions own labels, layout, bindings, and action names. The native
adapter owns widget IDs, focus, drag state, text editing, measurement, and
capture. Derive widget IDs from stable control keys. Reordering a list must
not retarget focus or queued actions. Add widgets when a migrated panel needs
them. Keep description values independent of `egui` types so custom rendering
can be added later.

Keep script-facing presentation and UI values CPU-only. Place initial
game-specific descriptions in the Playground library. Its frontend interprets
those descriptions using `loam-egui` widgets. Engine crates never import
Playground types. Put neutral geometry and physics values in their engine
owners. Register frontend operations in the Playground adapter. Promote a
shared description module when the migrated callers establish its contract.

Complete UI support includes shape-card and Composer-term reordering, plane
transfer between terms, slider popups, context menus, formula editing and
errors, focus retention, movable windows, tabs, and scrolling. It also includes
the slice ruler, world callouts, readouts, and bivector matrix.

The full conversion has the following ownership map:

| Current module or feature | Script ownership | Native ownership or destination |
|---|---|---|
| `shell.rs`, scene entries | Scene labels, order, entry scripts, defaults | `loam-app` startup and scene lifecycle |
| `catalog.rs`, `shapes.rs` | Catalogue, aliases, row layout, add/remove/reorder, subject selection | Shape factories, resource ownership, validated edits |
| `active.rs`, `spins.rs` | Active planes, authored angles, clock, mode policy | `loam-math` rotor operations |
| `composer.rs` | Formula parser, term editing, displayed formula, plane transfer between terms | Bivector arithmetic and rotor exp/log |
| `director.rs`, timeline setup | Track binding, slot names, precedence, playback choices | Existing `loam-time::Director` sampling and easing |
| `physics.rs`, `state.rs` | Row damping, reset and inactive layout policy | World, authoritative poses, read queries, resource lifetime |
| `projections.rs`, `sections.rs` | Mode selection, defaults, annotations, clipping settings | Existing math/shape projection and section operations |
| `render.rs`, `wireframe_geom.rs`, `color.rs` | Enabled layers, styles, color choices | `loam-render` caches, mesh builders, buffers, pass recording |
| `filmstrip.rs` | Axes, counts, samples, labels, subject | View subdivision and batched presentation without mutating bodies |
| `hypergimbal.rs` | Visibility, selection, action mapping | Existing gizmo picking, drag math, and rendering |
| `ui.rs` and scene panels | Menus, layout, labels, values, handlers | `loam-egui` widgets and capture |
| `console.rs`, `verbs.rs` | Playground verbs, help, completions, handlers | `loam-console` parsing, history, and dispatch |
| `hud.rs`, callouts, readouts | Content, format, anchors, visibility | `loam-text` caches, projection, scaling, and overlay recording |

Use modules such as `catalog`, `rotate`, `composer`, `toybox`, and scene UI
modules under `assets/game`. Keep timeline RON assets and the native sampler.
Keep the existing frame-indexed console automation in `loam-app/src/script.rs`; it is
a separate automation facility. Do not delete it as a duplicate Rhai runner.

Complete command coverage as each feature moves. Active-plane controls,
Composer, shape editing, menu resets, and shortcuts still mutate `Demo`
directly. They need typed actions even if the temporary panel remains Rust.

Replace the static scene metadata assumption in `SceneRegistry::SCENES` and
`SceneEntry` with owned entries for script scenes. Adapt existing Rust scenes
without retaining a second Playground catalogue. Preserve cached inactive
scene state when switching. Define successful source reload as replacing the
affected session and its scene-owned registrations.

Change console completion storage in `loam-console::Command` and `FnCommand`
to accept owned script metadata. Release old handlers, help, and completion
entries on replacement. Do not leak strings to satisfy their current static
lifetimes. Scene replacement must also release callbacks that retain the old
session's command sink.

Use these semantics as migration baselines. Review defects as explicit
behavior changes with independent expectations:

- Active rotation uses an ordered product in `Plane4::ALL` order. Composer
  uses the exponential of a bivector sum and a distinct scrub operation.
- Directed slots retain timeline ownership during pause and after the last
  sample. The current Playground adapter rejects position tracks.
- Physical cross-sections use drop-w. Projected caps use the selected
  projection. Preserve Playground's xyz-after-projection preview placement
  as an explicit presentation offset. Keep that layout convention separate
  from world pose; translation and perspective projection do not commute.
  Body w still enters the section frame before projection.
- Smooth catalogue shapes use the SDF path. Keep the existing 120-cell and
  600-cell restrictions explicit. Schlegel exposure is a separate UI choice.
- Filmstrip currently uses the SDF strip path and samples authored poses.
  It does not provide the normal raster, wireframe, point, gizmo, and HUD
  composition. Preserve that scope during migration; extend it separately.
- Background clears first. Later passes load. Per-view uploads must retain
  their contents until all draws that reference them have executed.
- Capture/focus changes release a grab. Pointer input stages commands;
  physics applies their effects at its defined boundary.

After conversion, delete Playground's `Demo`, `RotateScene`, `ToyboxScene`,
`PlaygroundPhysics`, `SlotSpins`, and `Playback` policy implementations.
Delete duplicated pending queues, native catalogue and verb registration,
native UI definitions, and replaced constants. Remove obsolete demo modules
after their reusable operations have owners and their policy lives in scripts.
Retain only concrete session composition, frontend registration, assets, and
startup glue. Do not keep a permanent feature flag for the old game.

## 8. Verification and review

Use the repository's check ownership. CI owns Rust formatting checks, clippy,
builds, tests, GPU probes, rustdoc, and WASM builds. Local work uses source
review, the existing style hook, and diff checks. Reproduce a CI failure
locally only under the repository's stated exceptions.

Extend `.github/workflows/ci.yml` for the new boundaries:

- Build and test the Playground library with `--no-default-features` in an
  isolated feature selection. Check that its resolved dependency closure has
  no app, window, UI, or GPU dependency. Do the same for `loam-rhai`.
- Keep existing native workspace and GPU checks. Update the WASM build to
  select the frontend binary explicitly.
- Add one bounded native frontend run on a CI runner with a display and GPU
  adapter. Replay the scripted Toybox sequence, check structured results, and
  exit through the host. Existing unit tests do not exercise the window runner.
- Add browser execution of the built distribution. Start a supported WebGPU
  browser, load the worker, run the shipped Toybox script, and observe the
  committed command results and a worker acknowledgement after frame
  submission/presentation. Check browser and GPU errors. Use structured
  completion instead of screenshot judgment. A Cargo WASM build does not
  establish browser execution.
- Exercise missing modules, script failure, focus/capture release, and
  non-default fixed rate through the actual browser path. Treat an absent
  required browser capability as an infrastructure failure or an explicit
  unsupported configuration, not a successful test.
- Report LLVM coverage for the changed production modules alongside the
  integration results. Reuse instrumented executions rather than repeating
  the suite to generate another report. Distinguish native CPU, GPU, and
  browser evidence. Do not use a workspace percentage as proof of scripting
  correctness or add tests just to raise it.

Repurpose existing tests around the following defects. Move numerical tests
with their engine operations. Application policy tests should load the shipped
scripts through the real adapter. Delete the duplicate Rust policy fixture
after the script test covers its boundary.

| Boundary | Defect to catch |
|---|---|
| World lifetime | A stale or foreign handle changes a recycled body or replacement world |
| Coherent mutation | Rejected pose, collider, or mass edits partly change the live body; accepted edits leave stale contacts |
| Command routing | Pointer, console, and script actions reorder, repeat, or disappear around pause and catch-up |
| Spawn results | A pending request is mistaken for a live body; results cross a reset |
| Geometry and pose | A scaled, rotated, nonzero-w object disagrees between collision, section, and picking |
| Physics | Off-center impulse has the wrong torque; sleeping bodies miss contact wake; substeps omit a required boundary |
| Script lifecycle | Initialization repeats, scope grows each tick, imports disappear, or two sessions share state |
| Query lifetime | Reusing native storage changes a result retained by a script |
| Script failure | A failed callback leaks its commands; failed reload replaces working code; a faulted state runs again |
| Rotation and timelines | Rotor order changes; manual controls override directed slots; pause or final samples lose ownership |
| UI and input | Text focus triggers game shortcuts; drag capture sticks; scene reset retains old targets |
| Toybox policy | Stale release samples throw a body; wall clamps alter release intent; picking loses its visible-cap order or hidden-body fallback; slice reach or cap fade changes |
| Authoring and HUD | Active checkbox edits change the displayed angle; labels use logical pixels where physical pixels are required |
| Presentation | Cap projection changes physical sections; pole/slab boundaries break; indices or upload ranges alias across views |
| Storage reuse | Warm native section, wireframe, diagnostic, and presentation paths allocate or grow storage without bound |
| Host parity | Worker ignores fixed rate or command inputs; render cadence changes the result of a fixed command tape |
| Space boundary | Unsupported geometry is accepted as R⁴; points from different registered spaces are silently mixed |

Keep the existing multi-space mathematical conformance and substantive
collision tests. Use analytic expectations or an independent formulation
for invariant checks. Do not replace them with script round-trips that call
the same implementation twice. Keep data-against-itself, setter/getter, and
duplicate constructor tests deleted.

Measure warm frame cost, tick cost, script callback cost, allocations, source
load/reload cost, and browser bundle size for the migrated workload. Use the
current Rust behavior as the baseline for each replaced feature. Set budgets
from that evidence and the intended workload; do not invent frame or body
counts in advance. Include script/native call counts to expose an API that
requires fine-grained VM crossings.

Review the work in bounded parallel areas: physics contracts; app/browser
hosts; script and Playground policy; presentation and UI. Give each area
exclusive write ownership while it is active. Review shared API signatures
before consumers migrate. The integrating reviewer checks the dependency
graph, callback order, deletion map, and cross-crate tests. Independent review
does not substitute for that integration review.

## 9. Work after the first scripting deliverable

| Goal | Follow-up implementation | Timing |
|---|---|---|
| Native graphics freedom | Accept requested GPU features and limits when a native consumer needs them; retain existing custom encoder/pass access | Can follow Toybox without waiting for a device/surface split |
| Offscreen rendering | Split device/queue ownership from optional surfaces for a concrete offscreen consumer | Required before offscreen rendering or detached GPU compute |
| More geometry in scripts | Add one non-R⁴ geometric app or example using concrete `Space` bindings without requiring rigid-body physics | After the first adapter; use it to review accidental R⁴ assumptions |
| Regions and bridges | Define region identity, chart transitions, dimensional embeddings, and field transfer for one concrete pair of spaces | Separate research increment after the scripting boundary is proven |
| SDF portability | Distinguish exact distance, conservative bounds, and general implicit fields; make transform and stepping accuracy explicit | Before cross-space field composition, not before script-controlled Toybox |
| Editing | Add authored IDs, data serialization, and reversible command groups where editor operations require them | Build on the command boundary; do not serialize live VM internals |
| Audio | Add a separately owned audio service and resource/command bindings; implement native and browser lifecycles | Independent feature after its playback requirements are chosen |
| Custom UI and advanced rendering | Extend material/pass resources and add alternative UI rendering while retaining input and command contracts | Driven by a concrete interface or graphics feature |
| Research simulation | Reuse headless construction, reset, step, query, and batch operations; allow solvers outside `PhysicsSpace` | Available in stages; do not force fluids or optimization through rigid bodies |
| Parallelism and replay | Add explicit job completion boundaries and selected reproducibility modes around measured workloads | Preserve optimization freedom; avoid a workspace-wide deterministic scheduler |

Do not implement runtime geometry interchange by erasing `Space` into one
Vec4-based world interface. A chart transition, a metric change, and a
dimensional embedding have different contracts. Keep the first R⁴ adapter
small enough that those later contracts can be added independently.

## 10. Completion and unresolved decisions

The migration is complete when both Playground scenes and their controls run
from shipped scripts on native and browser, the CPU session runs without
graphics dependencies, and the old Rust game policy is removed. Geometry and
rendering remain compiled. Script errors leave the host usable. Reload replaces
the scene only after successful initialization. Existing mathematical and
presentation boundaries retain appropriate coverage.

Two decisions remain outside this plan's authority:

- The supported browser/device matrix and whether WebGL fallback is required.
  The first implementation uses the repository's existing WebGPU path.
- Whether measured allocations inside recurring Rhai execution may have a
  separate budget from allocation-free engine frame paths. Resolve this
  against the repository rule before accepting that execution path.

Reset-on-reload, one optional Rhai package, an initial R⁴ physics binding, and
incremental retained UI are the proposed implementation defaults. They do not
limit native engine capabilities or make R⁴ the engine's universal space.
