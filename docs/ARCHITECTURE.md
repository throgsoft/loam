# Architecture

## Crate boundaries

The manifests define the dependency graph. These groups describe
responsibility, not a universal capability shared by every crate.

| Crates | Responsibility |
|---|---|
| `loam-math` | Spaces, isometries, rotors, projections, and geometry WGSL |
| `loam-shape` | Shared shapes, polytope topology, cross-sections, and isovolumes |
| `loam-scene` | CSG scenes, CPU evaluation, loading, edits, and WGSL emission |
| `loam-physics` | Body handles, integration, collision, contacts, and constraint solving |
| `loam-time` | Fixed timestep, traces, replay data, and animation timelines |
| `loam-input`, `loam-camera` | Input state, cameras, and controllers |
| `loam-render` | GPU rendering, shader assembly, validation, and reload |
| `loam-text` | Text overlays and glyph geometry |
| `loam-console`, `loam-egui` | Console model and UI integration |
| `loam-app` | App callbacks, scene shell, command routing, platform runners, file watching, and capture |
| `loam` | Convenience exports |
| `polytope_playground`, `tesseract_demo`, `hero` | Demo applications |

Math, shape, scene, and physics have no GPU dependency. WGSL generation in
the CPU crates is still a shader ABI dependency. Keep changes to that ABI
explicit. The renderer does not own simulation state. Constant gravity is a value
on `World<S>`; games apply other forces through body impulses.

The app watches files and routes changed paths to `loam-render::shader`.
The shader cache retains its last valid module when a reload fails. It
does not depend on filesystem notification types.

`Space` is the high-level geometry abstraction. It defines metric operations. `IsometryGroup`, `WgslSpace`,
`RasterizableSpace` and `PhysicsSpace` express separate
requirements. Their implementations differ. Rendering support does not
imply rigid-body support.

## Geometry and physics

Math includes Euclidean R² through R⁴, hyperbolic H³, two S³
representations, flat tori, lens spaces, and a blended metric. The two S³
representations use different coordinates and cannot share arbitrary chart
arithmetic. See each type's conventions before combining operations.

Rigid-body implementations cover flat R², R³, and R⁴. Narrowphase dispatch
supports specific collider pairs. New geometry and shape implementations
must define the collision cases they support.

`World<S>` owns a generational body arena and persistent contacts. Body
handles survive storage compaction and reject reuse after despawn. A game
may own several worlds. The physics world is not a general entity store.

Physics processes contacts in a defined order. That order affects solver
results. It is an implementation property that games can use for replay,
not a workspace-wide ban on concurrency or optimization.

## App and command boundaries

`App` separates fixed ticks, frame updates, UI, and recording. The native
and browser runners own frame pacing and GPU submission. The scene shell
selects an active demo and forwards its ticks, commands, and presentation
callbacks. Each shell owns its scene control and cached scenes. The active
scene is always present. Scene construction and reload use the runner's
shader database. Failed construction keeps the current scene.

The Toybox applies queued interaction commands in its tick callback.
Dragging still uses sampled pointer input and its frame duration. Exact
replay requires recording those inputs. Rotate, Toybox, the tesseract, and
the wordmark advance simulation or animation in fixed ticks. Cameras and
UI use frame time. Timelines sample authored frames from elapsed tick time.

`loam-console` owns `CommandLine` and instance-owned `CommandQueue` values.
These types need no window or GPU. Each runner owns a `Runtime` handle for
commands, output, cursor requests, pacing, capture, and exit. Setup and frame
callbacks receive that handle. Registered callbacks retain a clone for the
same runner. The runner dispatches commands before each frame's tick batch,
including frames with no ticks. This keeps presentation controls
available when simulation is stopped. Games queue simulation effects for
their tick callbacks. Runtime controls run on the event-loop thread; this
does not impose a threading policy on simulation or geometry work.
Logger and allocator statistics remain process-wide. Frame tracing is
thread-local. Browser callbacks connect one worker to its host page. The text command vocabulary does not yet define a
typed simulation API.

`script.rs` plays console commands at frame indices. It supports automated
demo and capture runs. Rhai gameplay integration still needs bindings,
execution limits, state queries, and defined tick ordering. Replay tape
serialization in `loam-time` does not supply that integration.

The intended gameplay boundary is a read interface plus commands that carry
stable handles and validated values. The game applies commands at its
simulation boundary. Rendering reads the resulting state. Rhai, UI, and
native game code should use the same operations.

## Demo extraction

Keep scene menus, catalog choices, color schemes, capture choreography,
and arena rules in the demos. Engine APIs should accept the data that those
choices produce.

Ray intersections for picking live in `loam-camera`. Polytope cell-frame
and eye setup lives in `loam-shape::projection`. The playground retains its
cell-selection policy and cached projection parameters. Projected edge
tessellation and pole clipping also live in `loam-shape`; callers supply
sampling and clipping limits. `loam-text` appends glyph section geometry
into retained meshes. `CameraRig` shares orbit and free-camera behavior.
Renderer upload and buffer lifetime belong in `loam-render`. Simulation
commands belong at the world or runtime boundary.

The shared shape enum is not yet a universal geometry representation.
Sphere centers have different meaning in scene evaluation and rigid-body
placement. Bindings must preserve those conventions until pose and shape
ownership have one explicit contract.

## App direction

See the [scripted Playground plan](SCRIPTING_PLAN.md) for the proposed
implementation sequence.

The next game exercises 4D interaction and physics. `Space` remains the
shared abstraction. A game selects its space; algorithms request the
capabilities they use. A physics world requires `PhysicsSpace`. A renderer
requires its rendering capabilities. Neither requirement belongs on every
game or every space.

Keep the host responsible for platform events, clocks, window and device
lifetime, and GPU submission. Keep simulation state in a game-owned value.
The current `App` callbacks can connect those owners. A new scheduler or
entity framework needs a concrete use case before it becomes a dependency.

The console, capture panel, traces, and scene browser should be optional
host tools. They use the same commands as game code. Editors come later.
The demo catalog and restart confirmation belong in `SceneShell`.

Before adding Rhai, define typed operations for the first interaction loop:
spawn a supported body, remove it, query it, pick it, grab it, release it,
and apply forces or impulses. Use stable handles and validate inputs at the
owning engine API. Keep arena rules and throw policy in the game. Shared
picking, contact visualization, and sleep mechanics belong in engine crates.

Rhai receives queries and a command sink. A script cannot retain a mutable
world reference. The game applies queued commands at a chosen simulation
boundary. Bindings register the concrete point, tangent, and rotation types
for that game's `Space`. This keeps the first R4 bindings from defining the
whole engine's geometry API.

Games that require replay must control command order, time, randomness,
and parallel reductions. The host should expose those choices. Geometry,
rendering, and games without that requirement can use parallel work and
approximations under their own accuracy contracts.

## Current limits

Rhai is not integrated. The console queue is a reusable input boundary, but
text commands do not replace typed game operations or stable entity queries.
The physics world remains directly mutable by Rust callers. A game runtime
must own that access before it exposes scripting.

Blended geometry uses numerical shooting on the CPU. Its WGSL logarithm and
distance use cheaper chart approximations. Those paths do not promise equal
results. Spherical charts also have finite domains; a rendered space is not
proof that every movement or physics operation handles its global topology.

Convex collision still allocates EPA storage per contact. GJK has absolute
tolerances, and EPA can return its best estimate at the iteration limit.
Rigid-body inertia uses scalar approximations. The friction solver stores a
tangent vector and applies changes when the Coulomb bound shrinks. The
inertia approximation and collision tolerances still need scene-specific
validation.

The extended polytope shader uses topology facets. Its result is exact
inside the polytope and a conservative distance bound outside. Browser SDF
restrictions remain until the new shader path is verified there.

## Verification

Workspace lints deny implicit unsafe operations and undocumented unsafe
blocks. CI does not currently collect line or branch coverage.

Tests retain geometric identities, singular and boundary cases, and contracts
between crates. The same identity can catch a different defect in each
`Space` implementation. Analytic cases and rendered GPU results provide
independent checks. Round trips alone do not establish correctness when
both operations can share an error. Test count and line coverage do not
measure whether those checks are sufficient.

CI owns formatting, clippy, workspace tests, GPU probes, rustdoc, and the
browser build. Replay tests run with the workspace tests. See
[ci.yml](../.github/workflows/ci.yml). The documentation workflow publishes
rustdoc from `main`.

Local review covers the changed code, documentation accuracy, and repository
hooks. Do not duplicate CI checks locally unless the user requests it or an
inspected CI failure requires a narrow reproduction. Report local findings
and CI results separately.
