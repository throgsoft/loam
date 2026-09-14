# Engine architecture review, 2026-09-14

Branch `ecs`, HEAD `0650128d0d7e5d246a1a8d96c186b8bd36ec4b46`, clean tree with no
untracked source. The candidate is 83 commits above `48f4609`, the base the
previous candidate was built on; the 2026-09-13 review graded `c355e22` plus a
dirty tree, so this is a new candidate rather than a re-grade. The target being
graded is the foundational engine architecture for the Polytope Playground and
the coming scripting layer. Four independent area reviewers covered the core
runtime, geometry and numerics, rendering and the hosts, and the applications and
facade; one referee challenged every finding against the source. All five read
this head. Per the owner's rule, a finding whose only content is that something
has not been observed on a physical host, browser, or device is an evidence gap
that lowers confidence and not the grade; no grade below was moved by one.

## Decision

Acceptable for the stated target, with explicitly accepted limitations. No
identified defect breaks a central current contract of that target. The two
defects a user can reach today, R1 and R2, have small local repairs and neither
corrupts state. The ownership, identity, and restore work is done and holds under
adversarial tests; physics and configuration admission is closed at the owner
boundary; the rendering extension path is proven on a real adapter across resize
and device replacement. Hold is not warranted: there is no demonstrated blocking
defect and no required release evidence that is unavailable rather than unrun.
R1 and R2 should land before the next demo push, both being cheap and visible.

Accepted limitations:

- No host or browser observation exists for any interactive behavior at this
  candidate. Behavior and presentation fidelity stays at C until Row B lands and
  host evidence is recorded.
- Performance budgets are not bridged to the engine's own numbers, and choosing
  the reference device, build, workload, and budget is a prerequisite, so Data
  layout and scaling cannot reach A on code changes alone.
- The blended space's GPU half is a first-order approximation measured at 1.9e-2
  with no declared accuracy tier. It has no shipped consumer.
- H3 is not reachable from any shipped application. R4, R12, R16, and R18 are
  latent and should land before the first H3 demo.
- The command boundary scripting must enter is one level deep and cannot return a
  value or carry a dynamic message. Deferred, not a defect.
- The browser routing layer is compiled by CI and never linted or executed. Its
  one uncovered decision has a real guard that holds and no test states it.

## Category grades

| Category | Grade | Confidence | Reason | Findings |
|---|---|---|---|---|
| Application ergonomics | B | High | Setup is six declared steps with most omissions caught at compile time and no application-side pose mirror, command queue, or epoch repair remains, but three obligations fail silently when missed. | R2, R1, R11 |
| End-to-end ownership | A | High | Every despawn, spawn rollback, and restore has one owner that fans out to application stores, domains, facilities, and bridges, restore rebases references only by exact scene, and assets sit outside the snapshot so restored ids always resolve. | none material |
| Abstraction value | B | High | Nine geometry capability traits, `Owner`, `stores!`, `Facility`, and `Records` each remove a real caller obligation, but `loam-time` carries a timeline subsystem with no consumer that alone justifies three of its seven dependencies. | R24, R11 |
| Geometry model | B | High | 172 conformance cases test properties rather than values and the H3 unit tests use independent expectations, but `chart_reach` admits the singular value every other H3 site clamps and `LOAM_MAX_ARC` is disabled outside S3. | R4, R12, R18, R16 |
| Domain and presentation boundaries | B | High | CPU and GPU agree to 1e-7 for E3, H3, S3, and Scene4 on a real adapter and raster and raymarch write the same projective depth, but the blended emitter measures 1.9e-2 with no declared accuracy tier and one Scene pass clears the shared color target under no engine rule. | R6, R7, R16 |
| Runtime correctness | B | High | Foreign and recycled handles cannot act and a failed phase blocks steps, publication, and snapshots until a restore succeeds, but one recovery press runs two full restores and a `SimConfig` value the runtime documents as supported panics the host at construction. | R1, R3 |
| Data layout and scaling | B | Medium | Dense stores, bounded rings, and reused scratch are right and five executed allocation oracles hold on warm paths, but the shipped default log capacity has a cache cliff both large bench cases size around and no bench run exists at this head. | R14, R5 |
| Cache correctness and value | B | High | A differential oracle proves the patch path matches a fresh publication across seven edit kinds and library immutability closes a whole invalidation class by construction, but three accessors give up more cache than they need to and no shipped build reads the counters. | R13, R8 |
| Rendering architecture | B | High | One shared reversed-Z depth buffer, a negotiated `FrameFormat`, pass-attributed errors preserving the wgpu validation cause, and device replacement re-attaching every pass are enforced and probed, but the pass contract validates depth writers only. | R7, R8 |
| Native and WASM separation | B | Medium | One crate owns all 59 wasm configuration sites and the job that keeps that honest holds, but the browser routing table is compiled and never linted or executed and the browser asks for no timestamp query. | R15 |
| Numerical and physics quality | B | Medium | Configuration and timestep admission are closed, the field error bound is a real interval forward bound checked against an f64 oracle, and the contact gate bites 50x below solver slop while still producing useful contacts, but varying-blend transport rests on one pinned f32 endpoint. | R17, R6, R18 |
| Rust implementation quality | A | High | No unsafe in `loam-runtime`, two panics in production code and both are documented exhaustion stops, every other failure is a `Result`, typed borrows cannot reach another domain or session, and clippy is clean at deny-warnings on all targets. | R22 |
| Behavior and presentation fidelity | C | Medium | The cursor policy, touch cancellation, consumed-release handling, and queue overflow are careful and tested off target, but the flagship presents a menu label that overstates what it does, two console commands that report success without acting, and a recovery key that changes hero's reseed result under fault. | R10, R1 |
| Verification and diagnosis | B | High | Test bodies are real oracles that name the defect they catch and errors name phase, system, and pass with the cause intact, but three error families print a bare or wrong noun, GPU errors never reach the console, and the allocation primitive six oracles depend on has no installed-allocator guard. | R5, R9, R8, R2 |
| Extension cost | B | High | A custom pass costs four trait methods and is attached, re-attached, timed, and error-attributed for free and a homogeneous space is one macro line, but the command boundary scripting must enter cannot return a value or carry a dynamic message. | R23, R6 |

Changes against the 2026-09-13 review:

- Abstraction value, A to B. `loam-time`'s timeline half has no consumer and
  carries three dependencies for it (R24), and the camera registration has no
  counterpart the flagship can use (R11). Neither was in scope then.
- Geometry model, A to B. Three H3 items the earlier review did not reach: the
  singular `chart_reach` (R4), the disabled arc cap (R12), and a fixture covering
  14 percent of the declared envelope for three properties (R18).
- Runtime correctness, A to B. The double restore on one recovery press (R1) and
  the unvalidated `SimConfig` (R3). Both are new candidate behavior.
- Data layout and scaling, C to B. Bounded storage and allocation behavior are now
  covered by five executed oracles on warm paths. What remains is a budget the
  engine cannot choose for itself, not a design weakness.

End-to-end ownership and Rust implementation quality both hold at A.

## Accepted findings

**R1. One recovery press runs two session restores. Defect.**
`crates/loam-app/src/session/frame.rs:279`, `crates/loam-app/src/session/commands.rs:208`,
`crates/loam-runtime/src/session.rs:981`. An application that declares
`recover_on_fault(A)` and also handles `A` with a reset in a Dispatch system gets
two full restores from one press, because the same gathered input reaches the
application afterward. `CommandSender::submit` refuses a duplicate `Reset`, so
one press is meant to mean one reset. Repair, decided by the lead: the session
applies at most one restore per boundary, so a `Reset` queued in the boundary
that follows a host recovery is a no-op; the press still reaches the
application's own handler, which hero needs for its reseed. Verify by changing
`frame.rs:1015` and `:1028` to one advance, then testing both crates.

**R2. A refused command submitted from a system never reaches the console. Defect.**
`crates/loam-app/src/session/commands.rs:248`,
`crates/loam-runtime/src/command.rs:82`. Switch the playground to Toybox and press
`f`: the refusal lands in `session.results()`, `CommandInbox::collect` finds no
matching submission because the command never entered the host inbox, and falls
through to a trace warning naming a request number. Refusals should reach the
console naming what failed, as they do for the host sender. Repair:
add `name` to `CommandResult` and route an unmatched rejection through `deliver`.

**R3. `SimConfig` is not validated at the host boundary. Defect.**
`crates/loam-runtime/src/session.rs:29`, `crates/loam-app/src/session/frame.rs:72`,
`crates/loam-time/src/fixed_timestep.rs:21`. The runtime documents `fixed_hz` of
zero as stopping the simulation and `Session::tick` honors it, but `Frame::new`
passes it into an assert admitting only one and above, so the host panics before
the first frame. Separately `max_ticks_per_frame` of zero stops ticking forever.
Repair: treat zero `fixed_hz` as paused in `Frame::new` and
assert a positive catch-up. Verify with one `loam-app` test.

**R4. `HyperbolicH3::chart_reach` returns infinity at bounding radius one. Defect, latent.**
`crates/loam-runtime/src/domain.rs:730`, consumed at `:2139`. A prepared geometry
with bounding radius at or above one gives `atanh(1.0)`, infinite in f32, so
`hit_ball` returns zero for every ray, `exp` returns the ray origin unchanged,
`Views::depth` refuses it, and the entity is skipped with no refusal counted.
Every other H3 path clamps with `POINCARE_R2_MAX.sqrt()`. Repair: use that
constant at `domain.rs:730`. Verify with one pick test.

**R5. The allocation measurement primitive is unguarded. Design debt.**
`crates/loam-time/src/alloc.rs:95`, against its sibling at `:83`. `current_snapshot`
guards on `ALLOC_INSTALLED` so that "allocated nothing" and "allocator never
installed" cannot be confused; `bytes_allocated_by` does not. Six `loam-runtime`
oracles rely on the allocator declared in `crates/loam-runtime/src/store.rs:824`'s
test module, so moving it turns them green while measuring nothing. `realloc` at
`:55` also double counts a grow-in-place. Repair: one debug assertion plus a doc
line. Verify with one `loam-time` test that installs the allocator.

**R6. The blended WGSL emitter substitutes first-order stand-ins. Design debt.**
`crates/loam-math/src/blended.rs:841` and `:852`, ABI at
`crates/loam-math/src/space.rs:68`. The emitted `loam_log` is a chart difference
and `loam_distance` is a midpoint-rule chord, where the CPU runs a Gauss-Newton
solve and a geodesic length. Measured worst scene SDF residual is 1.889e-2 against
3.6e-7 for H3 at the same extent, roughly 95x the hit epsilon. The approximation
is disclosed; the gap is that `WgslSpace` declares no accuracy relation. Repair:
add an accuracy tier and check the blended fixture at it.

**R7. The pass contract governs depth writers only. Design debt.**
`crates/loam-render/src/pass.rs:68` and `:154`,
`crates/loam-render/src/raymarch/hyperslice4d.rs:901`,
`crates/loam-render/src/passes/line.rs:107`. A Scene-stage pass may clear the
shared color target with nothing to stop it, and the filmstrip does, discarding
the presenter's passes registered before it. Separately `LinePass` reads shared
depth while declaring no convention, so a third-party reader under the other
compare would register silently. Repair: redefine the declaration as the
convention a pass uses reading or writing, and validate a color-load policy.
Verify by extending the registration-refusal test.

**R8. The engine's diagnosis never reaches the surface the user is looking at. Design debt.**
`crates/loam-app/src/session/frame.rs:234`, `crates/loam-app/src/trace.rs:187`,
`crates/loam-time/src/frame_trace.rs:103`, `crates/loam-app/src/session/app.rs:29`.
`Frame::step` returns on an uncaptured device error before anything reaches the
console, so the errors carrying pass name and phase are the ones it never sees.
`PerfOverlay` is constructed only by its own test, no heap sampler is ever
installed, and no application reads the per-pass sections. Repair: note the error
to the console and print sections and uploads from the `trace` command.

**R9. Three error families name a bare noun or the wrong noun. Defect.**
`crates/loam-runtime/src/domain.rs:244`, `:251`, `:2418`, `:2608`, and
`crates/loam-runtime/src/command.rs:58`. `Unsupported` and `InvalidCoordinate`
write their argument verbatim, so the console renders `rejected, image space`.
`DomainError::Stale` is returned for a live entity a domain holds no row for, when
`Dispatch::apply` already proved it resolves and `StoreError::Missing` says that.
`Domains::named` reports missing, ambiguous, and space-mismatched alike. Repair:
give the two variants a clause, return `Missing` at the eight sites, and return
`SpaceMismatch` from `named`. Verify with one console and one runtime test.

**R10. The flagship presents two reset meanings and two inert commands. Defect.**
`crates/polytope_playground/src/ui.rs:74`,
`crates/polytope_playground/src/mode.rs:86`,
`crates/polytope_playground/src/main.rs:607`,
`crates/polytope_playground/src/display.rs:110`. The Edit menu offers "Reset all"
for a command that restores poses, time, angles, spin, and slice and leaves the
shape row and the filmstrip alone; the console names it accurately. `spin` and
`seek` in Toybox return success and change nothing while the console prints
`done`. `mode::set_strip` refuses this class with a named reason. Repair: rename
the item and return those two refusals.

**R11. The root eye's aspect is an unenforced caller obligation. Design debt.**
`crates/loam-app/src/session/camera.rs:95`,
`crates/polytope_playground/src/main.rs:1069` and `:516`,
`crates/loam-app/src/session/frame.rs:276`. The host repairs the root aspect
before and after Dispatch but not after the tick loop, and aspect is load-bearing
for the projection matrix, so a Simulation-phase eye write that drops it publishes
a stretched frame with no error. Three copies of the idiom exist because `orbit()`
has no freecam counterpart. Repair: one engine helper that writes an eye
preserving aspect, plus the repair after the tick loop.

**R12. `LOAM_MAX_ARC` is a disabled sentinel outside S3. Design debt.**
`crates/loam-math/src/hyperbolic.rs:199` and four peers emit 1e9;
`crates/loam-math/src/spherical.rs:214` emits 1.5. The kernel escapes above
0.92 of that value, but H3's origin distance saturates near 17.6, so the escape
and the arc cap can never fire and the kernel can report a hit at a clamped point
that `valid_point` refuses. The loop still terminates on the iteration and scene
caps, so this is debt rather than a defect. Repair: emit the constant from
`chart_envelope()`. Verify with an out-of-envelope H3 probe case.

**R13. Three places where the publication cache gives up more than it needs to. Design debt.**
`crates/loam-runtime/src/domain.rs:1770`, `crates/loam-runtime/src/store.rs:703`,
`crates/loam-runtime/src/session.rs:838`. `view_mut` bumps the view revision on
access while the two typed setters beside it compare and return early; an
untracked domain always resyncs with nothing on the public API saying so; and a
skipped view target truncates every published view after it. The flagship guards
all three of its `view_mut` calls, so none of this costs a shipped frame. Repair:
comparing setters, a tracking accessor, and keying published views by domain and
target. Verify with one unchanged-view stamp test.

**R14. The default log capacity has a cache cliff the bench sizes around. Design debt and evidence gap.**
`crates/loam-runtime/src/store.rs:49`, `crates/loam-runtime/src/command.rs:329`.
Above 4096 rows one whole-store edit overruns the dirty ring, every cursor reports
lost, and publication rebuilds every view every frame; both 10000-row bench cases
set capacity to population plus 64, so neither crosses it. Separately a placed
spawn with one row costs three heap allocations and the churn oracle covers
neither. No contract is violated. Repair: one bench case at the default capacity
and one recorded churn line.

**R15. The browser host is compiled but not exercised or timed. Evidence gap.**
`crates/loam-app/src/session/browser.rs:276` and `:671`. The worker overrides the
default feature request with an empty set, so the section timer is always absent
and the GPU half of the browser frame budget is unmeasurable. The routing table is
type-checked and never run: `Worker::apply` drops button and wheel events while
the cursor is locked where native marks them consumed, and the guard that makes
both safe is the held-pointer cancel in
`crates/loam-app/src/session/input.rs:153`, which no test states. Repair: drop the
override and lift routing into a testable free function.

**R16. `Klein::depth_envelope` is unconditional but its guarantee is not. Design debt.**
`crates/loam-runtime/src/view.rs:729`, `crates/loam-math/src/hyperbolic.rs:12`.
The envelope is reported as 6.0 regardless of eye position, while the constant's
own documentation scopes the ordering claim to an eye within `H3_EYE_CHART_REACH`
of the chart origin and `valid_point` admits an eye out to 6.0. Nothing outside
the depth test reads that reach. The envelope is genuinely proved for the stated
eye position, so what is missing is the precondition. Repair: take the eye pose
and shrink `far`. Verify by sweeping eye distance in the depth envelope test.

**R17. Two numerical paths rest on a pin or a bench. Evidence gap.**
`crates/loam-runtime/tests/blended_transport.rs:120`,
`crates/loam-runtime/benches/fields.rs:386`. The varying-blend endpoint and its
three frame columns are hardcoded f32 literals and the Euler-Lagrange oracle the
comment cites is not computed in the tree, so nothing can check the constants. The
runtime's compiled field program meets the physics contact path only in a
benchmark; the contact tests use hand-written field implementations. Repair:
compute the variational oracle at f64 across several blend positions, and add one
integration test that compiles an exact program and asserts the rest position.

**R18. The H3 conformance fixture covers 14 percent of the declared envelope. Evidence gap.**
`crates/loam-math/tests/space_conformance.rs:1070`. The fixture samples to chart
radius 0.4, metric distance 0.85, against a declared envelope of 6.0. The H3 unit
tests sweep the ball to 0.9999 and the Klein arc-length test reaches 6.0 exactly,
so transport and the Klein chart are covered to the envelope; what is not is exp
and log round-tripping, the triangle inequality, and the angle-excess check.
Repair: raise the fixture radius toward 0.995 and widen the tolerance honestly.

**R19. Hero bypasses the host's headless entry. Design debt.**
`crates/examples/src/bin/hero.rs:877`,
`crates/loam-app/src/session/native.rs:35`, `crates/loam-app/src/args.rs:38`.
`launch_or_headless` installs tracing and the native executor before it branches;
hero parses its flag out of the raw argument vector and calls build and headless
directly, so the loop runs on the default executor and hero's own refusal reports
are dropped. The cause is shared: `Args` accepts only key-equals-value. Repair:
accept the equals form and route through the shared entry.

**R20. `TextPass` turns a bad font into a fatal host error. Design debt, latent.**
`crates/loam-text/src/pass.rs:61` and `:111`. `ready()` is documented as false
when the font never parsed, so the pass draws nothing, but `attach` propagates the
parse error, which `Frame::attach` turns into a host error that stops the native
host or fails browser initialization. The playground passes empty bytes when a
font is absent. It stays latent only because egui always ships default fonts.
Repair: log the failure and leave the renderer unset, as `ready()` promises.

**R21. The shape catalog's category windows are hand-maintained indices. Design debt.**
`crates/polytope_playground/src/catalog.rs:138` and `:111`,
`crates/polytope_playground/src/main.rs:271`. Categories store start and end
indices into a flat array, so inserting a seventh regular polychoron at index six
silently files it under smooth solids and appending files it there too, with no
compile error and no test. Two further couplings ride the same array: card lookup
by struct equality, and cards that must stay index-aligned. Repair: put the
category name on the entry and group at menu build time.

**R22. Two conventions for the same projective singularity. Design debt.**
`crates/loam-math/src/rasterizable.rs:147` and `:184`,
`crates/loam-runtime/src/view.rs:561`. The rasterizer clamps the perspective
denominator with a 1e-4 epsilon and returns a large finite value for a vertex at
the eye, where the runtime returns none and propagates a refusal. The projection
enum is generic over dimension and every variant exists at every dimension, so a
three-dimensional Schlegel type-checks and collapses every point to the origin.
Not reachable from shipped callers. Repair: return an option, or split the
four-dimensional variants out.

**R23. The script command boundary is one level deep. Deferred capability.**
`crates/loam-runtime/src/command.rs:20`, `:43`, `:353`,
`crates/loam-runtime/src/session.rs:673`, `crates/loam-runtime/src/view.rs:253`.
`Outcome` has two variants, so a scripted query has no channel back; the
unsupported rejections take static strings, so a script error cannot name a line
or a value; `Dispatch` has no command queue, so a command cannot queue a
follow-up; there is no public bridge removal and no way to move a placed image
space; and manipulation sits outside the boundary. Additive shape: one outcome
variant carrying a value and one rejection carrying an owned string.

**R24. Four of `loam-time`'s seven dependencies serve a subsystem with no consumer. Design debt.**
`crates/loam-time/Cargo.toml:17`, `crates/loam-time/src/director.rs`. `loam-math`,
`glam`, `serde`, and `ron` are used only by the director. Within it, the
interpolation types have one consumer in hero and the timeline, playhead, drive,
error, validation, and RON loader have none; the RON loader's only caller is its
own test. `glam` is direct while `loam-math` already re-exports it. Replay and
`web-time` are load-bearing and `thiserror` backs three real enums. Repair: delete
the unconsumed half and drop three dependencies, or record the intent.

**R25. The performance baseline contradicts itself about the browser pixel cap. Evidence gap.**
`docs/PERFORMANCE_BASELINE.md` says in one place that the production browser build
has no pixel cap and in another that the host exposes the cap as a launch option
off by default. The code matches the first:
`crates/loam-app/src/wasm/main_launcher.rs:48` reads the cap only under the measure
feature. Recorded with it: the facade does not re-export `ab_glyph`, which a public
text signature requires, so the examples crate declares its own copy. Repair: one
documentation line, and re-export or take bytes.

## Repair rows

- **Row A, `loam-app`.** R1, R2, R8, R11 and the host half of R3. Moves Runtime
  correctness to A and Application ergonomics to A, and is a prerequisite for
  Verification and diagnosis reaching A.
- **Row B, `polytope_playground`.** R10, R21. Moves Behavior and presentation
  fidelity to B; reaching A also needs host observation.
- **Row C, `loam-runtime`.** R9, R13 and the runtime half of R3. Moves Cache
  correctness and value to A, and with Rows A and F moves Verification and
  diagnosis to A.
- **Row D, `loam-math` with two lines in `loam-runtime`.** R4, R12, R16, R18, R22.
  Moves Geometry model to A, and with Row E moves Domain and presentation
  boundaries to A.
- **Row E, `loam-math`, verified through `loam-render` probes.** R6, R17. Moves
  Numerical and physics quality to A, and joins Row D on boundaries.
- **Row F, `loam-time`.** R5, R24 and the catch-up assertion in R3. Moves
  Abstraction value to A.
- **Row G, `loam-render` with one file in `loam-text`.** R7, R20 and the attach
  rollback noted under R8. Moves Rendering architecture to A.
- **Row H, `loam-app`, `loam-runtime` benches, `examples`.** R15, R14, R19, R25.
  Moves Native and WASM separation to A; moves Data layout and scaling to A only
  after the reference device, build, workload, and budget are selected.
- **Row I, `loam-runtime`.** R23. Not required here; Extension cost stays B and
  this is the first work item when the scripting layer starts.

Rows A, C, and F are cheapest: every repair is a few lines with a named test.
Rows D and E are the only ones needing GPU probe reruns.

## Judged not worth doing

- `RecordBuffer::stamp` advancing on unchanged frames. It has a real oracle
  proving a foreign buffer does not restamp a previous session's records. One doc
  line at most; do not delete it.
- The timestep's nanosecond truncation above 500 MHz. Real, but no game runs
  at that rate.
- `Session::restore` leaving a partly restored session readable. `check_restore`
  is documented to refuse whatever restore would refuse while changing nothing,
  the shipped physics facility honors it with a test, and the only violator is a
  test facility breaking it on purpose.
- The parallel chunk runner's poisoned-lock early return, neutralized by the
  scoped executor re-raising the panic, and the two identity-exhaustion panics,
  where stopping instead of wrapping is the documented behavior.
- The frame trace section buffer drained from another crate, which drains every
  frame in the shipped app, and the orphaned placed image space never returning
  to the free list, bounded by nested bridge count with no cycle reachable.
- The field contact error bounding the value and not the normal. The doc makes no
  normal claim and the degenerate gradient refuses rather than normalizing.
- Quotient spaces stopping at `Space`, a correctly scoped deferred capability.
- The line pass whole-slice compare and the presenter refilling every view on one
  rebuild, both documented and correct, and the common input representation under
  a module named for wasm, where the ownership is right and only the name is not.
- Every item whose only content is that something was not observed on a host,
  browser, or floor device. These lower confidence and nothing else.

## Checks run

Referee checks on this head: `git status --short` empty before and after
detaching; `cargo test -p loam-runtime -p loam-app -p loam-math -p loam-physics`
passed 692 with zero failures and three ignored; `cargo clippy -p loam-runtime -p
loam-app -p loam-render --all-targets` clean at deny-warnings.

Area reviewer checks on this head: `cargo test` and `cargo clippy` at
deny-warnings per area, all clean; 25 `space_scene` probes and 44 named GPU
probes on a real adapter; the platform configuration grep; a web bundle build;
and the two demo headless runs.

Gate on this head: `cargo fmt` clean; clippy clean at deny-warnings; 1036 tests
passing; the replay and persist fixtures green; rustdoc clean at deny-warnings;
the benchmark suite builds; 25 `space_scene` probes and 46 GPU probes on a real
adapter; the wasm playground and session crates build; and playground and hero
headless outputs identical to the `48f4609` references.

Not run by anyone: any browser execution, any benchmark result at this head, and
any host observation of appearance or interaction.

## What should remain intact

- Generational identity with the scene epoch, the deliberate refusal to rebase
  application references across a restore, and the `Owner` token gating every
  lifecycle hook that must stay a public trait method.
- The conservative patch bail, the differential publication oracle, and library
  immutability: prepared geometry, materials, and palettes have no mutable
  accessor, so no library edit can invalidate a view cache.
- The five allocation oracles, especially the exact-bytes one, which the realloc
  double count cannot distort.
- The nine geometry capability traits and the compile-time refusals they enforce,
  and the checked-integrator discipline in the blended adapter with its checked
  and unchecked entry points kept separate.
- The conservative field contact gate and its interval evaluation, the conformance
  suite's property-based design, the H3 unit tests, and the spherical arc cap.
- The pass schedule's two enforced rules, error attribution hoisting a device
  error into the running pass, and device replacement with per-cache re-upload.
- The presenter's upload skip keyed on domain, target, and built stamp, with the
  placement hole closed by minting a new image space id.
- The input transport's guards: the bounded queue forcing a release on overflow,
  pointer coalescing that never crosses a transition, and a release delivered for
  a gesture already in flight even when the UI consumed the event.
- The host command inbox rule that an unreported sender still surfaces refusals
  while suppressing successes. R2's repair extends this path, not replaces it.
- The platform configuration job and the pattern pulling wasm-only logic into
  host tests.
- The replay decoder: saturating size arithmetic before any allocation, checkpoint
  ordering enforced on write and read, and fourteen tests each naming a distinct
  malformed input.


## Affected-contract review after the repair rows, head 350a51a

Range `0650128d0d7e5d246a1a8d96c186b8bd36ec4b46..350a51ae6ab6b869d66353a7e8d904685b419274`,
29 commits, 69 files, 1800 insertions and 323 deletions, clean tree with no
untracked source. Two independent reviewers covered the range, one on the
runtime, host, and applications and one on rendering, text, and geometry; this
referee opened every cited site, challenged every closure verdict and every new
finding, and renumbered the two independent finding sets as S1 onward in
consequence order. The target is unchanged: the foundational engine architecture
for the Polytope Playground and the coming scripting layer.

Owner's rules applied. A finding whose only content is that something has not
been observed on a host, browser, or device is an evidence gap that lowers
confidence and not the grade. Nothing built but unused may be deleted, so R24 is
accepted. R23 is deferred to the scripting kickoff, and R14 is covered by
`docs/SCALING_EXPERIMENT.md`, planned and not run.

Referee corrections. The ground clear is latent, not shipped: the only caller of
`TriangleFeed::set_ground` is `crates/examples/src/bin/hero.rs:794`, and hero
never calls `add_view`, so both presenter Scene passes record nothing before the
clear. R22's second half is closed, not untouched:
`crates/loam-math/src/rasterizable.rs:149` refuses the four-dimensional variants
at `N` of three, where the review found a Schlegel collapsing to the origin. The
range is 29 commits; the two reports said 28 and 32.

### Reconciled category grades at 350a51a

| Category | Grade | Confidence | Pass | Reason | Below A |
|---|---|---|---|---|---|
| Application ergonomics | B | High | Regraded | Named refusals reach the console from system submissions and one press means one reset, but a Publication-phase eye write still publishes a stretched frame with no error. | R11, S9 |
| End-to-end ownership | A | High | Carried | The restore fan-out to domains, app state, and views is unchanged by this range; only a boundary flag was added beside it. | none |
| Abstraction value | B | High | Regraded | `look` and the catalog category removed real caller duplication, but the range added two required surfaces with no consumer. | S4, S7 |
| Geometry model | A | High | Regraded | `chart_reach` is finite past the ball at the shared constant, every emitted space carries a real arc cap, the H3 fixture covers its whole 6.0 envelope, and unsupported projections now refuse by dimension. | none |
| Domain and presentation boundaries | B | High | Regraded | Every emitted space declares an accuracy tier that executed probes assert on a real adapter, but the depth envelope that states the boundary's validity limit has no reader on any render or publication path. | S4, S2 |
| Runtime correctness | B | High | Regraded | Both reachable defects are closed with production-path tests, but the one-restore-per-boundary rule is set by every `Session::restore` and documented on neither side of the coupling. | S3 |
| Data layout and scaling | B | Medium | Carried | Neither the store default nor either bench changed in the range, and no bench ran at this head; the allocation probe can no longer confuse zero bytes with an absent allocator. | R14, R5 |
| Cache correctness and value | B | High | Regraded | Access no longer bumps, a real style change bumps before every publication, and published views are keyed by domain and target, but the comparison driving all of it is a hand-listed field set. | S6 |
| Rendering architecture | B | High | Regraded | The pass contract now governs color loads and depth readers and a failed attach cannot record, but a declaration is a live query rather than a commitment and nothing checks a pass against what it records. | S2, S5, R20 |
| Native and WASM separation | B | Medium | Regraded | A wasm32 clippy job now keeps the browser host under deny-warnings, but it omits `--all-targets`, the worker still forces an empty feature set, and the routing table is still executed by no test. | R15 |
| Numerical and physics quality | B | Medium | Regraded | The blended approximation carries a declared residual an executed probe checks and the H3 fixture reaches its full envelope, but the transport endpoint is still a pin and the new eye-scoped envelope has one proved point and one chosen point. | R17, S10 |
| Rust implementation quality | A | High | Regraded | The projective singularity now has one convention at the source and refuses by dimension; the single consumer that discards the refusal is unreachable from any shipped call site. | none |
| Behavior and presentation fidelity | B | Medium | Regraded | Every source-provable item that held the C is repaired, with the labels agreeing, both Toybox refusals named and tested, and one recovery press running one restore. | R11 |
| Verification and diagnosis | B | High | Regraded | Three error families name what was refused, `Missing` replaces `Stale` at ten sites, and `trace passes` prints per-pass CPU and GPU time, but the console half of R8 reaches no rendered frame. | S1, R9, R20 |
| Extension cost | B | High | Regraded | A custom pass costs four methods with the two new declarations defaulted safely, and a new space costs one line that buys an asserted parity tier, but a pass's declaration is a promise nothing keeps. | S5 |

Movement against `0650128`: Geometry model B to A and Behavior and presentation
fidelity C to B. Every other grade held, with different findings holding it.

### Closure of R1 to R25

- R1. Closed. One restore per boundary at `crates/loam-runtime/src/session.rs:1026`, flag cleared at `:810`, tested; hero still reseeds once. Residual S3.
- R2. Closed. `CommandResult.name` at `crates/loam-runtime/src/command.rs:87` and unmatched rejections routed through `deliver` at `crates/loam-app/src/session/commands.rs:238`, with an exact-line console test.
- R3. Closed. `SimConfig::check` at `crates/loam-runtime/src/session.rs:37` enforced at `crates/loam-app/src/session/frame.rs:63`, and a zero rate now pauses instead of tripping the timestep assert.
- R4. Closed. `crates/loam-runtime/src/domain.rs:729` clamps with the shared `POINCARE_R2_MAX` constant, tested at three radii.
- R5. Partly closed. `crates/loam-time/src/alloc.rs:95` returns `None` without the allocator; `realloc` at `:55` still charges gross on a grow in place.
- R6. Closed. `WgslAccuracy` is required on `WgslSpace` at `crates/loam-math/src/space.rs:87` and executed parity probes assert the declared tier.
- R7. Partly closed. `ColorLoad` and `depth_read` are declared and refused at `crates/loam-render/src/pass.rs:216`, and the filmstrip clear moved to Background. Residuals S2 and S5.
- R8. Partly closed. `trace passes` prints per-pass CPU and GPU time from `crates/loam-app/src/trace.rs:119`; the console half is S1, and `PerfOverlay` still has no shipped constructor.
- R9. Partly closed. Three error families name what was refused and ten sites return `Missing`; `crates/loam-runtime/src/domain.rs:2074` still returns `Stale` for a live row-less entity, and `:2644` and `:2647` report missing and ambiguous alike.
- R10. Closed. The Edit item, the refresh hover, and the console help all read the same thing, and `spin` and `seek` return named refusals in Toybox with a test.
- R11. Partly closed. `look` at `crates/loam-app/src/session/camera.rs:95` and the post-tick repair at `crates/loam-app/src/session/frame.rs:332` landed untested, and a Publication-phase eye write still publishes stretched.
- R12. Closed. `H3_MAX_ARC` of 17.5 at `crates/loam-math/src/hyperbolic.rs:10` is emitted by the H3 kernel, with flat and blended peers carrying real caps.
- R13. Closed. Access no longer bumps, a real change bumps inside `synchronize`, and published views are keyed by domain and target, correctly for a skipped middle target. Residual S6.
- R14. Accepted decision. Covered by `docs/SCALING_EXPERIMENT.md`, planned and not run.
- R15. Partly closed. A wasm32 clippy job runs clean at this head; it omits `--all-targets`, `crates/loam-app/src/session/browser.rs:276` still forces an empty feature set, and the routing table is untested.
- R16. Partly closed. `klein_depth_envelope` at `crates/loam-runtime/src/view.rs:744` takes the eye distance. Residuals S4 and S10.
- R17. Not addressed by this range. The pinned transport endpoint stands.
- R18. Closed. The H3 fixture samples to chart radius 0.99506 and metric distance 6.0, with 172 conformance cases passing at unchanged tolerances.
- R19. Closed. Hero enters through `launch_or_headless` and the bare-flag value form is parsed and tested. Residual S9.
- R20. Partly closed. A bad font leaves the renderer unset and the host alive; the message at `crates/loam-text/src/pass.rs:131` is stored and nothing logs or reads it outside one test.
- R21. Closed. The category is a field on the entry and the menu filters on it, with a test; the two couplings R21 named as riding along are unchanged.
- R22. Closed. `project_point` returns `Option`, both clamps are gone, and `crates/loam-math/src/rasterizable.rs:149` refuses the four-dimensional variants at three dimensions. Residual S8.
- R23. Deferred by decision to the scripting kickoff.
- R24. Accepted decision. Nothing built but unused may be deleted.
- R25. Closed. `docs/PERFORMANCE_BASELINE.md:40` separates today's build from the decision, and the glyph type reaches callers through `loam-text`.

### Residual findings

**S1. The frame-failure console note cannot be rendered. Defect.**
`crates/loam-app/src/session/frame.rs:243`. Trigger: any error `Frame::step`
returns. The note is written after `stepped` returns, and that frame's console
was drawn inside `drive` at `:440`; every error on this path is terminal for both
hosts. Contract: R8 asked for the engine's diagnosis on the surface the user is
looking at. Repair: note the error where the non-terminal branch already notes,
at `:427`. Verify with a test that forces an error and reads the console history.

**S2. A ground set after registration clears the shared target in Scene. Design debt, latent.**
`crates/loam-render/src/triangle_pass.rs:104` and `:141`,
`crates/loam-render/src/pass.rs:216`. Trigger: register a triangle pass with no
ground, then set one. `color_load` answers from live feed state and `register`
queries it once, so the Scene-stage clear gate is passed and then contradicted.
Contract: no Scene pass clears the shared color or depth target. Latent because
hero, the only ground caller, publishes no views. Repair: take the color load by
value in `TriangleFeed::pass`. Verify by extending the registration-refusal test.

**S3. Any restore arms the one-restore-per-boundary rule. Design debt, latent.**
`crates/loam-runtime/src/session.rs:967`, `:978`, `:1026`. Trigger: call the
public `Session::restore` with a saved snapshot, then let a queued
`Command::Reset` reach the next boundary. It returns `Ok(Outcome::Done)` while
the session stays on the saved snapshot. Contract: undocumented on both sides.
Not reachable from a shipped caller today; the scripting layer is the next
consumer of both. Repair: set the flag inside `reset`, or state the rule in
`restore`. Verify with one snapshot test that restores and then resets.

**S4. The depth envelope family has no production reader. Design debt.**
`crates/loam-runtime/src/view.rs:735`, `:739`, `:744`. Trigger: none; the
declaration, the default, five implementations, two benches, and three tests are
the only occurrences. R16's repair is expressed and never consumed, and the
unconditional value moved from 6.0 to 1.0 with nothing to observe either.
Contract: a declared validity limit should reach the depth path. Repair: record
the intent in one doc line, since deletion is off the table. Verify by reading.

**S5. Nothing checks a pass against what it records. Design debt.**
`crates/loam-render/src/passes/hyperslice.rs:81` against
`crates/loam-render/src/raymarch/hyperslice4d.rs:810`. Trigger: the full-frame
hyperslice path declares `Clear` and records `Load`. The direction is
conservative and the presenter's own clear guarantees a cleared target, so
nothing is discarded. Contract: the declaration is enforced at registration and
never again. Repair: fold into S2. Verify with the same test.

**S6. The view stamp comparison is a hand-listed field set. Design debt.**
`crates/loam-runtime/src/domain.rs:1208`. Trigger: add a sixth field to
`ViewStyle` at `crates/loam-runtime/src/view.rs:467`. It compiles clean and
silently stops invalidating the view that reads it, which is the failure mode
R13 existed to remove. Contract: every supported edit invalidates the right
output. Repair: destructure both sides so a new field fails to compile.

**S7. `Domain::tracks_changes` is required and has no consumer. Design debt.**
`crates/loam-runtime/src/domain.rs:1579` and `:2101`. Added by this referee.
Trigger: none; the declaration and one implementation are the only occurrences.
R13 asked for a tracking accessor and shipped one nothing reads, while every
future `Domain` implementor pays for it. Repair: one doc line beside it.

**S8. The projective refusal reaches the raster paths as three conventions. Design debt, latent.**
`crates/loam-render/src/triangle_raster.rs:215`,
`crates/loam-render/src/line_raster.rs:378`,
`crates/loam-render/src/point_raster.rs:258`. Trigger: a refused vertex under a
four-dimensional projection. The point path skips, the line path makes a NaN and
filters it, and the triangle path pushes `Vec3::NAN` into the vertex buffer with
no guard. Contract: an explicit refusal should not become undefined clip space.
Latent: every shipped upload passes `EuclideanR3` with `Projection::Identity`,
which never refuses. Repair: drop the triangle rather than write NaN. Verify with
a raster unit test under `Perspective4D`.

**S9. The equals form of a bare flag silently opens a window. Design debt.**
`crates/loam-app/src/args.rs:100`, `crates/loam-app/src/session/native.rs:43`.
Trigger: `hero --headless=1200`. The value lands in the key map, not the bare
flags, so the branch falls through to the windowed host with no message, while
`crates/polytope_playground/src/catalog.rs:197` refuses the space form for
`--shapes` by name. Contract: R19 asked for the equals form to be accepted.
Repair: have `bare_flag_value` fall back to `get`. Verify with one args test.

**S10. The eye-scoped envelope interior is chosen, not derived. Evidence gap.**
`crates/loam-runtime/src/view.rs:744`, `crates/loam-math/src/hyperbolic.rs:15`.
The documented proof covers an eye within `H3_EYE_CHART_REACH`, where the
expression saturates at 6.0. Nothing establishes that an eye at distance three
keeps `H3_DEPTH_SEPARATION` ordering out to a far of four. Repair: sweep eye
distance against the f32 ordering measurement in the render depth test, or say
in the doc that the interior is conservative and unproved.

**S11. The blended parity assertion has 0.57 percent headroom on one adapter. Evidence gap.**
`crates/loam-math/src/blended.rs:875`, asserted at
`crates/loam-render/tests/space_scene.rs:905`. The declared residual of 1.9e-2 is
checked against a measured worst of 1.8892527e-2 on one adapter, where the Exact
spaces sit three orders below their 1e-5. Repair: none until a second adapter is
available.

Noted and not a finding: `the_menu_categories_partition_the_catalog_in_order` at
`crates/polytope_playground/src/catalog.rs:219` pins static data, now that the
menu filters by each entry's own category.

### What to repair and what to accept

One more small repair row is worth it, in one crate each. In `loam-app`, S1, S9,
and the Publication-phase half of R11 with tests for `look` and the post-tick
repair; this moves Application ergonomics to A and is the last thing Verification
and diagnosis needs from the host. In `loam-runtime`, S3, S6, and the two R9 sites
at `crates/loam-runtime/src/domain.rs:2074`, `:2644`, and `:2647`; this moves
Runtime correctness and Cache correctness and value to A. In `loam-render` with
one line in `loam-text`, S2 and S5 taken together as one change plus logging the
stored font failure; this moves Rendering architecture to A. Adding
`--all-targets` to the wasm32 clippy job is one word and belongs here. Every one
of these is a few lines with a named test.

Record as accepted limitations and do not spend a row on them. S4 and S7, where
deletion is off the table and inventing a consumer would be speculative; one doc
line each states the intent. S8, which no shipped call site can reach and which
becomes required before any application rasterizes triangles under a
four-dimensional projection. S10 and S11, both evidence gaps. R5's gross charge
on a grow in place, which the exact bytes oracle depends on staying as it is.
R15's empty feature override and untested routing table, which need a browser run
and not a code change. R17, R14, R23, and R24, all already decided.

### Decision

Acceptable for the stated target, with explicitly accepted limitations. No
identified defect breaks a central current contract of the foundational engine
architecture for the Polytope Playground and the coming scripting layer. The
repair rows closed both defects a user could reach at `0650128` and every
geometry finding, each with a production-path test. The one defect this pass adds,
S1, degrades diagnosis on a frame that was already terminal; it corrupts no state
and blocks no supported operation. Everything else standing is latent, deferred,
or an evidence gap. Hold is not warranted: there is no demonstrated blocking
defect and no required release evidence that is unavailable rather than unrun.

Accepted limitations carried forward unchanged: no host or browser observation
exists for any interactive behavior at this candidate, and the stage move in this
range changed what composites over what, so Behavior and presentation fidelity
and Native and WASM separation both stay at Medium confidence until a host run is
recorded; performance budgets still wait on a reference device, build, workload,
and budget; H3 remains unreachable from any shipped application; and the command
boundary scripting must enter is still one level deep.

Checks run by this referee at `350a51a` on a clean tree: `git status --short`
empty after detaching, and `cargo test -p loam-runtime -p loam-app -p loam-math`
passing with zero failures. Prior evidence used as reported by the two reviewers:
per-crate test and clippy runs, the wasm32 clippy command, 25 `space_scene`
probes, and 46 named GPU probes on a real adapter. Still owed by everyone: any
browser execution, any benchmark result at this head, and any host observation.

## Residual row, head e69d962

The residuals the affected-contract review marked worth repairing landed as one
row of three commits, merged as `d3a1e67`, with the lead's tests in `e69d962`.
Nothing built but unused was deleted.

- S1. `Frame::step` logs a failed frame through `tracing::error` with the pass
  and phase intact. The console note is gone: no path could draw it after an
  error that stops the host.
- S9. `--headless=1200` runs headless like `--headless 1200`; `Args::flag_value`
  reads either form and hero uses it.
- R11. The root eye's aspect is repaired once more after publication, before
  the presenter reads it, so a Publication-phase eye write no longer publishes a
  stretched frame. Test: `a_publication_phase_eye_write_keeps_the_frame_aspect`.
- S3. Only `Session::reset` arms the one-restore rule; a restore to a saved
  snapshot leaves the next `Reset` live. Test:
  `a_restore_to_a_saved_snapshot_does_not_swallow_the_next_reset`.
- S6. `same_style` destructures `ViewStyle` with a full pattern, so a new field
  fails to compile until the comparison names it.
- R9. `spawn_body` reports a missing pose instead of a stale entity, and
  `Domains::named` reports `AmbiguousDomainName` when two domains share a name.
  Tests: `spawning_a_body_on_a_live_entity_without_a_pose_is_missing_not_stale`,
  `a_domain_name_shared_by_two_domains_is_reported_as_ambiguous`.
- S2 and S5. A pass's color load is fixed when the pass is built and is the
  value its render pass opens with. `TriangleFeed::set_ground` refuses to add or
  remove a ground once a pass exists (test:
  `a_ground_cannot_appear_or_vanish_after_the_pass_declared_its_color_load`).
  The Hyperslice pass declares and records a load in both of its modes, so the
  filmstrip cells now show the sky-ground pass behind them rather than a flat
  horizon clear. Hero draws its ground through a registered sky-ground pass in
  the Background stage.
- R20. `TextPass::attach` warns through tracing when a font does not parse.
- R15. `--all-targets` on the wasm32 clippy job is not possible: the loam-app
  bench and its native-only tests do not build for wasm32. The job lints the
  library targets, as before.

Accepted and recorded, not repaired: S4 and S7 (no production reader yet for
the eye-scoped Klein envelope or for `tracks_changes`), S8 (a NaN vertex on an
unreachable path), S10 and S11 (evidence gaps), R5's gross realloc charge, R14
(the planned measurement), R23 (deferred to the scripting kickoff), R24 (kept
by owner decision).

One visual change for the owner to judge: the filmstrip background in the
playground. Everything else in this row is pixel-neutral by construction and the
headless outputs are unchanged.

Owner observation, 2026-09-14, native host at head e69d962: the filmstrip
background drawn by the sky-ground pass was accepted; the reset label and its
behavior, the refused Toybox spin in the console, the hero floor toggle, and
hero reseeding once per press were observed as described. The attached
headless flag was confirmed equal to the space-separated form.
