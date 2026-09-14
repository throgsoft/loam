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

