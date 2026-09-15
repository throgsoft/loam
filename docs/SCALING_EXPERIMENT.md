# Scaling experiment

Written 2026-09-14. Not yet run. This is the plan for the one measurement pass
that decides whether the data layout needs work, and which work. It tests a
claim, not a hunch: per-frame cost is linear in what changed and flat in what
did not. The budgets it compares against are in
[PERFORMANCE_BASELINE.md](PERFORMANCE_BASELINE.md).

## Claims under test

1. Idle cost is independent of population. A frame in which nothing moved costs
   the same at ten thousand entities as at one hundred, within noise.
2. Camera-only cost is proportional to the number of views, not to the
   population.
3. One moved entity costs in proportion to that entity's geometry, not to the
   population.
4. All-moved cost is linear in population with no knee.
5. Spawn and despawn churn allocates a bounded amount per event and leaves
   storage bounded after the entities are gone.
6. The dirty log at its default capacity does not turn one whole-store edit into
   a full rebuild every frame for populations the flagship intends to ship.

Claim 6 is the one the review already suspects: the default dirty ring holds
4096 entries, and both large bench cases size their capacity around it.

## Axes

| Axis | Values |
|---|---|
| Population | 100, 200, 400, 800, 1600, 3200, 6400, 12800 entities |
| Edit pattern | idle; camera only; one moved; all moved; churn of five percent per frame |
| Geometry per entity | one segment; a tesseract's 32 edges |
| Views | one section view; a section and a projection |
| Space | R4; H3 |
| Log capacity | default; population plus 64 |

Not every cell runs. The full grid is population by edit pattern at one
geometry, one view, R4, default capacity. The other axes vary one at a time
from that base at two populations, 800 and 6400.

## Measures

- Publication time per case, from the existing harness in
  `crates/loam-runtime/benches/publication.rs`, which prints nanoseconds per
  publish and allocation counts per case. New cases follow its `run_case`
  shape and its name filter.
- Allocations after warm-up, from the counting allocator in `loam-time`.
- Uploaded bytes per frame. The presenter counts uploads but not bytes; add
  the byte count next to it before running.
- GPU section time per pass on native, from the section timer the presenter
  already owns.
- Store and log memory per case: the sizes of the dense stores and the dirty
  rings, read once at the end of each case.
- For the browser, worker CPU time per frame from the measure bundle
  (`cargo xtask web --features loam-app/measure --release`) under Chrome
  DevTools CPU throttling, 4x for the desktop floor, at the pixel cap, for the
  playground's own two workloads only.

Release build. Five repetitions per case. Report medians with the spread, the
adapter, the backend, and the commit. Run nothing else on the machine.

## Procedure

1. Add the byte counter to the presenter and the population sweep to the
   bench. The sweep is one function over the population list, not eight copies.
2. Add a headless flag to the playground that enables wireframe, so its
   headless run publishes segments and measures something. Today it prints a
   constant zero.
3. Run the base grid. Fit each edit pattern's cost against population. Claims
   1 and 2 hold if the fitted slope is zero within the spread. Claims 3 and 4
   hold if the fit is linear and the residuals show no knee.
4. Run the log capacity axis at 6400 with the all-moved pattern, at the default
   capacity and at population plus 64. Claim 6 holds if the two agree. If the
   default is slower by more than the spread, the ring overran and every view
   rebuilt.
5. Run the churn pattern for two thousand frames at 800 and 6400. Claim 5 holds
   if allocations per frame are constant after warm-up and store memory at
   the end equals store memory at the start.
6. Run the two browser workloads and record worker CPU time per frame.

## Bridge to the budget

The development machine is about three times the desktop floor on a single
thread. Publication plus upload measured here, multiplied by three, must fit
inside the share of the 8 ms frame budget assigned to them, which this
experiment sets at 2 ms, at the largest population the flagship intends to
ship. Choose that population before running and write it here.

Intended flagship population: not yet chosen.

## Decision rule

A claim that holds needs no work and gets one line in PERFORMANCE_BASELINE.md.
A claim that fails names its own repair, and only that repair is done:

| Failed claim | Repair |
|---|---|
| 1 or 2 | A per-frame cost that scales with population on an idle or camera-only frame is a cache defect first; find the invalidation before touching layout. |
| 3 | Range uploads keyed by the changed entity's rows instead of whole-slice uploads. |
| 4 | Chunked publication through the `par` shim, or place-once with a cached matrix for camera-driven work, whichever the profile names. |
| 5 | Bounded scratch for the allocating path the counter names; the churn oracle then pins it. |
| 6 | Size the default dirty ring from the population at registration, or make the overrun resync cheaper than a rebuild. |

A budget miss with every claim holding means the budget or the population is
wrong, not the layout. Record it and stop.

## What this does not decide

Frame appearance, GPU cost on the floor devices, and phone thermals. Those
belong to the baseline's device checks, not to this experiment.
