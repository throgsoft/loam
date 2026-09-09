# Performance

## Capture

Run the target scene in a release build. Let it settle, open the console
with backtick, and run `trace summary`. The output reports section p50,
p95, p99, and maximum durations.

Record the revision, scene settings, resolution, sample count, CPU, GPU,
OS, backend, presentation mode, and frame cap. Capture the same workload
before and after a hot-path change.

CPU and GPU sections can overlap. Do not add their percentiles or subtract
their medians to attribute frame time. Use `unscoped` to locate missing CPU
instrumentation, then inspect the relevant frame or section. Display waits
and scheduling delays need measurements that distinguish them.

## Physics

### Island solve, 2026-09-09, 32 logical cores, ecs 0ef4ee8 to unit4b-physics-persist 30d5660

`cargo bench -p loam-app --bench island_solve` in release, with `LOAM_PAR=1`
for the parallel column. The scene is columns of three stacked spheres on a
half-space floor, one island per column, settled for 180 steps. Each cell is
nanoseconds per `World::step`, the median of 15 batches of 20. The CPU
model, OS, and compiler were not recorded.

| islands | bodies | pgs_iters | ecs serial | branch serial | branch parallel |
|---|---|---|---|---|---|
| 64 | 193 | 8 | 87810 | 81630 | 83100 |
| 64 | 193 | 64 | 382505 | 297265 | 291390 |
| 256 | 769 | 8 | 555665 | 480365 | 488895 |
| 256 | 769 | 64 | 2155895 | 1302335 | 1317445 |
| 512 | 1537 | 8 | 1177115 | 975125 | 960955 |
| 512 | 1537 | 64 | 4976110 | 2626305 | 1880030 |
| 1024 | 3073 | 8 | 2819960 | 2045330 | 1804610 |
| 1024 | 3073 | 64 | 12073055 | 5866640 | 3201985 |
| 2048 | 6145 | 8 | 6597855 | 4584680 | 4843110 |
| 2048 | 6145 | 64 | 32460725 | 11959480 | 8011430 |

The serial column got faster on its own. The solve now gathers each island
into scratch, which replaced a `BTreeMap` lookup per constraint per
iteration; the gain runs from 7% at 64 islands and 8 iterations to 63% at
2048 islands and 64 iterations. Below 512 islands the parallel path is the
serial path. From 512 islands it is break-even to 12% faster at 8 iterations
and 28% to 45% faster at 64, with one regression of 5.6% at 2048 islands and
8 iterations that a persistent pool would remove. The first shape tried, one
scoped thread per chunk, was 8x to 14x slower at 16 islands because each
scoped thread costs 30 to 40 us to start, so the worker count scales with
the island count through `ISLANDS_PER_SOLVE_WORKER`.

### Broadphase sweep, same date, machine, and commits

`cargo bench -p loam-physics --bench broadphase`, nanoseconds per sweep,
ecs then branch: 101 bodies 8174 and 8610, 201 bodies 22416 and 20955,
401 bodies 63043 and 62727.

## Presentation

### Edit to result, 2026-09-09, debug build, unit5b-materials 10c4862

`cargo run -p examples --bin twospace -- --edit-latency`, a debug build,
headless. A Dispatch-phase system submits a command through
`Commands::app_fn` that moves the R⁴ landmark; the host ticks and publishes
until the published instance record moves. Over 128 samples the median was
0 ticks and 0.0172 to 0.0173 ms of wall time across three runs, measured
from the submission inside the system to the publish that shows the move.
Zero ticks is the contract: a command a dispatch entry submits commits in
the same boundary. `app_fn` boxes a closure per command; the typed path,
`Domain::apply(&ChartCommand)`, is still `todo!()`. The CPU model and OS
were not recorded.

## Fields

### Compile and traversal, 2026-09-09, debug build, unit6a-fields 72ba7c1

The scene is 1000 spheres in a balanced union tree. `FieldCompiler::compile`
reports these costs for one edit each, from a fixture that spawned the tree
in a domain store and edited it in place:

| edit | changed_inputs | affected_dependencies | program_layout | index_maintenance | full_rebuild |
|---|---|---|---|---|---|
| one primitive moved | 1 | 10 | 1 | 0 | no |
| one operator's operand list changed | 1 | 0 | 2999 | 4 | no |
| one primitive added | 1 | 0 | 3002 | 1998 | yes |

Traversal on the same scene, `cargo test -p loam-render --test
field_traversal -- --nocapture`, 16 rays: 336 steps and 336000 primitive
evaluations for both the interpreter and loam-scene's specialized emit, 21
steps per ray and 1000 evaluations per step; the interpreter retired 671664
instructions and the hits agree to 1e-4. The two differ only in
per-instruction dispatch, and both are linear in the population until a
hierarchy prunes it. The CPU model and OS were not recorded.

### Hierarchy, contact query, and build, 2026-09-09, unit6b-field-bounds d6ee385

`cargo test -p loam-render --test field_traversal -- --nocapture` for the
CPU rows, and the same with `--include-ignored --test-threads=1` for the
GPU rows. Two scenes of 1000 spheres: the balanced union tree above, radius
0.35, and a 100-unit cube, a 10 by 10 by 10 lattice with radius 4. CPU rows
are 16 rays from the origin in a debug build; GPU rows are a 32 by 32
counting kernel that the shipped kernel does not contain. Every count is
per ray.

| scene | path | hits | steps | evaluations | node visits | node skips |
|---|---|---|---|---|---|---|
| balanced, CPU | unculled | 16 of 16 | 21 | 21000 | | |
| balanced, CPU | hierarchy | 16 of 16 | 21 | 784.4 | 6861.3 | 2656.7 |
| cube, CPU | unculled | 14 of 16 | 27.44 | 27437.5 | | |
| cube, CPU | hierarchy | 14 of 16 | 27.44 | 1041.9 | 9854.7 | 3899.1 |
| balanced, GPU | unculled | 1014 pixels | | 26226.8 | | |
| balanced, GPU | hierarchy | 1014 pixels | | 894.9 | 8112.3 | 3174.3 |
| cube, GPU | unculled | 848 pixels | | 31988.2 | | |
| cube, GPU | hierarchy | 848 pixels | | 1181.2 | 10766.6 | 4218.1 |

The hit sets are identical culled and unculled on both interpreters, and
the CPU values are bit-identical. Culling pays only when subtree balls are
separated relative to the current best distance; the worst case is the
unculled work plus 2n - 1 ball tests, with no O(log n) promise. The build
is O(n log n) and runs on every compile that changes a pose: 1000
primitives give 1999 nodes, a full compile reports changed_inputs 1999,
program_layout 2999, and index_maintenance 3997 (1998 edges plus 1999 node
writes), and a pose-only compile rewrites all 1999 nodes.

A field contact query on the warmed balanced field, release build, median
of 1000: 6.5 us (p90 7.3) for one distance plus the eight-sample gradient,
against 47.7 us unculled; the `FieldNarrowphase::test` wrapper measured
below the clock's resolution. These two timings came from throwaway tests
the writer deleted, because no crate links both loam-physics with `r3` and
loam-runtime; nothing in the tree reproduces them.
