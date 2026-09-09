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
