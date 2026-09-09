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
