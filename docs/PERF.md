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
