# Performance baseline

Decided 2026-09-14. Allowed to be wrong, rewritten without ceremony.

This is the product target that scaling work measures against. The first
shipped consumer is the Polytope Playground embedded in a web page, running
in a dedicated worker with the page's own UI drawn over the canvas. The page
has its own fallback when WebGPU is missing, so Loam targets the devices that
run WebGPU at all and nothing below that line.

## Floor devices

The floor is the weakest hardware class that current browsers ship WebGPU on
by default.

| Host | Floor | Stands in for |
|---|---|---|
| Desktop browser | 2018 thin laptop: Intel UHD 620, four cores, 8 GB, 1920x1080 at device pixel ratio 1, current Chrome or Edge on Windows | Any laptop or desktop Chrome does not blocklist |
| Phone browser | 2021 mid-range Android: Adreno 619 or Mali-G57 class, 6 GB, Android 12, Chrome; or an iPhone 11 on iOS 26 Safari | Any phone with WebGPU on by default |

Native builds inherit the desktop floor. Anything below the floor gets the
page's fallback, not a degraded Loam.

## Budgets

Loam's share of the frame, measured on the floor. The page's UI owns the main
thread; Loam runs in a worker.

| Budget | Desktop floor | Phone floor |
|---|---|---|
| Frame | Presents at the display rate. Worker CPU plus GPU under 8 ms per frame. | 30 frames per second sustained after five minutes. Under 16 ms per frame. |
| Main thread | Under 1 ms per frame of Loam work: input forwarding and messages. | Same |
| Render target | At most 2.1 million pixels, a 1920x1080 frame. Device pixel ratio is honored up to that cap. | Same |
| Startup | Under 1.5 s from script start to the first presented frame, network excluded. | Under 3 s |
| Bundle | Playground under 1.5 MB after brotli. | Same |
| Memory | Under 256 MB of wasm linear memory at peak. | Same |
| Allocation | No recurring allocation in Simulation, Publication, or upload after warm-up. | Same |

At the time of the decision the playground bundle is 6.25 MB raw and
1.76 MB after brotli, and the production browser build has no pixel cap. The
cap exists only in the measure build.

## Workloads

The ladder for the playground. Each row runs idle and interacting, native and
in the browser.

1. Rotate mode with the default shape.
2. Toybox with a dragged shape.
3. Toybox, camera only, with the shipped population.
4. Toybox growth: spawn until a budget is missed. That population is the
   demo's content limit and is recorded with the candidate commit.

Record CPU and GPU time separately, the frame-time distribution, uploaded
bytes, allocations after warm-up, wasm memory, adapter, backend, and commit.

## Measuring without the floor in the room

The development machine is an RTX 4090 Laptop with an i9-13980HX. Until a
floor device is available, a candidate is accepted on these proxies with
25 percent headroom. A run on a real floor device is the truth when one
appears.

- CPU: build the measure bundle with
  `cargo xtask web --features loam-app/measure --release` and run it under
  Chrome DevTools CPU throttling, 4x for the desktop floor and 6x for the
  phone floor. This CPU's single-thread speed is about three times the
  desktop floor and four to five times the phone floor.
- GPU: this GPU fills pixels about thirty times faster than either floor.
  Run Chrome with `--disable-gpu-vsync --disable-frame-rate-limit`, measure
  at the pixel cap and at four times the cap, take the per-pixel slope, and
  multiply by thirty.
- Phone thermals: no proxy. The sustained-rate budget is checked only on a
  device.

## What this decides

- The browser host exposes its existing render target cap as a launch
  option, off by default. The engine enforces nothing; the page that embeds
  the playground sets the cap, and the measurements below assume it did.
- The wasm build gets a release profile (fat LTO, one codegen unit, abort on
  panic, size-optimized) so measured timings and bundle size are the shipped
  ones. This is build hygiene, not engine work.
- Scaling work such as range uploads, placing each vertex once, per-cell
  edge lists, and chunked publication proceeds only against a measured miss
  on this ladder.
