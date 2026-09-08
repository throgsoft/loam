[![CI](https://github.com/throgsoft/loam/actions/workflows/ci.yml/badge.svg)](https://github.com/throgsoft/loam/actions/workflows/ci.yml)

![Letters tumbling into the Loam wordmark](assets/readme/hero.webp)

Loam is a Rust game engine in development for games in higher dimensions,
curved spaces, and quotient spaces.

The engine has geometry, rigid-body physics in flat 2D through 4D, SDF and
raster rendering, and native and browser app runners. Rhai gameplay
scripting is a design target. The current script runner plays console
commands at specified frame indices.

[Project thesis](docs/THESIS.md) explains the purpose.
[Architecture](docs/ARCHITECTURE.md) describes the implementation and its limits.
[API documentation](https://throgsoft.github.io/loam/) describes the crates.

## Polytope Playground

Rotate 4D shapes and inspect their 3D cross-sections.

![Regular 4-polytopes turning through a rotation plane while their 3D cross-sections change](assets/readme/rotate.webp)

Pick up and throw objects in the 4D Toybox, inspired by
[Marc ten Bosch](https://marctenbosch.com/)'s [4D Toys](https://4dtoys.com/).

![Polychora dropped into a box under 4D gravity, picked up and thrown](assets/readme/toybox.webp)

## Run

Install [Rust](https://rust-lang.org/tools/install/). Rustup selects the
project's pinned toolchain. Windows needs the C++ build tools. macOS needs
the command line tools from `xcode-select --install`.

On Ubuntu or Debian:

```sh
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libwayland-dev \
  libxkbcommon-dev libxkbcommon-x11-dev libx11-dev libxi-dev \
  libxcursor-dev libxrandr-dev
```

```sh
git clone https://github.com/throgsoft/loam.git
cd loam
cargo run --release --locked -p polytope_playground
```

Use the Demo menu to switch scenes. Space pauses rotation. Backtick opens
the console.

For the browser build:

```sh
rustup target add wasm32-unknown-unknown
cargo install --locked trunk
trunk serve crates/polytope_playground/index.html
```

## License

MIT OR Apache-2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
