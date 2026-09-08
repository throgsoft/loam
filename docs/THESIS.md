# Loam

Loam is a game engine for games set in spaces that are not flat 3D: higher
dimensions, curved manifolds, and spaces whose global shape does not match
their local appearance. It exists to build the games its author wants to play.

## Why build this engine

The geometry of a world can change what a player can do. A fourth spatial
axis changes how objects fit together. Curvature changes paths and apparent
distances. A space can look ordinary nearby and connect back to itself in
unexpected ways. These are material for game rules, puzzles, movement, and
atmosphere.

Showing these spaces is only part of the problem. The player needs to act
in them. A cross-section becomes useful when the player can move an object
through it, catch it, or use it to solve a problem. Loam should make these
interactions possible without requiring each game to build its own geometry,
physics, renderer, and platform layer.

The early repository explored whether the mathematics could work on screen.
The next purpose is to make games with it. A useful engine lets a developer
express a rule, change it, and see its effect. Rhai gameplay scripting is part
of that direction. Game rules should not require editing renderer code or
rebuilding the engine.

The project serves concrete games. Public demos, source, and writing can
share the work, but an imagined future audience does not justify an unused
abstraction. Reuse matters when it removes work from making a game.

## What matters

Geometry should be explicit where it affects an algorithm. A camera path
must not silently assume a flat space. A collision algorithm must state
where it works. One common interface can express useful shared operations;
it cannot make every algorithm valid in every geometry.

Interaction comes first. Polytope Playground asks how a person can manipulate
4D objects through a 2D input device. Physics, cross-sections, projections,
and controls all contribute to that question. A mathematically accurate
picture that the player cannot use is an unfinished interaction.

Correctness and approximation serve the same game. Reference algorithms,
analytic identities, and independent examples can establish what a fast
path preserves. Some games need accurate contact points. Others need a
convincing image within a frame budget. The required accuracy depends on
the use. Maintaining two implementations of every operation is not a goal.

Performance matters on the hardware that will run the game. A renderer that
works only in a small demonstration does not establish that a whole game
will fit. Measure representative scenes, including interaction and physics.
Keep those measurements with enough context to repeat them.

Determinism is a choice a game developer must be able to enforce. Replay,
debugging, or game rules may need repeatable simulation. The engine should
provide control over time, input order, randomness, and state changes.
Fixed timestep is one part of that control, not proof of repeatability.

That choice must leave room for parallel work, SIMD, approximation, and GPU
computation. A game that needs exact replay can require a stable execution
order for its simulation. Other work can accept results that vary within
its accuracy requirements. Universal bit identity across platforms is not
the project's default promise. A historical floating-point hash should not
prevent an otherwise valid improvement.

Stable Rust and manageable dependencies remain practical goals. So do clear
ownership and APIs that a script can use. Neither a ban on ECS nor a custom
scheduler follows from the geometry. Choose storage and execution from the
game's actual needs.

## Which spaces

Flat higher-dimensional spaces are the most direct starting point. R⁴
extends familiar mechanics while introducing rotation planes and interactions
that cannot occur in R³. It gives the Playground a concrete place to test
whether the engine makes geometry tangible.

Constant-curvature spaces offer a second direction. Spherical and hyperbolic
geometry change movement and vision while retaining useful analytic
structure. Rendering and geodesic motion are further along than general
curved rigid-body contact. A shared space type does not erase that gap.

Global connections offer another way to change a world. Tori, lens spaces,
and rooms joined through portals can alter navigation without requiring a
variable metric everywhere. Local curvature and global topology are separate
choices. A game may need one without the other.

Variable curvature remains useful to explore, but it is not a requirement
for every game. Numerical geometry can cost enough to limit where it belongs.
A research implementation earns continued maintenance when it supports a
used feature or answers a question relevant to a game.

## From demonstrations to games

The Playground should remain a place to understand and manipulate shapes.
The tesseract demo and the animated wordmark exercise smaller uses of the
same engine. Their reusable operations belong in engine libraries; their
menus, color choices, and choreography belong with the demonstrations.

The earlier direction included dimensional puzzle and roguelite ideas,
Suika-like physics, and horror built around unusual spatial connections.
Those ideas explain the engine's interests. They are not a fixed delivery
sequence. Scripting and reusable interaction should support whichever game
becomes concrete next.

The engine is ready for that next step when adding a game rule mostly means
writing game code. Geometry, picking, simulation, and rendering should fit
together through explicit contracts. The demos should expose missing pieces,
not accumulate private substitutes for the engine.

[Architecture](ARCHITECTURE.md) describes the implementation and its current
limits. This thesis states why the project exists and what should guide its
choices.
