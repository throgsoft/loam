use glam::{Quat, Vec3, Vec4};
use loam_math::{Bivector, Bivector4, EuclideanR4, Iso4Flat};
use loam_runtime::host::{self, HostConfig, HostError};
use loam_runtime::{
    Access, ActionId, AppCommand, Bindings, Command, Commands, Dispatch, DomainBuilder, Domains,
    Eye, Input, Instance, Key, LogCapacity, Material, Outcome, Phase, Pose, PreparedGeometry,
    Projection4, Rejection, Session, SimConfig, SpawnBundle, Step, Store, ViewSpec, Views,
};
use loam_shape::polytope::Polytope4;

const PAUSE: ActionId = ActionId(0);
const RESET: ActionId = ActionId(1);
const SPIN_RATE: f32 = 0.4;
const FOCAL_DISTANCE: f32 = 2.0;
const POLYTOPE_SCALE: f32 = 1.5;
const ORBIT_RADIANS_PER_NDC: f32 = 2.5;
const ORBIT_PITCH_LIMIT: f32 = 1.45;

#[derive(Clone, Copy)]
struct Spin {
    omega: Bivector4,
    paused: bool,
}

loam_runtime::stores! {
    pub struct TesseractStores {
        spin: Store<Spin>,
    }
    pub struct TesseractRecords {}
    pub struct TesseractSnapshot;
}

struct Pause;

impl AppCommand<TesseractStores> for Pause {
    fn name(&self) -> &'static str {
        "pause"
    }

    fn apply(
        &mut self,
        dispatch: &mut Dispatch<'_, TesseractStores>,
    ) -> Result<Outcome, Rejection> {
        for (_, spin) in dispatch.app.spin.iter_mut() {
            spin.paused = !spin.paused;
        }
        Ok(Outcome::Done)
    }
}

struct Orbit {
    yaw: f32,
    pitch: f32,
    distance: f32,
}

impl Orbit {
    fn drag(&mut self, [dx, dy]: [f32; 2]) {
        self.yaw -= dx * ORBIT_RADIANS_PER_NDC;
        self.pitch =
            (self.pitch - dy * ORBIT_RADIANS_PER_NDC).clamp(-ORBIT_PITCH_LIMIT, ORBIT_PITCH_LIMIT);
    }

    fn eye(&self) -> Eye {
        let back = Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch) * Vec3::Z;
        Eye::looking_at(
            (back * self.distance).to_array(),
            Vec3::ZERO.to_array(),
            Vec3::Y.to_array(),
        )
    }
}

fn tesseract_edges() -> PreparedGeometry {
    let topology = Polytope4::Tesseract.topology();
    let segments = topology
        .edges
        .iter()
        .map(|&[i, j]| {
            [
                (topology.vertices[i as usize] * POLYTOPE_SCALE).to_array(),
                (topology.vertices[j as usize] * POLYTOPE_SCALE).to_array(),
            ]
        })
        .collect();
    PreparedGeometry::Lines4 { segments }
}

fn main() -> Result<(), HostError> {
    let stores = TesseractStores {
        spin: Store::untracked(),
    };
    let mut session = Session::new(stores, SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let edges = session.prepare(tesseract_edges());
    let white = session.add_material(Material::Lines {
        color: [1.0, 1.0, 1.0, 0.95],
        width_px: 1.6,
    });
    let root = session.views().root();

    session.dispatch(|d| -> Result<(), Rejection> {
        d.spawn(
            SpawnBundle::new()
                .at(r4, Pose(Iso4Flat::IDENTITY))
                .instance(Instance {
                    geometry: edges,
                    material: white,
                })
                .row(Spin {
                    omega: Bivector4::basis(2) * SPIN_RATE,
                    paused: false,
                }),
        )?;
        let eye = d.spawn(SpawnBundle::new().at(
            r4,
            Pose(Iso4Flat::from_translation(Vec4::W * FOCAL_DISTANCE)),
        ))?;
        d.domains.typed(r4)?.add_view(ViewSpec {
            image: root,
            eye,
            mapping: Box::new(Projection4 {
                focal: FOCAL_DISTANCE,
            }),
        });
        Ok(())
    })?;

    session.system(
        Phase::Dispatch,
        "actions",
        Access::new().commands(),
        |_: &mut TesseractStores, input: &Input, commands: &mut Commands<TesseractStores>| {
            if input.pressed(PAUSE) {
                commands.app(Pause);
            }
            if input.pressed(RESET) {
                commands.submit(Command::Reset);
            }
        },
    );

    let mut orbit = Orbit {
        yaw: 0.0,
        pitch: -0.15,
        distance: 5.0,
    };
    session.system(
        Phase::Dispatch,
        "orbit",
        Access::new().views(),
        move |_: &mut TesseractStores, input: &Input, views: &mut Views| {
            orbit.drag(input.drag());
            views.root_mut().eye = orbit.eye();
        },
    );

    session.system(
        Phase::Simulation,
        "spin",
        Access::new().reads::<Spin>().domain(r4.id()),
        move |app: &mut TesseractStores, domains: &mut Domains, step: Step| {
            let Ok(r4) = domains.typed(r4) else {
                return;
            };
            for (entity, spin) in app.spin.iter() {
                if spin.paused {
                    continue;
                }
                if let Some(pose) = r4.poses.get_mut(entity) {
                    pose.0.rotation = ((spin.omega * step.dt).exp() * pose.0.rotation).normalize();
                }
            }
        },
    );

    session.set_initial();
    let bindings = Bindings::new()
        .key(Key::Letter('t'), PAUSE)
        .key(Key::Space, PAUSE)
        .key(Key::Letter('r'), RESET);
    host::run(
        session,
        HostConfig {
            title: "tesseract",
            bindings,
        },
    )
}
