use glam::Vec4;
use loam_math::{Bivector, Bivector4, EuclideanR4, Iso4Flat};
use loam_runtime::host::{self, HostConfig, HostError};
use loam_runtime::{
    Access, ActionId, Bindings, Command, Commands, Dispatch, DomainBuilder, Domains, Input,
    Instance, Key, LogCapacity, Material, Orbit, Phase, Pose, PreparedGeometry, Projection4,
    Rejection, Session, SimConfig, SpawnBundle, Step, ViewSpec, Views,
};
use loam_shape::polytope::Polytope4;

const PAUSE: ActionId = ActionId(0);
const RESET: ActionId = ActionId(1);
const SPIN_RATE: f32 = 0.4;
const FOCAL_DISTANCE: f32 = 2.0;
const POLYTOPE_SCALE: f32 = 1.5;

#[derive(Clone, Copy)]
struct Spin {
    omega: Bivector4,
    paused: bool,
}

loam_runtime::stores! {
    #[derive(Default)]
    pub struct TesseractStores {
        spin: Store<Spin>,
    }
}

fn main() -> Result<(), HostError> {
    let mut session = Session::new(TesseractStores::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let topology = Polytope4::Tesseract.topology();
    let edges = session.prepare(PreparedGeometry::edges_of(topology, POLYTOPE_SCALE));
    let white = session.add_material(Material::lines([1.0, 1.0, 1.0, 0.95], 1.6));
    let root = session.views().root();

    session.dispatch(|d| -> Result<(), Rejection> {
        let spin = Spin {
            omega: Bivector4::basis(2) * SPIN_RATE,
            paused: false,
        };
        d.spawn(
            SpawnBundle::new()
                .at(r4, Pose(Iso4Flat::IDENTITY))
                .instance(Instance::new(edges, white))
                .row(spin),
        )?;
        let eye = Pose(Iso4Flat::from_translation(Vec4::W * FOCAL_DISTANCE));
        let eye = d.spawn(SpawnBundle::new().at(r4, eye))?;
        let projection = Projection4 {
            focal: FOCAL_DISTANCE,
        };
        d.domains
            .typed(r4)?
            .add_view(ViewSpec::new(root, eye, projection));
        Ok(())
    })?;

    session.system(
        Phase::Dispatch,
        "actions",
        Access::new().commands(),
        |input: &Input, commands: &mut Commands<TesseractStores>| {
            if input.pressed(PAUSE) {
                commands.app_fn("pause", |d: &mut Dispatch<'_, TesseractStores>| {
                    for (_, spin) in d.app.spin.iter_mut() {
                        spin.paused = !spin.paused;
                    }
                });
            }
            if input.pressed(RESET) {
                commands.submit(Command::Reset);
            }
        },
    );

    let mut orbit = Orbit::around([0.0; 3], 5.0);
    orbit.pitch = -0.15;
    session.system(
        Phase::Dispatch,
        "orbit",
        Access::new().views(),
        move |input: &Input, views: &mut Views| {
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
    host::run(session, HostConfig::new("tesseract", bindings))
}
