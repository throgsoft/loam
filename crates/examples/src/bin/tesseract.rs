use glam::Vec4;
use loam::app::session::{launch, Orbit, SessionApp};
use loam::math::{Bivector, Bivector4, EuclideanR4};
use loam::runtime::host::{HostConfig, HostError};
use loam::runtime::{
    ActionId, Bindings, Command, Dispatch, DomainBuilder, DomainError, Instance, Key, LogCapacity,
    Material, Outcome, Phase, Pose, PreparedGeometry, Projection4, Rejection, Session, SimConfig,
    SpawnBundle, ViewSpec,
};
use loam::shape::polytope::Polytope4;

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

loam::runtime::stores! {
    #[derive(Default)]
    pub struct TesseractStores {
        spin: Store<Spin>,
    }
}

fn build(
    args: loam::app::args::Args,
) -> Result<(Session<TesseractStores>, SessionApp<TesseractStores>), HostError> {
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
                .at(r4, Pose::at(Vec4::ZERO))
                .instance(Instance::new(edges, white))
                .row(spin),
        )?;
        let eye = Pose::at(Vec4::W * FOCAL_DISTANCE);
        let eye = d.spawn(SpawnBundle::new().at(r4, eye))?;
        let projection = Projection4 {
            focal: FOCAL_DISTANCE,
        };
        d.domains
            .typed(r4)?
            .add_view(ViewSpec::new(root, eye, projection))?;
        Ok(())
    })?;

    session.system(
        Phase::Dispatch,
        "actions",
        |ctx: loam::runtime::Ctx<'_, TesseractStores>| {
            if ctx.input.pressed(PAUSE) {
                ctx.commands
                    .try_app_fn("pause", |d: &mut Dispatch<'_, TesseractStores>| {
                        for (_, spin) in d.app.spin.iter_mut() {
                            spin.paused = !spin.paused;
                        }
                        Ok(Outcome::Done)
                    });
            }
            if ctx.input.pressed(RESET) {
                ctx.commands.submit(Command::Reset);
            }
            Ok(())
        },
    );

    let mut orbit = Orbit::around([0.0; 3], 5.0);
    orbit.pitch = -0.15;
    loam::app::session::orbit(&mut session, orbit);

    session.system(
        Phase::Simulation,
        "spin",
        move |ctx: loam::runtime::Ctx<'_, TesseractStores>| -> Result<(), DomainError> {
            let r4 = ctx.domains.typed(r4)?;
            for (entity, spin) in ctx.app.spin.iter() {
                if spin.paused {
                    continue;
                }
                if let Some(pose) = r4.poses().get(entity) {
                    let frame = ((spin.omega * ctx.step.dt).exp() * pose.frame).normalize();
                    r4.set_frame(entity, frame)?;
                }
            }
            Ok(())
        },
    );

    session.set_initial()?;
    let bindings = Bindings::new()
        .key(Key::Letter('t'), PAUSE)
        .key(Key::Space, PAUSE)
        .key(Key::Letter('r'), RESET);
    let app =
        SessionApp::with_args(HostConfig::new("tesseract", bindings), args).recover_on_fault(RESET);
    Ok((session, app))
}

fn main() -> Result<(), HostError> {
    launch(build)
}
