use loam::prelude::*;

const PAUSE: ActionId = ActionId("pause");
const RESET: ActionId = ActionId("reset");
const SPIN_RATE: f32 = 0.4;
const FOCAL_DISTANCE: f32 = 2.0;
const POLYTOPE_SCALE: f32 = 1.5;

#[derive(Clone, Copy)]
struct Spin {
    omega: Bivector4,
    paused: bool,
}

stores! {
    #[derive(Default)]
    pub struct TesseractStores {
        spin: Store<Spin>,
        camera: Value<Orbit>,
    }
}

fn build(args: Args) -> Result<(Session<TesseractStores>, SessionApp<TesseractStores>), HostError> {
    let mut session = Session::new(TesseractStores::default(), SimConfig::default());
    let r4 = session.register_domain(DomainBuilder::new("r4", EuclideanR4));
    let topology = Polytope4::Tesseract.topology();
    let edges = session.prepare(PreparedGeometry::edges_of(topology, POLYTOPE_SCALE));
    let white = session.add_material(Material::lines([1.0, 1.0, 1.0, 0.95], 1.6));

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
        let projection = Projection4 {
            focal: FOCAL_DISTANCE,
        };
        d.add_view(r4, eye, projection)?;
        Ok(())
    })?;

    session.on_action(PAUSE, |d| {
        for (_, spin) in d.app.spin.iter_mut() {
            spin.paused = !spin.paused;
        }
        Ok(())
    });
    session.reset_on(RESET);

    let mut orbit = Orbit::around([0.0; 3], 5.0);
    orbit.pitch = -0.15;
    session.app.camera.set(orbit);
    loam::app::session::orbit(&mut session, |app: &mut TesseractStores| {
        app.camera.get_mut()
    });

    session.system(Phase::Simulation, "spin", move |ctx| {
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
    });

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
