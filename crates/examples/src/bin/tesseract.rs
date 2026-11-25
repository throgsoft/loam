use loam::prelude::*;

const SPIN_RATE: f32 = 0.4;
const FOCAL_DISTANCE: f32 = 2.0;
const POLYTOPE_SCALE: f32 = 1.5;

#[derive(Clone, Copy)]
struct Spin {
    omega: Bivector4,
}

stores! {
    #[derive(Default)]
    pub struct TesseractStores {
        spin: Store<Spin>,
    }
}

fn build(args: Args) -> Result<(Session<TesseractStores>, SessionApp<TesseractStores>), HostError> {
    let mut session = Session::new(TesseractStores::default(), SimConfig::default());
    let r4 = session.register_domain(DomainBuilder::new("r4", EuclideanR4));
    let topology = Polytope4::Tesseract.topology();
    let edges = session.prepare(PreparedGeometry::edges_of(topology, POLYTOPE_SCALE));
    let white = session.add_material(Material::lines([1.0, 1.0, 1.0, 0.95], 1.6));

    session.try_dispatch(|d| {
        let spin = Spin {
            omega: Plane4::Xw.unit_bivector() * SPIN_RATE,
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
        d.add_view(r4, eye, projection)
    })?;

    session.set_rig(Rig::Orbit(Orbit {
        pitch: -0.15,
        ..Orbit::around([0.0; 3], 5.0)
    }));

    session.system(Phase::Simulation, "spin", move |ctx| {
        let r4 = ctx.domains.typed(r4)?;
        for (entity, spin) in ctx.app.spin.iter() {
            if let Some(pose) = r4.poses().get(entity) {
                let frame = ((spin.omega * ctx.step.dt).exp() * pose.frame).normalize();
                r4.set_frame(entity, frame)?;
            }
        }
        Ok(())
    });

    let app = SessionApp::with_args(HostConfig::new("tesseract", Bindings::new()), args)
        .pause_and_reset(&mut session);
    Ok((session, app))
}

fn main() -> Result<(), HostError> {
    launch(build)
}
