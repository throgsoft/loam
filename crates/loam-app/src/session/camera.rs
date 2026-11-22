pub use loam_runtime::camera::{FreeCamera, Orbit};
use loam_runtime::{Ctx, Eye, Phase, Session, Stores, Views};

pub fn look(views: &mut Views, eye: Eye) {
    views.root_mut().look(eye);
}

pub fn orbit<A: Stores>(
    session: &mut Session<A>,
    camera: impl Fn(&mut A) -> &mut Orbit + Send + 'static,
) {
    session.system(Phase::Dispatch, "orbit", move |ctx: Ctx<'_, A>| {
        let orbit = camera(ctx.app);
        orbit.apply(ctx.input);
        let eye = orbit.eye();
        look(ctx.views, eye);
        Ok(())
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    loam_runtime::stores! {
        #[derive(Default)]
        pub struct Viewer {
            camera: Value<Orbit>,
        }
    }

    #[test]
    fn a_reset_brings_the_orbit_camera_back_with_the_scene() {
        use loam_runtime::{Input, Pointer, PointerButton, PointerPhase, Session, SimConfig};

        let mut session = Session::new(Viewer::default(), SimConfig::default());
        session.app.camera.set(Orbit::around([0.0; 3], 5.0));
        orbit(&mut session, |app| app.camera.get_mut());
        session.set_initial().unwrap();
        session.boundary(Input::default()).unwrap();
        let root = session.views().root();
        let boot = session.views().get(root).unwrap().eye.position;

        let drag = Input {
            pointers: vec![Pointer {
                id: 0,
                button: Some(PointerButton::Secondary),
                ndc: [0.0; 2],
                delta: [40.0, 0.0],
                phase: PointerPhase::Moved,
                time: 0.0,
            }],
            ..Input::default()
        };
        session.boundary(drag).unwrap();
        let turned = session.views().get(root).unwrap().eye.position;
        assert_ne!(turned, boot, "a secondary drag left the orbit where it was");

        session.reset().unwrap();
        session.boundary(Input::default()).unwrap();
        assert_eq!(
            session.views().get(root).unwrap().eye.position,
            boot,
            "the reset did not bring the camera back"
        );
    }
}
