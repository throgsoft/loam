use std::f32::consts::TAU;

use loam::prelude::*;

const PAUSE: ActionId = ActionId("pause");
const RESET: ActionId = ActionId("reset");
// Radians per second, one turn in about twenty seconds.
const TURN_RATE: f32 = 0.3;

#[derive(Clone, Copy, Default)]
struct Turn {
    previous: f32,
    angle: f32,
    paused: bool,
}

impl Turn {
    fn drawn(&self, alpha: f32) -> f32 {
        self.previous + (self.angle - self.previous).rem_euclid(TAU) * alpha
    }
}

stores! {
    #[derive(Default)]
    pub struct MandelbulbStores {
        turn: Value<Turn>,
    }
}

fn build(
    args: Args,
) -> Result<(Session<MandelbulbStores>, SessionApp<MandelbulbStores>), HostError> {
    let mut session = Session::new(MandelbulbStores::default(), SimConfig::default());

    session.on_action(PAUSE, |d| {
        let turn = d.app.turn.get_mut();
        turn.paused = !turn.paused;
        Ok(())
    });
    session.reset_on(RESET);
    session.set_rig(Rig::Orbit(Orbit {
        pitch: -0.3,
        ..Orbit::around([0.0; 3], 3.0)
    }));

    session.system(Phase::Simulation, "turn", |ctx| {
        let turn = ctx.app.turn.get_mut();
        turn.previous = turn.angle;
        if !turn.paused {
            turn.angle = (turn.angle + TURN_RATE * ctx.step.dt) % TAU;
        }
        Ok(())
    });

    let pass = RaymarchPass::new(include_str!("../../shaders/mandelbulb.wgsl").to_owned());
    let bindings = Bindings::new()
        .key(Key::Space, PAUSE)
        .key(Key::Letter('r'), RESET);
    let app = SessionApp::with_args(HostConfig::new("mandelbulb", bindings), args)
        .recover_on_fault(RESET)
        .pass(Box::new(pass.clone()))
        .on_frame(move |hook: &mut FrameHook<'_, MandelbulbStores>| {
            let views = hook.session.views();
            if let Some(image) = views.get(views.root()) {
                let angle = hook.session.app.turn.drawn(hook.alpha);
                pass.publish(RayMarchUniforms {
                    params: [angle, 0.0, 0.0, 0.0],
                    ..RayMarchUniforms::from_eye(&image.eye)
                });
            }
        });
    Ok((session, app))
}

fn main() -> Result<(), HostError> {
    launch(build)
}
