#![cfg(feature = "physics")]

use std::f32::consts::FRAC_PI_2;

use loam_math::{Bivector4, EuclideanR4, Space};
use loam_physics::euclidean_r4::{register_default_narrowphase, sphere_body_r4};
use loam_runtime::{
    Ctx, DomainBuilder, DomainHandle, DragRelease, Entity, GrabConfig, GrabHold, Input, Instance,
    LogCapacity, Material, Phase, PhysicsConfig, Pointer, PointerButton, PointerPhase, Pose,
    PreparedGeometry, Section4, Session, SimConfig, SpawnBundle, ThrowConfig, ViewId,
};
use loam_time::par::Sequential;

type Vec4 = <EuclideanR4 as Space>::Point;

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Probe {
        carriable: Value<bool>,
    }
}

const BODY: Vec4 = Vec4::new(0.0, 0.0, -2.5, 0.0);
const RADIUS: f32 = 0.5;
const GAIN: f32 = 0.5;
const MAX_SPEED: f32 = 3.0;
const FLICK_SECONDS: f64 = 0.1;

struct Stage {
    session: Session<Probe>,
    r4: DomainHandle<EuclideanR4>,
    section: ViewId,
    body: Entity,
}

fn hold() -> GrabConfig<EuclideanR4> {
    GrabConfig::new(10.0, 100.0, 1.0e4)
}

fn throw() -> ThrowConfig {
    ThrowConfig {
        gain: GAIN,
        max_speed: MAX_SPEED,
        spin_gain: 0.0,
        max_spin: 1.0e3,
    }
}

fn stage(grab: GrabConfig<EuclideanR4>) -> Stage {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session.register_domain(
        DomainBuilder::new("r4", EuclideanR4)
            .tracked(LogCapacity::default())
            .physics(PhysicsConfig::new(register_default_narrowphase).grab(grab))
            .unwrap(),
    );
    // A quarter-turn field of view puts NDC 1 one unit off the axis per unit of depth.
    session.views_mut().root_mut().eye.fov_y = FRAC_PI_2;
    let outline = session.prepare(PreparedGeometry::Lines4 {
        segments: vec![[[RADIUS, 0.0, 0.0, 0.0], [-RADIUS, 0.0, 0.0, 0.0]]],
    });
    let material = session.add_material(Material::flat([1.0; 4]));
    let (section, body) = session.dispatch(|d| {
        let section = d
            .add_view(r4, Pose::at(Vec4::ZERO), Section4 { w: 0.0 })
            .unwrap();
        let body = d
            .spawn(
                SpawnBundle::new()
                    .at(r4, Pose::at(BODY))
                    .instance(Instance::new(outline, material))
                    .body(sphere_body_r4(Vec4::ZERO, Vec4::ZERO, RADIUS, 1.0).unwrap()),
            )
            .unwrap();
        (section, body)
    });
    Stage {
        session,
        r4,
        section,
        body,
    }
}

impl Stage {
    fn boundary(&mut self, input: Input) {
        self.session.boundary(input).unwrap();
    }

    fn grab(&mut self) {
        let pick = self.session.grab([0.0, 0.0], 0.0).unwrap();
        assert_eq!(pick.entity, self.body);
        self.boundary(Input::default());
    }

    // The grab plane is 2 units down the eye's axis, so the pointer crosses 2 * ndc_x units of it.
    fn drag(&mut self, ndc_x: f32) {
        self.session.drag([ndc_x, 0.0], FLICK_SECONDS).unwrap();
        self.boundary(Input::default());
        self.session.tick(&Sequential).unwrap();
    }

    fn motion(&self) -> (Vec4, Bivector4) {
        let physics = self.session.domains().read(self.r4).unwrap();
        let physics = physics.physics().unwrap();
        let row = physics
            .world()
            .body(physics.body(self.body).unwrap())
            .unwrap();
        (row.velocity, row.angular_velocity)
    }

    fn held(&self) -> bool {
        let domain = self.session.domains().read(self.r4).unwrap();
        domain.physics().unwrap().is_held(self.body)
    }
}

#[test]
fn release_throw() {
    let thrown = |ndc_x: f32| {
        let mut stage = stage(hold().throw(throw()));
        stage.grab();
        // A slice that moves under a held body drives it along the axis the section hides.
        let (r4, section) = (stage.r4, stage.section);
        stage.session.dispatch(|d| {
            let view = d.domains.typed(r4).unwrap().view_mut(section).unwrap();
            view.set_mapping(Section4 { w: 0.5 });
        });
        stage.drag(ndc_x);
        let carried = stage.motion().0;
        assert!(
            carried.w > 1.0,
            "the carry never moved the body along w: {carried}"
        );
        stage.session.release().unwrap();
        stage.boundary(Input::default());
        (carried, stage.motion().0)
    };

    let (carried, slow) = thrown(0.1);
    let want = Vec4::new(2.0 * GAIN, 0.0, 0.0, carried.w);
    assert!(
        (slow - want).length() < 1e-3,
        "a pointer at 2 units a second threw the body at {slow}, not {want}"
    );

    let (carried, fast) = thrown(0.4);
    let want = Vec4::new(MAX_SPEED, 0.0, 0.0, carried.w);
    assert!(
        (fast - want).length() < 1e-3,
        "a pointer at 8 units a second threw the body at {fast}, not the capped {want}"
    );
}

#[test]
fn cancel_plain_grab_and_anchor() {
    type End = fn(&mut Session<Probe>) -> Option<DragRelease>;
    let kept = |case: &str, grab: GrabConfig<EuclideanR4>, end: End| {
        let mut stage = stage(grab);
        stage.grab();
        stage.drag(0.4);
        let carried = stage.motion();
        assert!(
            carried.0.length() > 0.5,
            "{case}: the carry left the body still"
        );
        end(&mut stage.session).unwrap();
        stage.boundary(Input::default());
        assert_eq!(stage.motion(), carried, "{case}");
        assert!(!stage.held(), "{case}: the body is still held");
    };
    kept("a cancelled carry", hold().throw(throw()), |session| {
        session.cancel_drag()
    });
    kept("a grab with no throw", hold(), |session| session.release());
    kept(
        "an anchor hold",
        hold().throw(throw()).hold(GrabHold::Anchor),
        |session| session.release(),
    );
}

#[test]
fn despawn_while_held() {
    let mut stage = stage(hold());
    stage.grab();
    assert!(stage.held());
    let body = stage.body;

    stage.session.dispatch(|d| d.despawn(body)).unwrap();

    stage
        .session
        .tick(&Sequential)
        .expect("the hold outlived its body and failed the step");
}

#[test]
fn program_refusal() {
    let mut stage = stage(hold());
    stage
        .session
        .system(Phase::Dispatch, "carry", |mut ctx: Ctx<'_, Probe>| {
            ctx.carry(PointerButton::Primary, |app, _| *app.carriable.get());
            Ok(())
        });
    let press = Input {
        pointers: vec![Pointer {
            id: 0,
            button: Some(PointerButton::Primary),
            ndc: [0.0, 0.0],
            delta: [0.0; 2],
            phase: PointerPhase::Began,
            time: 0.0,
        }],
        ..Input::default()
    };

    stage.boundary(press.clone());
    assert!(stage.session.dragging().is_none() && !stage.held());

    stage.session.app.carriable.set(true);
    stage.boundary(press);
    let carried = stage.session.dragging().map(|drag| drag.entity);
    assert_eq!(carried, Some(stage.body));
    assert!(stage.held());

    stage.session.app.carriable.set(false);
    stage.boundary(Input::default());
    assert!(stage.session.dragging().is_none() && !stage.held());
}
