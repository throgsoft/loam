use glam::{Vec3, Vec4};
use loam::app::session::{launch_with, look, FrameHook, SessionApp};
use loam::app::{LaunchMode, WasmConfig};
use loam::math::{Bivector, Bivector4, EuclideanR4, Rotor4};
use loam::render::sky_ground::linear_from_srgb8;
use loam::render::view::root_view_projection;
use loam::runtime::host::{HostConfig, HostError};
use loam::runtime::{
    Bindings, DomainBuilder, DomainError, Eye, Instance, Material, Outcome, Phase, Pose,
    PreparedGeometry, Rejection, Section4, Session, SimConfig, SpawnBundle, ViewSpec,
};
use loam::shape::polytope::Polytope4;

const TURN_RATE: f32 = 0.25;
const POLYTOPE_SCALE: f32 = 1.2;
const CAMERA_DISTANCE: f32 = 6.0;
const CAMERA_TOP: f32 = 2.5;
const CAMERA_DROP: f32 = 5.0;
const PAGE_RGB: [u8; 3] = [0xf3, 0xf2, 0xef];
const FACE_COLOR: [f32; 4] = [0.16, 0.16, 0.2, 1.0];
const CUT_COLOR: [f32; 4] = [0.04, 0.04, 0.06, 1.0];

loam::runtime::stores! {
    #[derive(Default)]
    pub struct BackdropStores {
        scroll: Value<f32>,
    }
}

fn build(
    args: loam::app::args::Args,
) -> Result<(Session<BackdropStores>, SessionApp<BackdropStores>), HostError> {
    let mut session = Session::new(BackdropStores::default(), SimConfig::default());
    let r4 = session.register_domain(DomainBuilder::new("r4", EuclideanR4));
    let tesseract = session.prepare(PreparedGeometry::Polytope4 {
        polytope: Polytope4::Tesseract,
        scale: POLYTOPE_SCALE,
    });
    let face = session.add_material(Material::flat(FACE_COLOR));
    let cut = session.add_material(Material::lines(CUT_COLOR, 1.5));
    let root = session.views().root();

    let body = session.dispatch(|d| -> Result<_, Rejection> {
        let body = d.spawn(
            SpawnBundle::new()
                .at(r4, Pose::at(Vec4::ZERO))
                .instance(Instance::new(tesseract, face).sectioned(cut)),
        )?;
        let eye = d.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))?;
        let mut spec = ViewSpec::new(root, eye, Section4 { w: 0.0 });
        spec.edges = false;
        d.domains.typed(r4)?.add_view(spec)?;
        Ok(body)
    })?;

    session.system(
        Phase::Dispatch,
        "page",
        move |ctx: loam::runtime::Ctx<'_, BackdropStores>| {
            let host = &ctx.input.host;
            if let Some(scroll) = host
                .latest("scroll")
                .and_then(|values| values.first())
                .filter(|value| value.is_finite())
            {
                ctx.app.scroll.set(scroll.clamp(0.0, 1.0));
            }
            if host.iter().any(|(topic, _)| topic == "reset") {
                ctx.commands.try_app_fn("reset", move |d| {
                    d.domains.typed(r4)?.set_frame(body, Rotor4::IDENTITY)?;
                    Ok(Outcome::Done)
                });
            }
            let height = CAMERA_TOP - ctx.app.scroll.get() * CAMERA_DROP;
            look(
                ctx.views,
                Eye::looking_at([0.0, height, CAMERA_DISTANCE], [0.0; 3], [0.0, 1.0, 0.0]),
            );
            Ok(())
        },
    );

    session.system(
        Phase::Simulation,
        "turn",
        move |ctx: loam::runtime::Ctx<'_, BackdropStores>| -> Result<(), DomainError> {
            let r4 = ctx.domains.typed(r4)?;
            if let Some(pose) = r4.poses().get(body) {
                let step = (Bivector4::basis(2) * (TURN_RATE * ctx.step.dt)).exp();
                r4.set_frame(body, (step * pose.frame).normalize())?;
            }
            Ok(())
        },
    );

    session.set_initial()?;
    let mut posted: Option<[f32; 4]> = None;
    let app = SessionApp::with_args(HostConfig::new("backdrop", Bindings::new()), args)
        .debug_layer(false)
        .background(linear_from_srgb8(PAGE_RGB))
        .on_frame(move |hook: &mut FrameHook<'_, BackdropStores>| {
            if let Some(rect) = screen_rect(hook) {
                if posted != Some(rect) {
                    posted = Some(rect);
                    hook.post("rect", &rect);
                }
            }
        });
    Ok((session, app))
}

fn screen_rect(hook: &FrameHook<'_, BackdropStores>) -> Option<[f32; 4]> {
    let root = hook.session.views().root();
    let clip = root_view_projection(&hook.session.views().get(root)?.eye);
    let mut rect = [1.0_f32, 1.0, 0.0, 0.0];
    let mut seen = false;
    for view in &hook.published.views {
        for triangle in view.records.triangles() {
            for corner in triangle.vertices {
                let point = clip * Vec3::from(view.placement.apply(corner)).extend(1.0);
                if point.w <= 0.0 {
                    continue;
                }
                let x = (0.5 + 0.5 * point.x / point.w).clamp(0.0, 1.0);
                let y = (0.5 - 0.5 * point.y / point.w).clamp(0.0, 1.0);
                rect = [
                    rect[0].min(x),
                    rect[1].min(y),
                    rect[2].max(x),
                    rect[3].max(y),
                ];
                seen = true;
            }
        }
    }
    seen.then_some(rect)
}

fn main() -> Result<(), HostError> {
    launch_with(
        WasmConfig {
            mode: LaunchMode::Background,
            max_pixels: Some(1920 * 1080),
            ..Default::default()
        },
        build,
    )
}
