use loam_runtime::{
    Input, Overlay, Phase, PointRecord, Publication, SegmentRecord, Session, SimConfig, TextRecord,
};
use loam_time::par::Sequential;

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Probe {
        shown: Value<bool>,
        publication_alphas: Value<Vec<f32>>,
        simulation_alphas: Value<Vec<f32>>,
    }
}

const GUIDE: SegmentRecord = SegmentRecord {
    start: [-1.0, 0.0, 0.0],
    _pad0: 0.0,
    end: [1.0, 0.0, 0.0],
    _pad1: 0.0,
    start_color: [1.0; 4],
    end_color: [1.0; 4],
    width_px: 2.0,
    _pad2: [0.0; 3],
};

const HANDLE: PointRecord = PointRecord {
    position: [0.0, 1.0, 0.0],
    radius_px: 6.0,
    color: [1.0; 4],
};

fn label(x: f32) -> TextRecord {
    TextRecord {
        origin_pt: [x, 16.0],
        size_pt: 16.0,
        color: [1.0; 4],
    }
}

#[test]
fn phase_and_frame() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    session.system(Phase::Dispatch, "stray", |ctx| {
        ctx.overlay.lines.push(SegmentRecord::default());
        Ok(())
    });
    session.system(Phase::Publication, "marks", |ctx| {
        if *ctx.app.shown.get() {
            ctx.overlay.lines.push(GUIDE);
            ctx.overlay.points.push(HANDLE);
            ctx.overlay.push_text("slice", label(1.0));
            ctx.overlay.push_text("", label(2.0));
            ctx.overlay.push_text("rate", label(3.0));
        }
        Ok(())
    });
    session.app.shown.set(true);
    let mut publication = Publication::default();

    session.boundary(Input::default()).unwrap();
    session.publish(&mut publication, 1.0).unwrap();
    assert_eq!(publication.overlay.lines, [GUIDE]);
    assert_eq!(publication.overlay.points, [HANDLE]);
    let text: Vec<(&str, TextRecord)> = publication
        .overlay
        .text()
        .map(|(text, record)| (text, *record))
        .collect();
    assert_eq!(
        text,
        [
            ("slice", label(1.0)),
            ("", label(2.0)),
            ("rate", label(3.0))
        ]
    );
    let first = publication.overlay.clone();
    let drawn = publication.content();

    session.boundary(Input::default()).unwrap();
    session.publish(&mut publication, 1.0).unwrap();
    assert_eq!(
        publication.overlay, first,
        "a second frame grew the overlay"
    );
    assert_eq!(
        publication.content(),
        drawn,
        "an unchanged overlay asked for a redraw"
    );

    session.app.shown.set(false);
    session.boundary(Input::default()).unwrap();
    session.publish(&mut publication, 1.0).unwrap();
    assert_eq!(
        publication.overlay,
        Overlay::default(),
        "last frame's overlay outlived its frame"
    );
    assert_ne!(
        publication.content(),
        drawn,
        "the overlay emptied without asking for a redraw"
    );
}

#[test]
fn alpha_by_phase() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    session.system(Phase::Publication, "publication alpha", |ctx| {
        let alpha = ctx.alpha;
        ctx.app.publication_alphas.get_mut().push(alpha);
        Ok(())
    });
    session.system(Phase::Simulation, "simulation alpha", |ctx| {
        let alpha = ctx.alpha;
        ctx.app.simulation_alphas.get_mut().push(alpha);
        Ok(())
    });
    let mut publication = Publication::default();

    for alpha in [0.25, 2.5, -1.0, f32::NAN] {
        session.publish(&mut publication, alpha).unwrap();
    }
    session.tick(&Sequential).unwrap();

    assert_eq!(*session.app.publication_alphas.get(), [0.25, 1.0, 0.0, 1.0]);
    assert_eq!(*session.app.simulation_alphas.get(), [1.0]);
}
