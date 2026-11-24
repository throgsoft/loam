use std::fmt;
use std::num::NonZeroU32;
use std::ops::Range;

use loam_runtime::{
    ActionEvent, ActionId, DomainError, Input, PhaseError, Publication, Rejection, RestoreError,
    Session, Stores,
};
use loam_time::par::Sequential;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    Space,
    Escape,
    Letter(char),
    Digit(u8),
}

#[derive(Clone, Debug, Default)]
pub struct Bindings {
    keys: Vec<(Key, ActionId)>,
}

impl Bindings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn key(mut self, key: Key, action: ActionId) -> Self {
        self.keys.push((key, action));
        self
    }

    pub fn action(&self, key: Key) -> Option<ActionId> {
        self.keys
            .iter()
            .find(|(bound, _)| *bound == key)
            .map(|(_, action)| *action)
    }
}

pub struct HostConfig {
    pub title: &'static str,
    pub bindings: Bindings,
}

impl HostConfig {
    pub fn new(title: &'static str, bindings: Bindings) -> Self {
        Self { title, bindings }
    }
}

#[derive(Debug)]
pub enum HostError {
    Setup(Rejection),
    Phase(PhaseError),
    Host(String),
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Setup(rejection) => fmt::Display::fmt(rejection, f),
            Self::Phase(error) => fmt::Display::fmt(error, f),
            Self::Host(message) => f.write_str(message),
        }
    }
}

impl From<Rejection> for HostError {
    fn from(rejection: Rejection) -> Self {
        Self::Setup(rejection)
    }
}

impl From<DomainError> for HostError {
    fn from(error: DomainError) -> Self {
        Self::Setup(Rejection::Domain(error))
    }
}

impl From<RestoreError> for HostError {
    fn from(error: RestoreError) -> Self {
        Self::Setup(Rejection::Restore(error))
    }
}

impl From<PhaseError> for HostError {
    fn from(error: PhaseError) -> Self {
        Self::Phase(error)
    }
}

/// Calls `each` after every publish; commands it submits apply at the next step's boundary.
pub fn run_headless<A: Stores>(
    session: &mut Session<A>,
    config: &HostConfig,
    steps: u32,
    holds: &[(Key, Range<u32>)],
    alpha: f32,
    mut each: impl FnMut(u32, &mut Session<A>, &Publication<A::Published>) -> Result<(), HostError>,
) -> Result<Publication<A::Published>, HostError> {
    session.capture_initial();
    let mut publication = Publication::default();
    let mut held: Vec<ActionId> = Vec::new();
    for step in 0..steps {
        let now: Vec<ActionId> = holds
            .iter()
            .filter(|(_, range)| range.contains(&step))
            .filter_map(|(key, _)| config.bindings.action(*key))
            .collect();
        let edges = |from: &[ActionId], to: &[ActionId], pressed: bool| {
            from.iter()
                .filter(|action| !to.contains(action))
                .map(|&action| ActionEvent { action, pressed })
                .collect::<Vec<_>>()
        };
        let mut actions = edges(&now, &held, true);
        actions.extend(edges(&held, &now, false));
        held = now;
        session.boundary(Input {
            actions,
            held: held.clone(),
            ..Input::default()
        })?;
        session.tick(&Sequential)?;
        session.publish(&mut publication, alpha)?;
        each(step, session, &publication)?;
    }
    Ok(publication)
}

/// Writes every `stride`th step's records and root eye to `out`; `each` gets `out` only on those steps, after them.
pub fn write_headless<A: Stores, W: fmt::Write>(
    out: &mut W,
    session: &mut Session<A>,
    config: &HostConfig,
    steps: u32,
    alpha: f32,
    stride: NonZeroU32,
    mut each: impl FnMut(u32, &mut Session<A>, Option<&mut W>) -> Result<(), HostError>,
) -> Result<(), HostError> {
    run_headless(
        session,
        config,
        steps,
        &[],
        alpha,
        |step, session, publication| {
            if write_stride(out, step, stride, session, publication)? {
                each(step, session, Some(&mut *out))
            } else {
                each(step, session, None)
            }
        },
    )?;
    Ok(())
}

/// Writes `step`'s records and root eye when `step` ends a `stride`, and says whether it wrote.
pub fn write_stride<A: Stores>(
    out: &mut impl fmt::Write,
    step: u32,
    stride: NonZeroU32,
    session: &Session<A>,
    publication: &Publication<A::Published>,
) -> Result<bool, HostError> {
    if !(step + 1).is_multiple_of(stride.get()) {
        return Ok(false);
    }
    write_records(out, step, publication)
        .and_then(|()| write_eye(out, session))
        .map_err(|_| HostError::Host("the record dump failed".into()))?;
    Ok(true)
}

fn write_eye<A: Stores>(out: &mut impl fmt::Write, session: &Session<A>) -> fmt::Result {
    let eye = session.views().root().eye;
    writeln!(
        out,
        "eye position {:?} right {:?} up {:?} forward {:?} fov_y {:?} aspect {:?} near {:?} far {:?}",
        eye.position, eye.right, eye.up, eye.forward, eye.fov_y, eye.aspect, eye.near, eye.far
    )
}

/// Writes one line per published record with every float in `{:?}`, so equal text means equal bits.
pub fn write_records<V>(
    out: &mut impl fmt::Write,
    step: u32,
    publication: &Publication<V>,
) -> fmt::Result {
    writeln!(
        out,
        "publication step {step} tick {} alpha {:?}",
        publication.stamp.tick.0, publication.alpha
    )?;
    for view in &publication.views {
        writeln!(
            out,
            "view {} {} {}",
            view.domain, view.target.view, view.target.image
        )?;
        for instance in view.records.instances.rows() {
            writeln!(
                out,
                "instance {} geometry {} material {} chart {} coordinates {:?} frame {:?} image {:?}",
                instance.entity,
                instance.geometry.index(),
                instance.material.index(),
                instance.pose.chart.0,
                instance.pose.coordinates,
                instance.pose.frame,
                instance.image_point
            )?;
        }
        for segment in view.records.segments() {
            writeln!(
                out,
                "segment {:?} {:?} colors {:?} {:?} width {:?}",
                segment.start,
                segment.end,
                segment.start_color,
                segment.end_color,
                segment.width_px
            )?;
        }
        for triangle in view.records.triangles() {
            writeln!(
                out,
                "triangle {:?} color {:?}",
                triangle.vertices, triangle.color
            )?;
        }
        let fields = view.records.fields();
        if let Some(cut) = view.records.field_cut().filter(|_| !fields.is_empty()) {
            writeln!(out, "fields cut {cut:?}")?;
        }
        for field in fields {
            writeln!(
                out,
                "field {} shape {:?} size {:?} material {} color {:?} position {:?} frame {:?}",
                field.entity,
                field.shape,
                field.size,
                field.material.index(),
                field.color,
                field.position,
                <[f32; 8]>::from(field.frame)
            )?;
        }
        let refusals = view.records.refusals();
        for refusal in [refusals.first, refusals.last].into_iter().flatten() {
            writeln!(
                out,
                "refusal of {} {} {:?} from {:?}",
                refusals.count, refusal.entity, refusal.error, refusal.source
            )?;
        }
    }
    for line in &publication.overlay.lines {
        writeln!(
            out,
            "overlay line {:?} {:?} colors {:?} {:?} width {:?}",
            line.start, line.end, line.start_color, line.end_color, line.width_px
        )?;
    }
    for point in &publication.overlay.points {
        writeln!(
            out,
            "overlay point {:?} radius {:?} color {:?}",
            point.position, point.radius_px, point.color
        )?;
    }
    for (text, record) in publication.overlay.text() {
        writeln!(
            out,
            "overlay text {text:?} at {:?} size {:?} color {:?}",
            record.origin_pt, record.size_pt, record.color
        )?;
    }
    Ok(())
}
