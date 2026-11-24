use std::fmt;
use std::num::NonZeroU64;

use loam_runtime::{Publication, Session, Stores};

use super::app::SessionApp;
use super::headless::run_script;
use crate::host::HostError;
use crate::script::{driver_from_args, Script, ScriptDriver};

// Runs `session` as `--headless [frames]`, `--records` and `--script=path` ask and prints the record dump to stdout.
pub(crate) fn run<A: Stores>(
    session: &mut Session<A>,
    app: &mut SessionApp<A>,
) -> Result<(), HostError> {
    if !app.args.has_bare_flag("records") {
        return Err(HostError::Host("a headless run needs --records".into()));
    }
    let script =
        driver_from_args(&app.args).map_err(|error| HostError::Host(format!("{error:#}")))?;
    let mut driver = script.unwrap_or_else(|| ScriptDriver::new(Script::default()));
    if let Some(raw) = app.args.flag_value("headless") {
        let frames = raw.parse::<NonZeroU64>().map_err(|_| {
            HostError::Host(format!("--headless {raw:?} is not a positive frame count"))
        })?;
        driver = driver.with_frames(frames);
    }
    let stride = app.record_stride;
    let mut extra = app.record.take();
    let mut dump = String::new();
    run_script(session, app, driver, 1.0, |step, session, publication| {
        if !(step + 1).is_multiple_of(stride.get()) {
            return Ok(());
        }
        write_records(&mut dump, step, publication)
            .and_then(|()| write_eye(&mut dump, session))
            .and_then(|()| {
                extra
                    .as_mut()
                    .map_or(Ok(()), |extra| extra(session, &mut dump))
            })
            .map_err(|_| HostError::Host("the record dump failed".into()))
    })?;
    print!("{dump}");
    Ok(())
}

fn write_eye<A: Stores>(out: &mut impl fmt::Write, session: &Session<A>) -> fmt::Result {
    let eye = session.views().root().eye;
    writeln!(
        out,
        "eye position {:?} right {:?} up {:?} forward {:?} fov_y {:?} aspect {:?} near {:?} far {:?}",
        eye.position, eye.right, eye.up, eye.forward, eye.fov_y, eye.aspect, eye.near, eye.far
    )
}

// Every float prints in `{:?}`, so equal text means equal bits.
fn write_records<V>(
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
