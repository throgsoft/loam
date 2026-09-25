use loam_console::{cmd, Console, ConsoleWriter};
use loam_render::pass::{GpuTime, Section};
use loam_time::frame_trace;
use std::sync::Mutex;
use std::time::Duration;

#[cfg(feature = "egui")]
mod overlay;
#[cfg(feature = "egui")]
pub use overlay::{PerfOverlay, MAX_WINDOW};

fn fmt_dur(d: std::time::Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{ns}ns")
    } else if ns < 1_000_000 {
        format!("{:.1}us", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.2}ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", ns as f64 / 1_000_000_000.0)
    }
}

// Saturating: parent and children come from independent `Instant` reads.
fn unscoped(frame: &frame_trace::FrameTrace) -> Option<Duration> {
    let mut total = None;
    let mut covered = Duration::ZERO;
    for section in &frame.sections {
        if section.name == "frame" {
            total = Some(section.elapsed);
        } else if crate::FRAME_LOOP_SECTIONS.contains(&section.name) {
            covered += section.elapsed;
        }
    }
    total.map(|t| t.saturating_sub(covered))
}

fn unscoped_stats() -> Option<frame_trace::SectionStats> {
    let mut samples: Vec<Duration> =
        frame_trace::with_history(|history| history.iter().filter_map(unscoped).collect());
    if samples.is_empty() {
        return None;
    }
    samples.sort();
    let n = samples.len();
    Some(frame_trace::SectionStats {
        name: "unscoped",
        samples: n,
        mean: samples.iter().sum::<Duration>() / n as u32,
        p50: frame_trace::percentile(&samples, 50),
        p95: frame_trace::percentile(&samples, 95),
        p99: frame_trace::percentile(&samples, 99),
        max: samples[n - 1],
    })
}

fn summary_rows() -> Vec<frame_trace::SectionStats> {
    let mut stats = frame_trace::aggregate();
    if let Some(residual) = unscoped_stats() {
        stats.push(residual);
        stats.sort_by_key(|s| std::cmp::Reverse(s.p95));
    }
    stats
}

fn print_summary(out: &mut ConsoleWriter) {
    let stats = summary_rows();
    if stats.is_empty() {
        out.line("trace: no frames in window (collect runs once the demo is rendering)");
        return;
    }
    let history_len = frame_trace::with_history(|frames| frames.len());
    out.line(format!(
        "trace summary ({history_len} frames, sorted by p95 desc):"
    ));
    out.line(format!(
        "  {:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "section", "n", "mean", "p50", "p95", "p99", "max",
    ));
    for s in stats {
        out.line(format!(
            "  {:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}",
            truncate(s.name, 18),
            s.samples,
            fmt_dur(s.mean),
            fmt_dur(s.p50),
            fmt_dur(s.p95),
            fmt_dur(s.p99),
            fmt_dur(s.max),
        ));
    }
}

#[derive(Default)]
struct Presentation {
    sections: Vec<Section>,
    uploads: u64,
}

static PRESENTATION: Mutex<Option<Presentation>> = Mutex::new(None);

fn presentation<R>(read: impl FnOnce(&mut Option<Presentation>) -> R) -> R {
    let mut held = match PRESENTATION.lock() {
        Ok(held) => held,
        Err(poisoned) => poisoned.into_inner(),
    };
    read(&mut held)
}

pub(crate) fn record_presentation(sections: &[Section], uploads: u64) {
    presentation(|held| {
        let held = held.get_or_insert_with(Presentation::default);
        held.sections.clear();
        held.sections.extend_from_slice(sections);
        held.uploads = uploads;
    });
}

fn print_passes(out: &mut ConsoleWriter) {
    presentation(|held| {
        let Some(held) = held.as_ref() else {
            out.line("trace: the presenter has not recorded a frame yet");
            return;
        };
        out.line(format!(
            "trace passes ({} sections, {} record uploads):",
            held.sections.len(),
            held.uploads,
        ));
        for section in &held.sections {
            let gpu = match section.gpu {
                GpuTime::Measured(elapsed) => fmt_dur(elapsed),
                GpuTime::Unavailable => "unavailable".to_owned(),
            };
            out.line(format!(
                "  {:<18} cpu {:>10} gpu {:>12}",
                truncate(section.name, 18),
                fmt_dur(section.cpu),
                gpu,
            ));
        }
    });
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_owned()
    } else if max == 0 {
        String::new()
    } else {
        s.chars()
            .take(max - 1)
            .chain(std::iter::once('~'))
            .collect()
    }
}

fn print_last(out: &mut ConsoleWriter) {
    let Some(frame) = frame_trace::last_frame() else {
        out.line("trace: no frames in window yet");
        return;
    };
    let total = frame.total();
    out.line(format!(
        "trace last-frame ({} sections, sum {}):",
        frame.sections.len(),
        fmt_dur(total),
    ));
    for section in &frame.sections {
        let pct = if !total.is_zero() {
            (section.elapsed.as_nanos() as f64 * 100.0) / total.as_nanos() as f64
        } else {
            0.0
        };
        out.line(format!(
            "  {:<18} {:>10} ({:>4.1}%)",
            truncate(section.name, 18),
            fmt_dur(section.elapsed),
            pct,
        ));
    }
}

fn format_summary() -> String {
    let mut out = ConsoleWriter::new();
    print_summary(&mut out);
    out.take_lines()
        .into_iter()
        .map(|line| line.text)
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn register_command<Ctx: 'static>(console: &mut Console<Ctx>) {
    console.register(
        cmd(
            "trace",
            "show CPU per-section frame timings (collected by loam-time::frame_trace)",
            |args, _ctx: &mut Ctx, out| {
                match args.first().copied() {
                    None | Some("summary") => {
                        print_summary(out);
                        print_passes(out);
                    }
                    Some("last") => print_last(out),
                    Some("passes") => print_passes(out),
                    Some("dump") => {
                        let summary = format_summary();
                        tracing::info!("\n{summary}");
                        out.line("trace: dumped to browser console (open DevTools to copy)");
                    }
                    Some("clear") => {
                        frame_trace::clear_history();
                        out.line("trace: history cleared");
                    }
                    Some("cap") => {
                        let n = args
                            .get(1)
                            .copied()
                            .and_then(|s| s.parse::<usize>().ok());
                        match n {
                            Some(n) if n >= 1 => {
                                frame_trace::set_capacity(n);
                                out.line(format!("trace: capacity set to {n} frames"));
                            }
                            _ => {
                                out.line("usage: trace cap <N>  (N >= 1)");
                            }
                        }
                    }
                    Some(other) => {
                        out.line(format!(
                            "trace: unknown subcommand '{other}' (try summary | last | passes | dump | clear | cap)"
                        ));
                    }
                }
                Ok(())
            },
        )
        .with_args(&[&["summary", "last", "passes", "dump", "clear", "cap"]]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_preserves_short_names_and_marks_long_ones() {
        assert_eq!(truncate("frame", 18), "frame");
        assert_eq!(truncate("abcdefghijklmnopqr", 18).len(), 18);
        let long = "supercalifragilisticexpialidocious";
        let t = truncate(long, 18);
        assert_eq!(t.len(), 18);
        assert!(t.ends_with('~'));
        assert_eq!(truncate("αβγδε", 3), "αβ~");
        assert_eq!(truncate("frame", 0), "");
    }

    fn frame_of(sections: &[(&'static str, u64)]) -> frame_trace::FrameTrace {
        frame_trace::FrameTrace {
            sections: sections
                .iter()
                .map(|(name, us)| frame_trace::Section {
                    name,
                    elapsed: Duration::from_micros(*us),
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn unscoped_ignores_sections_nested_inside_frame_loop_sections() {
        let flat = frame_of(&[("frame", 4000), ("presentation", 130)]);
        let nested = frame_of(&[("frame", 4000), ("presentation", 130), ("pp-sdf", 94)]);
        assert_eq!(unscoped(&nested), unscoped(&flat));
    }

    #[test]
    fn unscoped_ignores_the_sections_that_do_not_nest_in_the_frame() {
        let frame = frame_of(&[
            ("frame", 4000),
            ("presentation", 130),
            ("between-frames", 4130),
            ("idle", 110),
            ("gpu-total", 800),
        ]);
        assert_eq!(unscoped(&frame), Some(Duration::from_micros(3870)));
    }

    #[test]
    fn unscoped_is_absent_for_a_frame_without_a_frame_section() {
        let frame = frame_of(&[("presentation", 130), ("present", 40)]);
        assert_eq!(unscoped(&frame), None);
    }

    #[test]
    fn unscoped_saturates_at_zero_when_the_children_overrun_the_parent() {
        let frame = frame_of(&[("frame", 100), ("presentation", 130)]);
        assert_eq!(unscoped(&frame), Some(Duration::ZERO));
    }
}
