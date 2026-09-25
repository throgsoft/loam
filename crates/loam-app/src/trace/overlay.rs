use std::time::Duration;

use loam_time::frame_trace;

pub struct PerfOverlay {
    visible: bool,
    toggle_key: loam_egui::egui::Key,
    window: usize,
}

impl Default for PerfOverlay {
    fn default() -> Self {
        Self {
            visible: false,
            toggle_key: loam_egui::egui::Key::F3,
            window: 60,
        }
    }
}

const OVERLAY_WIDTH: f32 = 260.0;

const OVERLAY_MARGIN: f32 = 12.0;

// `content_rect` ignores panels and would seat the readout behind the menu bar.
fn perf_overlay_seat(ctx: &loam_egui::egui::Context) -> loam_egui::egui::Pos2 {
    let area = ctx.available_rect();
    loam_egui::egui::pos2(
        (area.right() - OVERLAY_MARGIN - OVERLAY_WIDTH).max(area.left()),
        area.top() + OVERLAY_MARGIN,
    )
}

impl PerfOverlay {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_toggle_key(mut self, key: loam_egui::egui::Key) -> Self {
        self.toggle_key = key;
        self
    }

    pub fn always_visible(mut self) -> Self {
        self.visible = true;
        self
    }

    pub fn show(&mut self, ctx: &loam_egui::egui::Context) {
        use loam_egui::egui;

        if ctx.input(|i| i.key_pressed(self.toggle_key)) {
            self.visible = !self.visible;
        }
        if !self.visible {
            return;
        }

        let window = self.window.min(MAX_WINDOW);
        let mut cadence = StackBuf::new();
        let mut frames_buf = StackBuf::new();
        let mut idles = StackBuf::new();
        let mut heap_count = 0usize;
        let mut heap_peak = 0i64;
        let mut heap_net = 0i64;
        let mut alloc_frames = 0usize;
        let mut alloc_count_sum: u64 = 0;
        let mut alloc_peak_bytes: u64 = 0;
        let mut alloc_net_bytes: i64 = 0;
        let mut any = false;

        frame_trace::with_history(|history| {
            let start = history.len().saturating_sub(window);
            for frame in history.iter().skip(start) {
                any = true;
                for section in &frame.sections {
                    match section.name {
                        "between-frames" => cadence.push(section.elapsed),
                        "frame" => frames_buf.push(section.elapsed),
                        "idle" => idles.push(section.elapsed),
                        _ => {}
                    }
                }
                if let Some(d) = frame.heap_delta_bytes {
                    heap_count += 1;
                    if d > heap_peak {
                        heap_peak = d;
                    }
                    heap_net = heap_net.saturating_add(d);
                }
                if let Some(a) = frame.allocs {
                    alloc_frames += 1;
                    alloc_count_sum = alloc_count_sum.saturating_add(a.alloc_count);
                    if a.alloc_bytes > alloc_peak_bytes {
                        alloc_peak_bytes = a.alloc_bytes;
                    }
                    alloc_net_bytes = alloc_net_bytes.saturating_add(a.net_bytes);
                }
            }
        });

        if !any {
            return;
        }

        let cadence_mean = cadence.mean();
        let cadence_p99 = cadence.percentile(99);
        let frame_mean = frames_buf.mean();
        let frame_p99 = frames_buf.percentile(99);
        let idle_mean = idles.mean();
        let idle_p99 = idles.percentile(99);

        let cadence_max_ever = frame_trace::max_ever("between-frames");
        let idle_max_ever = frame_trace::max_ever("idle");
        let frame_max_ever = frame_trace::max_ever("frame");

        let fps = if cadence_mean.as_secs_f32() > 0.0 {
            1.0 / cadence_mean.as_secs_f32()
        } else {
            0.0
        };

        // `anchor` ignores drag; `default_pos` lets egui persist the dragged offset.
        egui::Area::new(egui::Id::new("loam-perf-overlay"))
            .default_pos(perf_overlay_seat(ctx))
            .movable(true)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style())
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 60, 75)))
                    .show(ui, |ui| {
                        ui.set_min_width(OVERLAY_WIDTH);
                        let mono = egui::FontId::monospace(11.0);
                        let label_color = egui::Color32::from_rgb(180, 190, 200);
                        ui.label(
                            egui::RichText::new(format!("FPS    {fps:5.1}"))
                                .font(mono.clone())
                                .color(egui::Color32::from_rgb(220, 230, 240)),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "total  {:>5.1}  p99 {:>5.1}  ms",
                                cadence_mean.as_secs_f32() * 1000.0,
                                cadence_p99.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(label_color),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "idle   {:>5.1}  p99 {:>5.1}  ms",
                                idle_mean.as_secs_f32() * 1000.0,
                                idle_p99.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(label_color),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "frame  {:>5.2}  p99 {:>5.2}  ms",
                                frame_mean.as_secs_f32() * 1000.0,
                                frame_p99.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(label_color),
                        );
                        ui.separator();
                        ui.label(
                            egui::RichText::new("worst-ever (session)")
                                .font(mono.clone())
                                .color(egui::Color32::from_rgb(140, 150, 160)),
                        );
                        let worst_color = |d: Duration| {
                            let ms = d.as_secs_f32() * 1000.0;
                            if ms >= 100.0 {
                                egui::Color32::from_rgb(220, 100, 80)
                            } else if ms >= 50.0 {
                                egui::Color32::from_rgb(220, 180, 90)
                            } else {
                                label_color
                            }
                        };
                        ui.label(
                            egui::RichText::new(format!(
                                "total  {:>6.1} ms",
                                cadence_max_ever.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(worst_color(cadence_max_ever)),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "idle   {:>6.1} ms",
                                idle_max_ever.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(worst_color(idle_max_ever)),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "frame  {:>6.1} ms",
                                frame_max_ever.as_secs_f32() * 1000.0,
                            ))
                            .font(mono.clone())
                            .color(worst_color(frame_max_ever)),
                        );
                        if alloc_frames > 0 {
                            ui.separator();
                            ui.label(
                                egui::RichText::new("allocs (Rust heap)")
                                    .font(mono.clone())
                                    .color(egui::Color32::from_rgb(140, 150, 160)),
                            );
                            let mean_count = alloc_count_sum / alloc_frames as u64;
                            let count_color = |n: u64| {
                                if n >= 1_000 {
                                    egui::Color32::from_rgb(220, 100, 80)
                                } else if n >= 100 {
                                    egui::Color32::from_rgb(220, 180, 90)
                                } else if n >= 10 {
                                    egui::Color32::from_rgb(180, 200, 130)
                                } else {
                                    egui::Color32::from_rgb(120, 200, 130)
                                }
                            };
                            let byte_color = |bytes: i64| {
                                let mb = bytes.abs() as f32 / (1024.0 * 1024.0);
                                if mb >= 10.0 {
                                    egui::Color32::from_rgb(220, 100, 80)
                                } else if mb >= 1.0 {
                                    egui::Color32::from_rgb(220, 180, 90)
                                } else {
                                    label_color
                                }
                            };
                            ui.label(
                                egui::RichText::new(format!("mean  {mean_count:>6} allocs/frame",))
                                    .font(mono.clone())
                                    .color(count_color(mean_count)),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "peak  {:>+6.2} KB/frame",
                                    alloc_peak_bytes as f32 / 1024.0,
                                ))
                                .font(mono.clone())
                                .color(byte_color(alloc_peak_bytes as i64)),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "net   {:>+6.2} MB / window",
                                    alloc_net_bytes as f32 / (1024.0 * 1024.0),
                                ))
                                .font(mono.clone())
                                .color(byte_color(alloc_net_bytes)),
                            );
                        }
                        if heap_count > 0 {
                            ui.separator();
                            ui.label(
                                egui::RichText::new("heap (Chromium)")
                                    .font(mono.clone())
                                    .color(egui::Color32::from_rgb(140, 150, 160)),
                            );
                            let heap_color = |bytes: i64| {
                                let mb = bytes.abs() as f32 / (1024.0 * 1024.0);
                                if mb >= 10.0 {
                                    egui::Color32::from_rgb(220, 100, 80)
                                } else if mb >= 2.0 {
                                    egui::Color32::from_rgb(220, 180, 90)
                                } else {
                                    label_color
                                }
                            };
                            ui.label(
                                egui::RichText::new(format!(
                                    "peak  {:>+6.2} MB/frame",
                                    heap_peak as f32 / (1024.0 * 1024.0),
                                ))
                                .font(mono.clone())
                                .color(heap_color(heap_peak)),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "net   {:>+6.2} MB / window",
                                    heap_net as f32 / (1024.0 * 1024.0),
                                ))
                                .font(mono.clone())
                                .color(heap_color(heap_net)),
                            );
                        }
                        ui.separator();
                        draw_sparkline(ui, cadence.as_slice());
                    });
            });
    }
}

pub const MAX_WINDOW: usize = 256;

#[derive(Clone)]
struct StackBuf {
    samples: [Duration; MAX_WINDOW],
    len: usize,
}

impl StackBuf {
    fn new() -> Self {
        Self {
            samples: [Duration::ZERO; MAX_WINDOW],
            len: 0,
        }
    }

    fn push(&mut self, d: Duration) {
        if self.len < MAX_WINDOW {
            self.samples[self.len] = d;
            self.len += 1;
        }
    }

    fn as_slice(&self) -> &[Duration] {
        &self.samples[..self.len]
    }

    fn mean(&self) -> Duration {
        if self.len == 0 {
            return Duration::ZERO;
        }
        let sum: Duration = self.samples[..self.len].iter().sum();
        sum / self.len as u32
    }

    // Sorts a copy: the sparkline reads `self` in time order.
    fn percentile(&self, percent: u8) -> Duration {
        if self.len == 0 {
            return Duration::ZERO;
        }
        let mut local: [Duration; MAX_WINDOW] = self.samples;
        local[..self.len].sort_unstable();
        frame_trace::percentile(&local[..self.len], percent)
    }
}

fn draw_sparkline(ui: &mut loam_egui::egui::Ui, gaps: &[Duration]) {
    use loam_egui::egui;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(240.0, 36.0), egui::Sense::hover());
    if gaps.is_empty() {
        return;
    }
    let painter = ui.painter();
    painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(18, 18, 24));

    // Outliers clamp to the top so one spike does not squash the baseline.
    let y_max_ms = 50.0_f32;
    let y_for_ms = |ms: f32| {
        let clamped = ms.min(y_max_ms);
        rect.bottom() - (clamped / y_max_ms) * rect.height()
    };
    let ref_60 = y_for_ms(16.67);
    let ref_30 = y_for_ms(33.33);
    painter.line_segment(
        [
            egui::pos2(rect.left(), ref_60),
            egui::pos2(rect.right(), ref_60),
        ],
        egui::Stroke::new(0.5, egui::Color32::from_rgb(60, 100, 70)),
    );
    painter.line_segment(
        [
            egui::pos2(rect.left(), ref_30),
            egui::pos2(rect.right(), ref_30),
        ],
        egui::Stroke::new(0.5, egui::Color32::from_rgb(120, 90, 60)),
    );

    let n = gaps.len() as f32;
    let dx = rect.width() / n.max(1.0);
    let bar_w = dx.max(1.0);
    for (i, gap) in gaps.iter().enumerate() {
        let x = rect.left() + i as f32 * dx;
        let ms = gap.as_secs_f32() * 1000.0;
        let y = y_for_ms(ms);
        let color = if ms > 33.3 {
            egui::Color32::from_rgb(220, 100, 80)
        } else if ms > 20.0 {
            egui::Color32::from_rgb(220, 180, 90)
        } else {
            egui::Color32::from_rgb(120, 200, 130)
        };
        painter.line_segment(
            [egui::pos2(x, rect.bottom()), egui::pos2(x, y)],
            egui::Stroke::new(bar_w, color),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stackbuf_push_silently_drops_past_max_window() {
        let mut buf = StackBuf::new();
        for i in 0..(MAX_WINDOW + 10) {
            buf.push(Duration::from_nanos(i as u64));
        }
        assert_eq!(
            buf.as_slice().len(),
            MAX_WINDOW,
            "push beyond cap must not allocate; samples drop"
        );
    }

    #[test]
    fn stackbuf_percentile_picks_nearest_rank() {
        let mut buf = StackBuf::new();
        for ms in [10, 4, 7, 1, 8, 2, 9, 6, 3, 5] {
            buf.push(Duration::from_millis(ms));
        }
        assert_eq!(buf.percentile(50), Duration::from_millis(5));
        assert_eq!(buf.percentile(95), Duration::from_millis(10));
        assert_eq!(buf.percentile(99), Duration::from_millis(10));
    }
}

#[cfg(test)]
mod seat_tests {
    use super::*;
    use loam_egui::egui;
    const BAR_HEIGHT: f32 = 64.0;

    const PANEL_WIDTH: f32 = 120.0;
    const VIEWPORT: egui::Vec2 = egui::vec2(1280.0, 800.0);
    const NARROW_VIEWPORT: egui::Vec2 = egui::vec2(200.0, 400.0);
    fn viewport(size: egui::Vec2) -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
            ..Default::default()
        }
    }

    #[test]
    fn perf_overlay_boots_into_the_top_right_of_the_panel_band() {
        frame_trace::begin_frame();
        frame_trace::end_frame();

        let ctx = egui::Context::default();
        let mut overlay = PerfOverlay::new().always_visible();
        let mut bar_bottom = 0.0;
        let mut band = egui::Rect::NOTHING;
        let _ = ctx.run(viewport(VIEWPORT), |ctx| {
            bar_bottom = egui::TopBottomPanel::top("shell-menu-bar")
                .exact_height(BAR_HEIGHT)
                .show(ctx, |ui| {
                    ui.label("bar");
                })
                .response
                .rect
                .bottom();
            band = ctx.available_rect();
            overlay.show(ctx);
        });
        let rect = ctx
            .memory(|m| m.area_rect(egui::Id::new("loam-perf-overlay")))
            .expect("overlay area is registered once shown");
        assert!(
            rect.top() >= bar_bottom,
            "overlay top {} must clear the menu bar bottom {bar_bottom}",
            rect.top(),
        );
        assert!(
            rect.center().x > band.center().x,
            "overlay center x {} must sit in the band's right half of {band:?}",
            rect.center().x,
        );
        assert!(
            rect.center().y < band.center().y,
            "overlay center y {} must sit in the band's top half of {band:?}",
            rect.center().y,
        );
    }

    #[test]
    fn perf_overlay_seat_clamps_to_the_band_left_when_the_band_is_narrower_than_the_readout() {
        let ctx = egui::Context::default();
        let mut seat = None;
        let mut band = egui::Rect::NOTHING;
        let _ = ctx.run(viewport(NARROW_VIEWPORT), |ctx| {
            egui::SidePanel::left("left")
                .exact_width(PANEL_WIDTH)
                .show(ctx, |ui| {
                    ui.label("left");
                });
            band = ctx.available_rect();
            seat = Some(perf_overlay_seat(ctx));
        });
        let seat = seat.expect("run closure sets the seat");
        assert!(
            band.width() < OVERLAY_MARGIN + OVERLAY_WIDTH && band.left() > 0.0,
            "fixture must leave a band {band:?} too narrow for the readout and \
             offset from the viewport origin, or the clamp goes untested",
        );
        assert_eq!(
            seat.x,
            band.left(),
            "a band too narrow for the readout must seat it at the band's left \
             edge, not at the viewport origin or off-screen",
        );
    }
}
