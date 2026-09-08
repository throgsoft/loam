//! Wasm caps frames by skipping animation callbacks; the browser sets the upper rate.

use std::time::Duration;
use web_time::Instant;

impl crate::Runtime {
    /// Nonpositive or nonfinite rates remove the cap.
    pub fn set_target_fps(&self, fps: f32) {
        self.0.period.set(if fps.is_finite() && fps > 0.0 {
            Some(Duration::from_nanos(
                ((1_000_000_000.0 / fps as f64) as u64).max(1),
            ))
        } else {
            None
        });
    }
    pub fn target_period(&self) -> Option<Duration> {
        self.0.period.get()
    }
    pub fn target_fps(&self) -> f32 {
        self.target_period()
            .map_or(0.0, |period| 1.0 / period.as_secs_f32())
    }
    pub fn request_vsync(&self, enabled: bool) {
        self.0.vsync.set(Some(enabled));
    }
    pub(crate) fn apply_present_mode(&self, rd: &mut loam_render::device::RenderDevice) {
        let Some(enabled) = self.take_vsync_request() else {
            return;
        };
        let target = if enabled {
            wgpu::PresentMode::Fifo
        } else {
            [wgpu::PresentMode::Mailbox, wgpu::PresentMode::Immediate]
                .into_iter()
                .find(|mode| rd.supported_present_modes().contains(mode))
                .unwrap_or(rd.present_mode())
        };
        if let Err(error) = rd.set_present_mode(target) {
            tracing::warn!(?error, "present mode rejected");
        }
    }

    pub(crate) fn take_vsync_request(&self) -> Option<bool> {
        self.0.vsync.take()
    }
}

// Worst-case `std::thread::sleep` overshoot seen on Win11's 15.625 ms timer tick.
#[cfg(not(target_arch = "wasm32"))]
const SPIN_TAIL: Duration = Duration::from_millis(2);

/// Sleeps, then spins the last `SPIN_TAIL`; plain sleep rounds to the timer tick.
#[cfg(not(target_arch = "wasm32"))]
pub fn precise_sleep_until(deadline: Instant) {
    let now = Instant::now();
    if deadline <= now {
        return;
    }
    let total = deadline - now;
    if total > SPIN_TAIL {
        std::thread::sleep(total - SPIN_TAIL);
    }
    while Instant::now() < deadline {
        std::hint::spin_loop();
    }
}

#[cfg(target_arch = "wasm32")]
pub fn precise_sleep_until(_deadline: Instant) {}
