//! Wasm caps frames by skipping animation callbacks; the browser sets the upper rate.

use web_time::Instant;

// Worst-case `std::thread::sleep` overshoot seen on Win11's 15.625 ms timer tick.
#[cfg(not(target_arch = "wasm32"))]
const SPIN_TAIL: std::time::Duration = std::time::Duration::from_millis(2);

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

/// `enabled` asks for Fifo; otherwise the first of Mailbox and Immediate the surface supports, else the current mode.
pub fn apply_present_mode(rd: &mut loam_render::device::RenderDevice, enabled: bool) {
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
