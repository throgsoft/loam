use web_sys::HtmlCanvasElement;

#[derive(Clone, Copy, PartialEq)]
pub(super) struct CanvasMetrics {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) scale: f32,
}

pub(super) fn canvas_metrics(
    canvas: &HtmlCanvasElement,
    device_scale: f32,
    max_pixels: Option<u32>,
) -> CanvasMetrics {
    let (width, height, scale) = crate::web::metrics::capped(
        canvas.client_width().max(1) as u32,
        canvas.client_height().max(1) as u32,
        device_scale,
        max_pixels,
    );
    CanvasMetrics {
        width,
        height,
        scale,
    }
}
