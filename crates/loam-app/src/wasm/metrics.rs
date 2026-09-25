pub(crate) fn capped(css_w: u32, css_h: u32, dpr: f32, max_pixels: Option<u32>) -> (u32, u32, f32) {
    let css_w = css_w.max(1);
    let css_h = css_h.max(1);
    let native_w = (f64::from(css_w) * f64::from(dpr)).round().max(1.0) as u32;
    let native_h = (f64::from(css_h) * f64::from(dpr)).round().max(1.0) as u32;
    let Some(max_pixels) = max_pixels.filter(|pixels| *pixels > 0) else {
        return (native_w, native_h, dpr);
    };
    if u64::from(native_w) * u64::from(native_h) <= u64::from(max_pixels) {
        return (native_w, native_h, dpr);
    }
    let css_area = f64::from(css_w) * f64::from(css_h);
    let scale = dpr.min((f64::from(max_pixels) / css_area).sqrt() as f32);
    let mut width = (f64::from(css_w) * f64::from(scale)).floor().max(1.0) as u32;
    let mut height = (f64::from(css_h) * f64::from(scale)).floor().max(1.0) as u32;
    if u64::from(width) * u64::from(height) > u64::from(max_pixels) {
        if width >= height {
            width = (max_pixels / height).max(1);
        } else {
            height = (max_pixels / width).max(1);
        }
    }
    (width, height, scale)
}
