#[path = "../src/wasm/metrics.rs"]
#[allow(dead_code)]
mod metrics;

use metrics::capped;

#[test]
fn a_capped_buffer_stays_within_the_budget_after_rounding_and_never_exceeds_device_scale() {
    const FULL_HD: u32 = 1920 * 1080;

    assert_eq!(capped(960, 540, 2.0, Some(FULL_HD)), (1920, 1080, 2.0));
    assert_eq!(capped(960, 540, 2.0, None), (1920, 1080, 2.0));
    assert_eq!(capped(960, 540, 2.0, Some(0)), capped(960, 540, 2.0, None));

    let (w, h, scale) = capped(1921, 1081, 1.0, Some(FULL_HD));
    assert!(w * h <= FULL_HD && w < 1921 && h < 1081);
    assert!(scale < 1.0);

    for css_w in [
        1, 2, 3, 7, 320, 375, 390, 414, 768, 1024, 1366, 1440, 1921, 2560, 3840,
    ] {
        for css_h in [
            1, 5, 9, 11, 640, 812, 844, 896, 1024, 1081, 1200, 1600, 2160,
        ] {
            for dpr in [0.5, 1.0, 1.25, 1.5, 2.0, 2.625, 3.0] {
                for max_pixels in [1, 7, 100, 4_999, 500_000, FULL_HD, u32::MAX] {
                    let (w, h, scale) = capped(css_w, css_h, dpr, Some(max_pixels));
                    let native = (f64::from(css_w) * f64::from(dpr)).round()
                        * (f64::from(css_h) * f64::from(dpr)).round();
                    assert!(w >= 1 && h >= 1, "{css_w}x{css_h}@{dpr} cap {max_pixels}");
                    assert!(
                        u64::from(w) * u64::from(h) <= u64::from(max_pixels),
                        "{css_w}x{css_h}@{dpr} cap {max_pixels} gave {w}x{h}"
                    );
                    assert!(
                        scale <= dpr && scale > 0.0,
                        "{css_w}x{css_h}@{dpr} cap {max_pixels}"
                    );
                    if native <= f64::from(max_pixels) {
                        assert_eq!(scale, dpr, "{css_w}x{css_h}@{dpr} cap {max_pixels}");
                    }
                }
            }
        }
    }
}
