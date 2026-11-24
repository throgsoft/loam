use anyhow::{anyhow, Result};

pub(super) fn scaled_dims(width: u32, height: u32, scale: Option<u32>) -> Result<(u32, u32)> {
    let Some(target_w) = scale else {
        return Ok((width, height));
    };
    if target_w == 0 || width == 0 || height == 0 {
        return Err(anyhow!(
            "invalid scale: target_w={target_w}, src={width}x{height}"
        ));
    }
    let h = ((target_w as u64 * height as u64 + (width as u64) / 2) / width as u64) as u32;
    Ok((target_w, h.max(1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_capture() {
        assert_eq!(scaled_dims(1920, 1080, Some(720)).unwrap(), (720, 405));
        assert_eq!(scaled_dims(1080, 1920, Some(360)).unwrap(), (360, 640));
        assert_eq!(scaled_dims(800, 600, None).unwrap(), (800, 600));
        assert_eq!(scaled_dims(10000, 1, Some(100)).unwrap(), (100, 1));

        assert!(scaled_dims(1920, 1080, Some(0)).is_err());
        assert!(scaled_dims(0, 1080, Some(720)).is_err());
        assert!(scaled_dims(1920, 0, Some(720)).is_err());
    }
}
