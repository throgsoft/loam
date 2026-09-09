fn loam_projective_depth(image: vec3<f32>, near: f32) -> f32 {
    return near / max(-image.z, 1e-20);
}
