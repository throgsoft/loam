// WGSL numeric literals: an unsuffixed integer must fit AbstractInt (i64), spec §15.2.
pub(crate) fn wgsl_f32(v: f32) -> String {
    assert!(
        v.is_finite(),
        "non-finite scene constant {v:?} has no WGSL literal",
    );
    format!("{v:?}")
}
