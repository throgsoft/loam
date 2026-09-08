use glam::{Vec3, Vec4};
use loam_math::{EuclideanR4, Projection, RasterizableSpace};
use loam_shape::{projected_edges::push_blended_edge, LineMesh};

const SPACE_TESSELLATION_SAMPLES: usize = 16;
const STEREOGRAPHIC_VIEW_RADIUS: f32 = 6.0;

fn project_to_world(p: Vec4, projection: &Projection<4>, position: Vec3) -> Vec3 {
    EuclideanR4::project_point(p, projection) + position
}

mod blended_edge_tests {
    use super::*;

    fn flat_drop_w() -> loam_math::Projection<4> {
        loam_math::Projection::Identity
    }

    const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

    #[test]
    fn the_arc_bows_onto_the_circumsphere_about_the_arc_center() {
        const RADIUS: f32 = 0.5;
        const LIFT: f32 = 0.3;
        let center = Vec4::W * LIFT;
        let a = Vec4::new(RADIUS, 0.0, 0.0, 0.0) + center;
        let b = Vec4::new(0.0, RADIUS, 0.0, 0.0) + center;
        let mut mesh = LineMesh::<3>::default();
        push_blended_edge(
            &mut mesh,
            a,
            b,
            center,
            WHITE,
            WHITE,
            1.0,
            1.0,
            &flat_drop_w(),
            Vec3::ZERO,
            SPACE_TESSELLATION_SAMPLES,
            STEREOGRAPHIC_VIEW_RADIUS,
        );
        assert_eq!(mesh.segments.len(), SPACE_TESSELLATION_SAMPLES);
        for (k, (p0, p1)) in mesh.segments.iter().enumerate() {
            for p in [Vec3::from_array(*p0), Vec3::from_array(*p1)] {
                assert!(
                    (p.length() - RADIUS).abs() < 1e-5,
                    "sample {k} sits at {} from the lifted body's centre, not {RADIUS}",
                    p.length()
                );
            }
        }
    }

    fn perspective() -> loam_math::Projection<4> {
        loam_math::Projection::Perspective4D {
            focal_distance: 3.0,
        }
    }

    #[test]
    fn blend_zero_is_bit_identical_to_flat_chord() {
        let a = Vec4::new(0.5, 0.5, 0.5, 0.5);
        let b = Vec4::new(0.5, 0.5, 0.5, -0.5);
        let proj = perspective();
        let body_pos = Vec3::new(1.0, -2.0, 0.5);

        let mut mesh = LineMesh::<3>::default();
        push_blended_edge(
            &mut mesh,
            a,
            b,
            Vec4::ZERO,
            WHITE,
            WHITE,
            1.0,
            0.0,
            &proj,
            body_pos,
            SPACE_TESSELLATION_SAMPLES,
            STEREOGRAPHIC_VIEW_RADIUS,
        );

        assert_eq!(mesh.segments.len(), 1, "affine flat chord is one segment");
        let expected_a = project_to_world(a, &proj, body_pos).to_array();
        let expected_b = project_to_world(b, &proj, body_pos).to_array();
        let (seg_a, seg_b) = mesh.segments[0];
        assert_eq!(seg_a, expected_a, "start equals projected a");
        assert_eq!(seg_b, expected_b, "end equals projected b");
    }

    #[test]
    fn blend_endpoints_exact_at_all_t() {
        let a = Vec4::new(0.5, 0.5, 0.5, 0.5);
        let b = Vec4::new(-0.5, 0.5, 0.5, -0.5);
        let proj = loam_math::Projection::Stereographic { pole: Vec4::W };
        let body_pos = Vec3::new(-0.25, 1.5, 0.0);
        let expected_a = project_to_world(a, &proj, body_pos).to_array();
        let expected_b = project_to_world(b, &proj, body_pos).to_array();

        for &blend in &[0.0_f32, 0.001, 0.25, 0.5, 0.75, 1.0] {
            let mut mesh = LineMesh::<3>::default();
            push_blended_edge(
                &mut mesh,
                a,
                b,
                Vec4::ZERO,
                WHITE,
                WHITE,
                1.0,
                blend,
                &proj,
                body_pos,
                SPACE_TESSELLATION_SAMPLES,
                STEREOGRAPHIC_VIEW_RADIUS,
            );
            assert!(!mesh.segments.is_empty(), "blend {blend}: emitted nothing");
            let first = mesh.segments.first().unwrap().0;
            let last = mesh.segments.last().unwrap().1;
            assert_eq!(
                first, expected_a,
                "blend {blend}: first point equals proj(a)"
            );
            assert_eq!(last, expected_b, "blend {blend}: last point equals proj(b)");
        }
    }
}

mod pole_clipping {
    use super::*;

    fn build_stereographic_edge_with_blend(
        a: Vec4,
        b: Vec4,
        blend: f32,
    ) -> Vec<([f32; 3], [f32; 3])> {
        let proj = loam_math::Projection::Stereographic { pole: Vec4::W };
        let mut mesh = LineMesh::<3>::default();
        let white = [1.0, 1.0, 1.0, 1.0];
        push_blended_edge(
            &mut mesh,
            a,
            b,
            Vec4::ZERO,
            white,
            white,
            1.0,
            blend,
            &proj,
            Vec3::ZERO,
            SPACE_TESSELLATION_SAMPLES,
            STEREOGRAPHIC_VIEW_RADIUS,
        );
        mesh.segments
    }

    fn build_stereographic_edge(a: Vec4, b: Vec4) -> Vec<([f32; 3], [f32; 3])> {
        build_stereographic_edge_with_blend(a, b, 0.0)
    }

    fn build_spherical_stereographic_edge(a: Vec4, b: Vec4) -> Vec<([f32; 3], [f32; 3])> {
        build_stereographic_edge_with_blend(a, b, 1.0)
    }

    fn near_pole(theta_deg: f32) -> Vec4 {
        let t = theta_deg.to_radians();
        Vec4::new(t.sin(), 0.0, 0.0, t.cos())
    }

    #[test]
    fn stereographic_zero_blend_near_pole_uses_endpoint_clip() {
        let zero = build_stereographic_edge(near_pole(1.0), Vec4::new(1.0, 0.0, 0.0, 0.0));
        let spherical =
            build_spherical_stereographic_edge(near_pole(1.0), Vec4::new(1.0, 0.0, 0.0, 0.0));
        assert!(
            zero.is_empty(),
            "flat near-pole chord should drop when an endpoint clips out"
        );
        assert!(
            !spherical.is_empty(),
            "sampled S3 edge should resume after clipped near-pole samples"
        );
    }

    #[test]
    fn stereographic_clip_cuts_to_boundary_and_drops_deep_pole() {
        let r = STEREOGRAPHIC_VIEW_RADIUS;
        // cot(15°) ≈ 3.73 < R keeps the endpoints; the midpoint passes through the pole.
        let off = 30.0_f32.to_radians();
        let a = Vec4::new(off.sin(), 0.0, 0.0, off.cos());
        let b = Vec4::new(-off.sin(), 0.0, 0.0, off.cos());
        let segs = build_spherical_stereographic_edge(a, b);
        assert!(!segs.is_empty(), "kept endpoints must emit segments");

        let max_extent = segs
            .iter()
            .flat_map(|(s, e)| [Vec3::from_array(*s).length(), Vec3::from_array(*e).length()])
            .fold(0.0_f32, f32::max);
        assert!(
            (max_extent - r).abs() < 1e-2,
            "straddling sub-segment must be cut to the boundary (max extent {max_extent}, R {r})"
        );

        assert!(
            segs.len() < SPACE_TESSELLATION_SAMPLES,
            "deep-pole samples must drop (got {} of {}); a rescale-clamp would keep them all",
            segs.len(),
            SPACE_TESSELLATION_SAMPLES
        );

        for (s, e) in &segs {
            for end in [Vec3::from_array(*s), Vec3::from_array(*e)] {
                assert!(
                    end.length() <= r + 1e-3,
                    "endpoint {end:?} (|.| = {}) exceeds the bound {r}",
                    end.length()
                );
            }
        }
    }

    #[test]
    fn stereographic_clip_arc_tip_holds_boundary_near_pole() {
        let r = STEREOGRAPHIC_VIEW_RADIUS;
        let tip_extent = |phi_deg: f32| -> f32 {
            let phi = phi_deg.to_radians();
            let a = Vec4::new(-phi.sin(), 0.0, 0.0, phi.cos());
            let b = Vec4::new(0.0, 1.0, 0.0, 0.0);
            build_spherical_stereographic_edge(a, b)
                .iter()
                .flat_map(|(s, e)| [Vec3::from_array(*s).length(), Vec3::from_array(*e).length()])
                .fold(0.0_f32, f32::max)
        };
        let samples = 60;
        let hi = 3.0_f32.ln();
        let lo = 0.05_f32.ln();
        for step in 0..=samples {
            let frac = step as f32 / samples as f32;
            let phi = (hi + (lo - hi) * frac).exp();
            let tip = tip_extent(phi);
            assert!(
                tip > r - 1.0 && tip <= r + 1e-2,
                "near-pole arc tip must hold the boundary R={r} at phi={phi} deg, got {tip}"
            );
        }
    }

    #[test]
    fn stereographic_pole_endpoint_edge_is_finite_and_bounded() {
        let segs = build_spherical_stereographic_edge(Vec4::W, Vec4::new(1.0, 0.0, 0.0, 0.0));
        assert!(!segs.is_empty());
        let r = STEREOGRAPHIC_VIEW_RADIUS;
        for (s, e) in &segs {
            for end in [Vec3::from_array(*s), Vec3::from_array(*e)] {
                assert!(
                    end.is_finite(),
                    "pole-edge endpoint must be finite: {end:?}"
                );
                assert!(
                    end.length() <= r + 1e-3,
                    "pole-edge endpoint {end:?} exceeds clip radius {r}"
                );
            }
        }
    }

    #[test]
    fn stereographic_clip_does_not_perturb_off_pole_edge() {
        let proj = loam_math::Projection::Stereographic { pole: Vec4::W };
        let a = Vec4::new(0.30, 0.60, 0.20, 0.10).normalize();
        let b = Vec4::new(0.70, 0.10, 0.40, -0.30).normalize();
        let segs = build_spherical_stereographic_edge(a, b);
        assert_eq!(
            segs.len(),
            SPACE_TESSELLATION_SAMPLES,
            "off-pole edge must retain every sub-segment (none clipped)"
        );
        let samples = SPACE_TESSELLATION_SAMPLES;
        let mut arc = Vec::new();
        <loam_math::SphericalS3Embedded as loam_math::RasterizableSpace<4>>::tessellate_segment(
            a,
            b,
            samples,
            |point| arc.push(point),
        );
        let mut prev = project_to_world(a, &proj, Vec3::ZERO).to_array();
        for (k, (seg, &sample)) in segs.iter().zip(arc.iter().skip(1)).enumerate() {
            let cur = project_to_world(sample, &proj, Vec3::ZERO).to_array();
            assert_eq!(seg.0, prev, "segment {k} start must match raw projection");
            assert_eq!(seg.1, cur, "segment {k} end must match raw projection");
            prev = cur;
        }
    }
}

#[test]
fn exact_pole_never_becomes_a_flat_or_sampled_endpoint_at_the_origin() {
    use loam_shape::projected_edges::push_projected_chord;
    let projection = Projection::Stereographic { pole: Vec4::W };
    let mut flat = LineMesh::default();
    push_blended_edge(
        &mut flat,
        Vec4::X,
        Vec4::W,
        Vec4::ZERO,
        [1.0; 4],
        [1.0; 4],
        1.0,
        0.0,
        &projection,
        Vec3::ZERO,
        16,
        6.0,
    );
    assert!(flat.segments.is_empty());
    let mut sampled = LineMesh::default();
    push_projected_chord(
        &mut sampled,
        Vec4::X,
        Vec4::W,
        [1.0; 4],
        [1.0; 4],
        1.0,
        &projection,
        Vec3::ZERO,
        6.0,
        16,
    );
    assert!(!sampled.segments.is_empty());
    for (a, b) in sampled.segments {
        for point in [Vec3::from_array(a), Vec3::from_array(b)] {
            assert!(point.is_finite() && point.length() >= 0.999 && point.length() <= 6.0);
        }
    }
}
