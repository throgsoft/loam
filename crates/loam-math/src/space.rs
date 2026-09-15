use std::borrow::Cow;

pub trait Space {
    type Point: Copy + Send + Sync + 'static;
    type Vector: Copy + Send + Sync + 'static;
    type Frame: Copy + Send + Sync + 'static;

    fn frame_at(&self, at: Self::Point) -> Self::Frame;

    fn distance(&self, a: Self::Point, b: Self::Point) -> f32;

    /// Advances for unit time along the geodesic with initial velocity `v`.
    fn exp(&self, at: Self::Point, v: Self::Vector) -> Self::Point;

    fn log(&self, from: Self::Point, to: Self::Point) -> Self::Vector;

    fn parallel_transport(
        &self,
        from: Self::Point,
        to: Self::Point,
        v: Self::Vector,
    ) -> Self::Vector;

    /// Composes transport along consecutive points; fewer than two points leave `v` unchanged.
    fn parallel_transport_along(&self, path: &[Self::Point], v: Self::Vector) -> Self::Vector {
        let mut current = v;
        for w in path.windows(2) {
            current = self.parallel_transport(w[0], w[1], current);
        }
        current
    }

    /// True when chart arithmetic respects the global geometry, including identifications.
    fn is_chart_flat(&self) -> bool {
        false
    }

    /// Geodesic distance from the chart origin beyond which [`Self::valid_point`] refuses a point.
    fn chart_envelope(&self) -> f32;

    /// Finite, inside the chart, and within [`Self::chart_envelope`] of the origin.
    fn valid_point(&self, p: Self::Point) -> bool;
}

pub trait IsometryGroup: Space {
    type Iso: Copy + Send + Sync + 'static;

    fn iso_identity(&self) -> Self::Iso;

    /// `a ∘ b`, apply `b` first, then `a`.
    fn iso_compose(&self, a: Self::Iso, b: Self::Iso) -> Self::Iso;

    fn iso_inverse(&self, a: Self::Iso) -> Self::Iso;

    fn iso_apply(&self, iso: Self::Iso, p: Self::Point) -> Self::Point;

    /// Maps tangents at `at` to tangents at `iso_apply(iso, at)`.
    fn iso_transport(&self, iso: Self::Iso, at: Self::Point, v: Self::Vector) -> Self::Vector;
}

pub const FLAT_CHART_MAX_ARC: f32 = 40.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WgslAccuracy {
    Exact,
    FirstOrder { residual: f32 },
}

impl WgslAccuracy {
    pub fn residual(self) -> f32 {
        match self {
            WgslAccuracy::Exact => 0.0,
            WgslAccuracy::FirstOrder { residual } => residual,
        }
    }
}

pub trait WgslSpace: Space {
    /// Emits `loam_distance`, `loam_origin_distance`, `loam_exp`, `loam_log`, `loam_geodesic_step`, and `LOAM_MAX_ARC` over the space's point type.
    fn wgsl_impl(&self) -> Cow<'static, str>;

    fn wgsl_accuracy(&self) -> WgslAccuracy;
}
