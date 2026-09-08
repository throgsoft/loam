use std::borrow::Cow;

/// A metric and its connection in one coordinate representation.
pub trait Space {
    type Point: Copy + Send + Sync + 'static;
    type Vector: Copy + Send + Sync + 'static;

    fn distance(&self, a: Self::Point, b: Self::Point) -> f32;

    /// Advances for unit time along the geodesic with initial velocity `v`.
    fn exp(&self, at: Self::Point, v: Self::Vector) -> Self::Point;

    /// Inverse exponential map; each implementation defines its cut-locus behavior.
    fn log(&self, from: Self::Point, to: Self::Point) -> Self::Vector;

    /// Uses the implementation's path; use [`Self::parallel_transport_along`] to specify a polyline.
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
}

/// A distance-preserving group action, with tangent transport given by its differential.
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

pub trait WgslSpace: Space {
    /// Emits vec3 `loam_distance`, `loam_origin_distance`, `loam_exp`, `loam_log`, `loam_parallel_transport`, and `LOAM_MAX_ARC`.
    fn wgsl_impl(&self) -> Cow<'static, str>;
}
