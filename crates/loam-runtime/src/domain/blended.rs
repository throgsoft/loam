use loam_math::blended::{
    geodesic_log_checked, integrate_geodesic_frame_checked, integrate_geodesic_frame_to_checked,
    BlendedSpace, BlendingField, ConformallyFlat, GeodesicError, GEODESIC_DEFAULT_STEPS,
    GEODESIC_ERROR_BUDGET, LOG_MAX_ITERS, LOG_RESIDUAL_TOL,
};
use loam_math::{Mat3, Space};

use super::space::{finite, frame_columns, frame_of, single_chart};
use super::{ChartId, ChartPoint, ChartPose, ChartTangent, DomainError, DomainSpace, Pose};
use crate::view::{self, DomainRay, Vec3};

fn geodesic_error(error: GeodesicError) -> DomainError {
    match error {
        GeodesicError::NonFinite => DomainError::InvalidCoordinate("geodesic"),
        GeodesicError::ChartBoundary => DomainError::ChartBoundary,
        GeodesicError::ErrorBudget { .. } => DomainError::ErrorBudget,
        GeodesicError::InvalidStepCount
        | GeodesicError::InvalidErrorBudget
        | GeodesicError::Singular
        | GeodesicError::NoConvergence => DomainError::NoConvergence,
    }
}

impl<A, B, F> DomainSpace for BlendedSpace<A, B, F>
where
    A: Space<Point = Vec3, Vector = Vec3> + ConformallyFlat + Send + Sync + 'static,
    B: Space<Point = Vec3, Vector = Vec3> + ConformallyFlat + Send + Sync + 'static,
    F: BlendingField,
{
    type Placement = Pose<Self>;

    type Relative = (Pose<Self>, Pose<Self>);

    const BLENDS: bool = false;

    fn origin(&self) -> Self::Point {
        Vec3::ZERO
    }

    fn check(&self, point: Self::Point) -> Result<(), DomainError> {
        finite(point.extend(0.0).to_array())?;
        self.valid_point(point)
            .then_some(())
            .ok_or(DomainError::ChartBoundary)
    }

    fn chart_dimension(&self) -> u32 {
        3
    }

    fn chart_point(&self, point: Self::Point) -> ChartPoint {
        ChartPoint {
            chart: ChartId(0),
            coordinates: point.extend(0.0).to_array(),
        }
    }

    fn local_point(&self, coordinates: [f32; 4]) -> Self::Point {
        Vec3::from_slice(&coordinates[..3])
    }

    // A Mat3 frame has no geodesic blend yet, so blended spaces draw each tick's pose until it does.
    fn blend(&self, _from: &Pose<Self>, to: &Pose<Self>, _t: f32) -> Pose<Self> {
        *to
    }

    fn chart_pose(&self, pose: &Pose<Self>) -> ChartPose {
        let relative = self.frame_at(pose.point).inverse() * pose.frame;
        ChartPose {
            chart: ChartId(0),
            coordinates: pose.point.extend(0.0).to_array(),
            frame: frame_of([relative.x_axis, relative.y_axis, relative.z_axis]),
        }
    }

    fn pose_from_chart(&self, pose: &ChartPose) -> Result<Pose<Self>, DomainError> {
        single_chart(pose.chart)?;
        finite(pose.coordinates)?;
        let point = Vec3::from_slice(&pose.coordinates[..3]);
        self.check(point)?;
        let [x, y, z] = frame_columns(&pose.frame)?;
        let frame = self.frame_at(point) * Mat3::from_cols(x, y, z);
        Ok(Pose { point, frame })
    }

    fn tangent_from_chart(&self, tangent: &ChartTangent) -> Result<Self::Vector, DomainError> {
        single_chart(tangent.chart)?;
        finite(tangent.vector)?;
        Ok(Vec3::from_slice(&tangent.vector[..3]))
    }

    fn chart_tangent(&self, tangent: Self::Vector) -> ChartTangent {
        ChartTangent {
            chart: ChartId(0),
            vector: tangent.extend(0.0).to_array(),
        }
    }

    fn chart_reach(&self, at: Self::Point, radius: f32) -> f32 {
        self.conformal_factor(at).sqrt() * radius
    }

    fn hit_ball(&self, ray: &DomainRay<Self>, center: Self::Point, radius: f32) -> Option<f32> {
        let unit = ray.direction.try_normalize()?;
        let chart_radius = radius / self.conformal_factor(center).sqrt();
        let offset = ray.origin - center;
        let entry = view::ball_entry(
            offset.dot(unit),
            offset.length_squared() - chart_radius * chart_radius,
        )?;
        Some(entry / ray.direction.length())
    }

    fn prepare(&self, pose: &Pose<Self>) -> Self::Placement {
        *pose
    }

    fn place(
        &self,
        placement: &Self::Placement,
        local: Self::Point,
    ) -> Result<Self::Point, DomainError> {
        self.walk(placement, local, 1.0).map(|pose| pose.point)
    }

    fn local(&self, pose: &Pose<Self>, point: Self::Point) -> Result<Self::Point, DomainError> {
        self.check(point)?;
        if self.conformal_factor(point).sqrt() * LOG_RESIDUAL_TOL > GEODESIC_ERROR_BUDGET {
            return Err(DomainError::ErrorBudget);
        }
        let chart = geodesic_log_checked(
            self,
            pose.point,
            point,
            GEODESIC_DEFAULT_STEPS,
            LOG_MAX_ITERS,
            GEODESIC_ERROR_BUDGET,
        )
        .map_err(geodesic_error)?
        .vector;
        let inverse = pose.frame.inverse();
        if !inverse.is_finite() {
            return Err(DomainError::InvalidFrame);
        }
        Ok(inverse * chart)
    }

    fn relative(&self, eye: &Pose<Self>, pose: &Pose<Self>) -> Result<Self::Relative, DomainError> {
        Ok((*eye, *pose))
    }

    fn place_relative(
        &self,
        relative: &Self::Relative,
        local: Self::Point,
    ) -> Result<Self::Point, DomainError> {
        let (eye, pose) = relative;
        self.local(eye, self.place(pose, local)?)
    }

    fn carry(
        &self,
        pose: &Pose<Self>,
        local: Self::Point,
        tangent: Self::Vector,
    ) -> Result<Self::Vector, DomainError> {
        let state = integrate_geodesic_frame_checked(
            self,
            pose.point,
            pose.frame * local,
            pose.frame,
            GEODESIC_DEFAULT_STEPS,
            GEODESIC_ERROR_BUDGET,
        )
        .map_err(geodesic_error)?;
        Ok(state.frame * tangent)
    }

    fn moved(&self, pose: &Pose<Self>, to: Self::Point) -> Result<Pose<Self>, DomainError> {
        self.check(to)?;
        let state = integrate_geodesic_frame_to_checked(
            self,
            pose.point,
            to,
            pose.frame,
            GEODESIC_DEFAULT_STEPS,
            LOG_MAX_ITERS,
            GEODESIC_ERROR_BUDGET,
        )
        .map_err(geodesic_error)?;
        Ok(Pose {
            point: state.point,
            frame: state.frame,
        })
    }

    fn walk(
        &self,
        pose: &Pose<Self>,
        tangent: Self::Vector,
        dt: f32,
    ) -> Result<Pose<Self>, DomainError> {
        let state = integrate_geodesic_frame_checked(
            self,
            pose.point,
            pose.frame * (tangent * dt),
            pose.frame,
            GEODESIC_DEFAULT_STEPS,
            GEODESIC_ERROR_BUDGET,
        )
        .map_err(geodesic_error)?;
        Ok(Pose {
            point: state.point,
            frame: state.frame,
        })
    }
}
