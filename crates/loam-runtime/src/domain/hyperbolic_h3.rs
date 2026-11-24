use loam_math::hyperbolic::{in_poincare_ball, poincare_to_hyperboloid};
use loam_math::{HyperbolicH3, Iso3H, IsometryGroup, Space};

use super::chart::{finite, frame_columns, frame_of, single_chart};
use super::space::{
    homogeneous_capabilities, homogeneous_carry, homogeneous_local, homogeneous_moved,
    homogeneous_place, homogeneous_relative, homogeneous_walk,
};
use super::{
    ChartId, ChartPoint, ChartPose, ChartTangent, DomainError, DomainSpace, Homogeneous, Pose,
};
use crate::view::{DomainRay, Vec3, Vec4};

fn lorentz(a: Vec4, b: Vec4) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z - a.w * b.w
}

impl DomainSpace for HyperbolicH3 {
    homogeneous_capabilities!();

    const BLENDS: bool = false;

    fn origin(&self) -> Self::Point {
        Vec3::ZERO
    }

    fn check(&self, point: Self::Point) -> Result<(), DomainError> {
        finite(point.extend(0.0).to_array())?;
        if !in_poincare_ball(point) {
            return Err(DomainError::ChartBoundary);
        }
        self.valid_point(point)
            .then_some(())
            .ok_or(DomainError::ErrorBudget)
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

    // An Iso3H frame has no geodesic blend yet, so H3 draws each tick's pose until it does.
    fn blend(&self, _from: &Pose<Self>, to: &Pose<Self>, _t: f32) -> Pose<Self> {
        *to
    }

    fn chart_pose(&self, pose: &Pose<Self>) -> ChartPose {
        let position = pose.point;
        let rotation = self.iso_compose(
            self.iso_inverse(Iso3H::from_translation(position)),
            pose.frame,
        );
        let matrix = rotation.matrix;
        ChartPose {
            chart: ChartId(0),
            coordinates: position.extend(0.0).to_array(),
            frame: frame_of([
                matrix.x_axis.truncate(),
                matrix.y_axis.truncate(),
                matrix.z_axis.truncate(),
            ]),
        }
    }

    fn pose_from_chart(&self, pose: &ChartPose) -> Result<Pose<Self>, DomainError> {
        single_chart(pose.chart)?;
        finite(pose.coordinates)?;
        let position = Vec3::from_slice(&pose.coordinates[..3]);
        self.check(position)?;
        let [x, y, z] = frame_columns(&pose.frame)?;
        let mut rotation = Iso3H::IDENTITY;
        rotation.matrix.x_axis = x.extend(0.0);
        rotation.matrix.y_axis = y.extend(0.0);
        rotation.matrix.z_axis = z.extend(0.0);
        Ok(self.pose_of(self.iso_compose(Iso3H::from_translation(position), rotation)))
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

    fn chart_reach(&self, _at: Self::Point, radius: f32) -> f32 {
        2.0 * radius
            .min(loam_math::hyperbolic::POINCARE_R2_MAX.sqrt())
            .atanh()
    }

    fn hit_ball(&self, ray: &DomainRay<Self>, center: Self::Point, radius: f32) -> Option<f32> {
        // Ratcliffe, Foundations of Hyperbolic Manifolds, 2006, §3.2: cosh d(x, y) = -⟨x, y⟩ on the hyperboloid.
        let start = poincare_to_hyperboloid(ray.origin);
        let ahead = poincare_to_hyperboloid(self.exp(ray.origin, ray.direction));
        let tangent = (ahead - start * 1.0_f32.cosh()) / 1.0_f32.sinh();
        let target = poincare_to_hyperboloid(center);
        let reach = -lorentz(start, target);
        let along = -lorentz(tangent, target);
        let nearest = (reach * reach - along * along).max(1.0).sqrt();
        let foot = -(along / reach).atanh();
        let ratio = radius.cosh() / nearest;
        if ratio < 1.0 {
            return None;
        }
        let half = ratio.acosh();
        let entry = foot - half;
        if entry <= 0.0 {
            return (foot + half >= 0.0).then_some(0.0);
        }
        Some(entry)
    }
}

impl Homogeneous for HyperbolicH3 {
    fn iso_of(&self, pose: &Pose<Self>) -> Self::Iso {
        pose.frame
    }

    fn pose_of(&self, iso: Self::Iso) -> Pose<Self> {
        Pose {
            point: self.iso_apply(iso, Vec3::ZERO),
            frame: iso,
        }
    }

    fn transvection(&self, to: Self::Point) -> Self::Iso {
        Iso3H::from_translation(to)
    }
}

impl From<Iso3H> for Pose<HyperbolicH3> {
    fn from(iso: Iso3H) -> Self {
        HyperbolicH3.pose_of(iso)
    }
}
