use loam_math::{EuclideanR3, Iso3, IsometryGroup};

use super::space::{
    finite, frame_columns, frame_of, homogeneous_capabilities, homogeneous_carry,
    homogeneous_local, homogeneous_moved, homogeneous_place, homogeneous_relative,
    homogeneous_walk, single_chart,
};
use super::{
    ChartId, ChartPoint, ChartPose, ChartTangent, DomainError, DomainSpace, Homogeneous, Pose,
};
use crate::view::{self, DomainRay, Vec3};

// Shepperd, Quaternion from Rotation Matrix, J. Guidance and Control 1(3), 1978.
fn rotation_of([x, y, z]: [Vec3; 3]) -> Iso3 {
    let mut pose = Iso3::IDENTITY;
    let q = &mut pose.rotation;
    let trace = x.x + y.y + z.z;
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        (q.w, q.x, q.y, q.z) = (s * 0.25, (y.z - z.y) / s, (z.x - x.z) / s, (x.y - y.x) / s);
    } else if x.x > y.y && x.x > z.z {
        let s = (1.0 + x.x - y.y - z.z).sqrt() * 2.0;
        (q.w, q.x, q.y, q.z) = ((y.z - z.y) / s, s * 0.25, (y.x + x.y) / s, (z.x + x.z) / s);
    } else if y.y > z.z {
        let s = (1.0 + y.y - x.x - z.z).sqrt() * 2.0;
        (q.w, q.x, q.y, q.z) = ((z.x - x.z) / s, (y.x + x.y) / s, s * 0.25, (z.y + y.z) / s);
    } else {
        let s = (1.0 + z.z - x.x - y.y).sqrt() * 2.0;
        (q.w, q.x, q.y, q.z) = ((x.y - y.x) / s, (z.x + x.z) / s, (z.y + y.z) / s, s * 0.25);
    }
    pose.rotation = pose.rotation.normalize();
    pose
}

impl DomainSpace for EuclideanR3 {
    homogeneous_capabilities!();

    fn origin(&self) -> Self::Point {
        Vec3::ZERO
    }

    fn check(&self, point: Self::Point) -> Result<(), DomainError> {
        finite(point.extend(0.0).to_array())
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

    fn blend(&self, from: &Pose<Self>, to: &Pose<Self>, t: f32) -> Pose<Self> {
        Pose {
            point: from.point.lerp(to.point, t),
            frame: from.frame.slerp(to.frame, t),
        }
    }

    fn chart_pose(&self, pose: &Pose<Self>) -> ChartPose {
        ChartPose {
            chart: ChartId(0),
            coordinates: pose.point.extend(0.0).to_array(),
            frame: frame_of([Vec3::X, Vec3::Y, Vec3::Z].map(|axis| pose.frame * axis)),
        }
    }

    fn pose_from_chart(&self, pose: &ChartPose) -> Result<Pose<Self>, DomainError> {
        single_chart(pose.chart)?;
        finite(pose.coordinates)?;
        let frame = rotation_of(frame_columns(&pose.frame)?).rotation;
        Ok(Pose {
            point: Vec3::from_slice(&pose.coordinates[..3]),
            frame,
        })
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
        radius
    }

    fn hit_ball(&self, ray: &DomainRay<Self>, center: Self::Point, radius: f32) -> Option<f32> {
        let offset = ray.origin - center;
        view::ball_entry(
            offset.dot(ray.direction),
            offset.length_squared() - radius * radius,
        )
    }
}

impl Homogeneous for EuclideanR3 {
    fn iso_of(&self, pose: &Pose<Self>) -> Self::Iso {
        Iso3 {
            rotation: pose.frame,
            translation: pose.point,
        }
    }

    fn pose_of(&self, iso: Self::Iso) -> Pose<Self> {
        Pose {
            point: iso.translation,
            frame: iso.rotation,
        }
    }

    fn transvection(&self, to: Self::Point) -> Self::Iso {
        Iso3::from_translation(to)
    }
}

impl From<Iso3> for Pose<EuclideanR3> {
    fn from(iso: Iso3) -> Self {
        EuclideanR3.pose_of(iso)
    }
}
