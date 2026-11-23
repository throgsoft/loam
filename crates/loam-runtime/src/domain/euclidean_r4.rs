use loam_math::{Bivector, EuclideanR4, Iso4Flat, IsometryGroup, Rotor, Rotor4};

use super::space::{
    finite, homogeneous_capabilities, homogeneous_carry, homogeneous_local, homogeneous_moved,
    homogeneous_place, homogeneous_relative, homogeneous_walk, single_chart,
};
use super::{
    ChartId, ChartPoint, ChartPose, ChartTangent, DomainError, DomainSpace, Homogeneous, Pose,
};
use crate::view::{self, DomainRay, Vec4};

impl DomainSpace for EuclideanR4 {
    homogeneous_capabilities!();

    fn field_pose(&self, relative: &Iso4Flat) -> Option<Pose<EuclideanR4>> {
        Some(Pose {
            point: relative.translation,
            frame: relative.rotation,
        })
    }

    fn origin(&self) -> Self::Point {
        Vec4::ZERO
    }

    fn check(&self, point: Self::Point) -> Result<(), DomainError> {
        finite(point.to_array())
    }

    fn chart_dimension(&self) -> u32 {
        4
    }

    fn chart_point(&self, point: Self::Point) -> ChartPoint {
        ChartPoint {
            chart: ChartId(0),
            coordinates: point.to_array(),
        }
    }

    fn local_point(&self, coordinates: [f32; 4]) -> Self::Point {
        Vec4::from_array(coordinates)
    }

    // Geodesic R0 exp(t log(R0^-1 R1)), the SO(4) slerp (Shoemake 1985, section 3).
    fn blend(&self, from: &Pose<Self>, to: &Pose<Self>, t: f32) -> Pose<Self> {
        let relative = from.frame.inverse() * to.frame;
        Pose {
            point: from.point.lerp(to.point, t),
            frame: from.frame * (relative.log() * t).exp(),
        }
    }

    fn chart_pose(&self, pose: &Pose<Self>) -> ChartPose {
        ChartPose {
            chart: ChartId(0),
            coordinates: pose.point.to_array(),
            frame: pose.frame.to_mat4(),
        }
    }

    fn pose_from_chart(&self, pose: &ChartPose) -> Result<Pose<Self>, DomainError> {
        single_chart(pose.chart)?;
        finite(pose.coordinates)?;
        let frame = Rotor4::from_mat4(&pose.frame).ok_or(DomainError::InvalidFrame)?;
        Ok(Pose {
            point: Vec4::from_array(pose.coordinates),
            frame,
        })
    }

    fn tangent_from_chart(&self, tangent: &ChartTangent) -> Result<Self::Vector, DomainError> {
        single_chart(tangent.chart)?;
        finite(tangent.vector)?;
        Ok(Vec4::from_array(tangent.vector))
    }

    fn chart_tangent(&self, tangent: Self::Vector) -> ChartTangent {
        ChartTangent {
            chart: ChartId(0),
            vector: tangent.to_array(),
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

impl Homogeneous for EuclideanR4 {
    fn iso_of(&self, pose: &Pose<Self>) -> Self::Iso {
        Iso4Flat {
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
        Iso4Flat::from_translation(to)
    }
}

impl From<Iso4Flat> for Pose<EuclideanR4> {
    fn from(iso: Iso4Flat) -> Self {
        EuclideanR4.pose_of(iso)
    }
}
