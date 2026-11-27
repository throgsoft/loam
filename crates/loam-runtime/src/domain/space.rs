use std::ops::Mul;

use loam_math::{EuclideanR4, IsometryGroup, Space};

use super::chart::{finite, single_chart};
use super::{ChartPoint, ChartPose, ChartTangent, DomainError, Pose};
use crate::view::DomainRay;

pub trait DomainSpace:
    Space<Vector: Mul<f32, Output = <Self as Space>::Vector>> + Send + Sized + 'static
{
    type Placement: Copy + Send + 'static;

    type Relative: Copy + Send + 'static;

    fn origin(&self) -> Self::Point;

    fn check(&self, point: Self::Point) -> Result<(), DomainError>;

    fn chart_dimension(&self) -> u32;

    fn chart_point(&self, point: Self::Point) -> ChartPoint;

    /// Reads the leading coordinates the space has and ignores the rest.
    fn local_point(&self, coordinates: [f32; 4]) -> Self::Point;

    fn point_from_chart(&self, point: &ChartPoint) -> Result<Self::Point, DomainError> {
        single_chart(point.chart)?;
        finite(point.coordinates)?;
        let point = self.local_point(point.coordinates);
        self.check(point)?;
        Ok(point)
    }

    fn chart_pose(&self, pose: &Pose<Self>) -> ChartPose;

    /// The pose drawn a fraction `t` of the way through a tick that moved `from` to `to`.
    fn blend(&self, from: &Pose<Self>, to: &Pose<Self>, t: f32) -> Pose<Self>;

    fn pose_from_chart(&self, pose: &ChartPose) -> Result<Pose<Self>, DomainError>;

    fn tangent_from_chart(&self, tangent: &ChartTangent) -> Result<Self::Vector, DomainError>;

    fn chart_tangent(&self, tangent: Self::Vector) -> ChartTangent;

    fn chart_reach(&self, at: Self::Point, radius: f32) -> f32;

    /// Arc length along `ray` into the metric ball of `radius` around `center`: zero from inside, `None` on a miss.
    fn hit_ball(&self, ray: &DomainRay<Self>, center: Self::Point, radius: f32) -> Option<f32>;

    fn prepare(&self, pose: &Pose<Self>) -> Self::Placement;

    fn place(
        &self,
        placement: &Self::Placement,
        local: Self::Point,
    ) -> Result<Self::Point, DomainError>;

    fn local(&self, pose: &Pose<Self>, point: Self::Point) -> Result<Self::Point, DomainError>;

    fn relative(&self, eye: &Pose<Self>, pose: &Pose<Self>) -> Result<Self::Relative, DomainError>;

    fn place_relative(
        &self,
        relative: &Self::Relative,
        local: Self::Point,
    ) -> Result<Self::Point, DomainError>;

    fn carry(
        &self,
        pose: &Pose<Self>,
        local: Self::Point,
        tangent: Self::Vector,
    ) -> Result<Self::Vector, DomainError>;

    fn moved(&self, pose: &Pose<Self>, to: Self::Point) -> Result<Pose<Self>, DomainError>;

    /// The flat 4D pose a field kernel draws `relative` at, or `None` for a space without one.
    fn field_pose(&self, _relative: &Self::Relative) -> Option<Pose<EuclideanR4>> {
        None
    }

    fn walk(
        &self,
        pose: &Pose<Self>,
        tangent: Self::Vector,
        dt: f32,
    ) -> Result<Pose<Self>, DomainError> {
        let step = self.carry(pose, self.origin(), tangent * dt)?;
        let to = self.exp(pose.point, step);
        self.check(to)?;
        self.moved(pose, to)
    }
}

pub trait Homogeneous: DomainSpace + IsometryGroup {
    fn iso_of(&self, pose: &Pose<Self>) -> Self::Iso;

    fn pose_of(&self, iso: Self::Iso) -> Pose<Self>;

    fn transvection(&self, to: Self::Point) -> Self::Iso;
}

pub(super) fn homogeneous_place<S: Homogeneous>(
    space: &S,
    placement: &S::Iso,
    local: S::Point,
) -> Result<S::Point, DomainError> {
    let point = space.iso_apply(*placement, local);
    space.check(point)?;
    Ok(point)
}

pub(super) fn homogeneous_local<S: Homogeneous>(
    space: &S,
    pose: &Pose<S>,
    point: S::Point,
) -> Result<S::Point, DomainError> {
    Ok(space.iso_apply(space.iso_inverse(space.iso_of(pose)), point))
}

pub(super) fn homogeneous_carry<S: Homogeneous>(
    space: &S,
    pose: &Pose<S>,
    local: S::Point,
    tangent: S::Vector,
) -> Result<S::Vector, DomainError> {
    Ok(space.iso_transport(space.iso_of(pose), local, tangent))
}

pub(super) fn homogeneous_relative<S: Homogeneous>(
    space: &S,
    eye: &Pose<S>,
    pose: &Pose<S>,
) -> Result<S::Iso, DomainError> {
    Ok(space.iso_compose(space.iso_inverse(space.iso_of(eye)), space.iso_of(pose)))
}

macro_rules! homogeneous_capabilities {
    () => {
        type Placement = <Self as IsometryGroup>::Iso;

        type Relative = <Self as IsometryGroup>::Iso;

        fn prepare(&self, pose: &Pose<Self>) -> Self::Placement {
            self.iso_of(pose)
        }

        fn place(
            &self,
            placement: &Self::Placement,
            local: Self::Point,
        ) -> Result<Self::Point, DomainError> {
            $crate::domain::space::homogeneous_place(self, placement, local)
        }

        fn local(&self, pose: &Pose<Self>, point: Self::Point) -> Result<Self::Point, DomainError> {
            $crate::domain::space::homogeneous_local(self, pose, point)
        }

        fn relative(
            &self,
            eye: &Pose<Self>,
            pose: &Pose<Self>,
        ) -> Result<Self::Relative, DomainError> {
            $crate::domain::space::homogeneous_relative(self, eye, pose)
        }

        fn place_relative(
            &self,
            relative: &Self::Relative,
            local: Self::Point,
        ) -> Result<Self::Point, DomainError> {
            $crate::domain::space::homogeneous_place(self, relative, local)
        }

        fn carry(
            &self,
            pose: &Pose<Self>,
            local: Self::Point,
            tangent: Self::Vector,
        ) -> Result<Self::Vector, DomainError> {
            $crate::domain::space::homogeneous_carry(self, pose, local, tangent)
        }

        fn moved(&self, pose: &Pose<Self>, to: Self::Point) -> Result<Pose<Self>, DomainError> {
            $crate::domain::space::homogeneous_moved(self, pose, to)
        }

        fn walk(
            &self,
            pose: &Pose<Self>,
            tangent: Self::Vector,
            dt: f32,
        ) -> Result<Pose<Self>, DomainError> {
            $crate::domain::space::homogeneous_walk(self, pose, tangent, dt)
        }
    };
}

pub(super) use homogeneous_capabilities;

pub(super) fn homogeneous_moved<S: Homogeneous>(
    space: &S,
    pose: &Pose<S>,
    to: S::Point,
) -> Result<Pose<S>, DomainError> {
    space.check(to)?;
    let pose_iso = space.iso_of(pose);
    let local = space.iso_apply(space.iso_inverse(pose_iso), to);
    let next = space.pose_of(space.iso_compose(pose_iso, space.transvection(local)));
    space.check(next.point)?;
    Ok(next)
}

pub(super) fn homogeneous_walk<S: Homogeneous>(
    space: &S,
    pose: &Pose<S>,
    tangent: S::Vector,
    dt: f32,
) -> Result<Pose<S>, DomainError> {
    let step = space.exp(space.origin(), tangent * dt);
    space.check(step)?;
    // Helgason, Differential Geometry, Lie Groups, and Symmetric Spaces, 1978, Ch. IV sec. 3: the transvection along a geodesic is parallel transport along it.
    let next = space.pose_of(space.iso_compose(space.iso_of(pose), space.transvection(step)));
    space.check(next.point)?;
    Ok(next)
}
