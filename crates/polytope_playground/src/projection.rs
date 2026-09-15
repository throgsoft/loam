use glam::{Vec3, Vec4};
use loam::math::{EuclideanR4, Projection, RasterizableSpace, SphericalS3Embedded};
use loam::runtime::{
    DepthEnvelope, DomainRay, DomainSpace, ImageRay, Pose, SectionCut, ViewMapping,
};
use loam::shape::projected_edges::stereographic_view_point;

pub(crate) const STEREOGRAPHIC_POLE: Vec4 = Vec4::W;

pub(crate) const PERSPECTIVE_FOCAL: f32 = 2.0;

const STEREOGRAPHIC_EDGE_SAMPLES: usize = 16;

const STEREOGRAPHIC_CLIP_RADIUS: f32 = 10.0;

// Pharr, Jakob, and Humphreys, Physically Based Rendering, 4th ed., §6.2.
fn clip_crossing(inside: Vec3, outside: Vec3) -> f32 {
    let direction = outside - inside;
    let a = direction.length_squared();
    let b = inside.dot(direction);
    let c = inside.length_squared() - STEREOGRAPHIC_CLIP_RADIUS.powi(2);
    let discriminant = (b * b - a * c).max(0.0);
    ((-b + discriminant.sqrt()) / a).clamp(0.0, 1.0)
}

const UNBOUNDED: DepthEnvelope = DepthEnvelope {
    near: 0.0,
    far: f32::INFINITY,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Family {
    #[default]
    Shadow,
    Perspective,
    Stereographic,
}

impl Family {
    pub(crate) const ALL: [Family; 3] =
        [Family::Shadow, Family::Perspective, Family::Stereographic];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Family::Shadow => "shadow",
            Family::Perspective => "perspective",
            Family::Stereographic => "stereographic",
        }
    }

    pub(crate) fn from_token(token: &str) -> Option<Self> {
        Family::ALL
            .into_iter()
            .find(|family| family.name() == token)
    }

    pub(crate) fn mapping(self, slice: f32) -> Projected {
        let projection = match self {
            Family::Shadow => Projection::Identity,
            Family::Perspective => Projection::Perspective4D {
                focal_distance: PERSPECTIVE_FOCAL,
            },
            Family::Stereographic => Projection::Stereographic {
                pole: STEREOGRAPHIC_POLE,
            },
        };
        Projected {
            name: self.name(),
            projection,
            slice,
        }
    }
}

pub(crate) struct Projected {
    name: &'static str,
    projection: Projection<4>,
    slice: f32,
}

impl Projected {
    fn project(&self, point: Vec4) -> Vec3 {
        if point == Vec4::ZERO {
            return Vec3::ZERO;
        }
        stereographic_view_point(point, &self.projection)
    }

    fn visible(&self, projected: Vec3) -> bool {
        projected.is_finite()
            && (!matches!(self.projection, Projection::Stereographic { .. })
                || projected.length_squared()
                    <= STEREOGRAPHIC_CLIP_RADIUS * STEREOGRAPHIC_CLIP_RADIUS)
    }
}

impl ViewMapping<EuclideanR4> for Projected {
    fn name(&self) -> &'static str {
        self.name
    }

    fn image_point(&self, eye: &Pose<EuclideanR4>, point: Vec4) -> Option<[f32; 3]> {
        let relative = EuclideanR4.local(eye, point).ok()?;
        let placed = self.project(relative);
        placed.is_finite().then(|| placed.to_array())
    }

    fn image_local(
        &self,
        space: &EuclideanR4,
        _eye: &Pose<EuclideanR4>,
        _pose: &Pose<EuclideanR4>,
        relative: &<EuclideanR4 as DomainSpace>::Relative,
        local: Vec4,
    ) -> Option<[f32; 3]> {
        let origin = space.place_relative(relative, Vec4::ZERO).ok()?;
        let point = space.place_relative(relative, local).ok()?;
        let projected = self.project(point - origin);
        self.visible(projected)
            .then(|| (projected + origin.truncate()).to_array())
    }

    fn image_segment(
        &self,
        space: &EuclideanR4,
        eye: &Pose<EuclideanR4>,
        pose: &Pose<EuclideanR4>,
        relative: &<EuclideanR4 as DomainSpace>::Relative,
        segment: [Vec4; 2],
        emit: &mut dyn FnMut(f32, Option<[f32; 3]>),
    ) {
        let [start, end] = segment;
        if !matches!(self.projection, Projection::Stereographic { .. }) {
            emit(0.0, self.image_local(space, eye, pose, relative, start));
            emit(1.0, self.image_local(space, eye, pose, relative, end));
            return;
        }
        let Ok(origin) = space.place_relative(relative, Vec4::ZERO) else {
            emit(0.0, None);
            emit(1.0, None);
            return;
        };
        let image = |local| {
            let point = space.place_relative(relative, local).ok()?;
            let projected = self.project(point - origin);
            projected
                .is_finite()
                .then(|| (projected, (projected + origin.truncate()).to_array()))
        };
        let clip_radius_squared = STEREOGRAPHIC_CLIP_RADIUS * STEREOGRAPHIC_CLIP_RADIUS;
        if start.length_squared() <= f32::MIN_POSITIVE || end.length_squared() <= f32::MIN_POSITIVE
        {
            let clipped = |local| {
                image(local).and_then(|(projected, world)| self.visible(projected).then_some(world))
            };
            emit(0.0, clipped(start));
            emit(1.0, clipped(end));
            return;
        }
        let start_radius = start.length();
        let end_radius = end.length();
        let mut index = 0usize;
        let mut previous = None;
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(
            start / start_radius,
            end / end_radius,
            STEREOGRAPHIC_EDGE_SAMPLES,
            |unit| {
                let t = index as f32 / STEREOGRAPHIC_EDGE_SAMPLES as f32;
                index += 1;
                let radius = start_radius + (end_radius - start_radius) * t;
                let Some((projected, world)) = image(unit * radius) else {
                    emit(t, None);
                    previous = None;
                    return;
                };
                let inside = projected.length_squared() <= clip_radius_squared;
                if let Some((previous_t, previous_projected, previous_inside)) = previous {
                    match (previous_inside, inside) {
                        (true, true) => emit(t, Some(world)),
                        (true, false) => {
                            let crossing = clip_crossing(previous_projected, projected);
                            let boundary = previous_projected.lerp(projected, crossing);
                            emit(
                                previous_t + (t - previous_t) * crossing,
                                Some((boundary + origin.truncate()).to_array()),
                            );
                            emit(t, None);
                        }
                        (false, true) => {
                            let crossing = clip_crossing(projected, previous_projected);
                            let boundary = projected.lerp(previous_projected, crossing);
                            emit(
                                t + (previous_t - t) * crossing,
                                Some((boundary + origin.truncate()).to_array()),
                            );
                            emit(t, Some(world));
                        }
                        (false, false) => emit(t, None),
                    }
                } else {
                    emit(t, inside.then_some(world));
                }
                previous = Some((t, projected, inside));
            },
        );
    }

    fn section(&self, eye: &Pose<EuclideanR4>, pose: &Pose<EuclideanR4>) -> Option<SectionCut> {
        let Projection::Perspective4D { focal_distance } = self.projection else {
            return None;
        };
        let depth = focal_distance - self.slice;
        let relative = EuclideanR4.local(eye, pose.point).ok()?;
        (depth > 0.0).then(|| SectionCut {
            offset: self.slice - relative.w,
            scale: focal_distance / depth,
        })
    }

    fn lift(&self, _eye: &Pose<EuclideanR4>, _ray: &ImageRay) -> Option<DomainRay<EuclideanR4>> {
        None
    }

    fn ray_lift(&self) -> bool {
        false
    }

    fn depth_envelope(&self) -> DepthEnvelope {
        UNBOUNDED
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::BODY_SIZE;

    const OFFSET: Vec4 = Vec4::new(1.8, 0.9, 0.0, 0.0);

    fn placed(family: Family, local: Vec4) -> Vec3 {
        let mapping = family.mapping(0.0);
        let (eye, pose) = (Pose::at(Vec4::ZERO), Pose::at(OFFSET));
        let relative = EuclideanR4.relative(&eye, &pose).expect("the eye places");
        let image = mapping
            .image_local(&EuclideanR4, &eye, &pose, &relative, local)
            .expect("the map places the vertex");
        Vec3::from_array(image)
    }

    #[test]
    fn the_perspective_map_foreshortens_the_vertex_and_leaves_the_body_where_it_stands() {
        let vertex = Vec4::splat(0.5) * BODY_SIZE;
        let scale = PERSPECTIVE_FOCAL / (PERSPECTIVE_FOCAL - vertex.w);
        let expected = Vec3::splat(vertex.x * scale) + OFFSET.truncate();
        let image = placed(Family::Perspective, vertex);
        assert!(
            (image - expected).length() < 1e-5,
            "the perspective map put the vertex at {image} rather than {expected}"
        );
    }

    #[test]
    fn the_stereographic_map_casts_the_vertex_onto_the_sphere_before_it_divides() {
        let vertex = Vec4::splat(0.5) * BODY_SIZE;
        let unit = vertex.normalize();
        let expected = unit.truncate() / (1.0 - unit.w) + OFFSET.truncate();
        let image = placed(Family::Stereographic, vertex);
        assert!(
            (image - expected).length() < 1e-5,
            "the stereographic map put the vertex at {image} rather than {expected}"
        );
    }

    #[test]
    fn stereographic_edges_follow_s3_arcs_and_stop_at_the_pole_clip() {
        let mapping = Family::Stereographic.mapping(0.0);
        let (eye, pose) = (Pose::at(Vec4::ZERO), Pose::at(Vec4::ZERO));
        let relative = EuclideanR4.relative(&eye, &pose).expect("the eye places");
        let mut samples = Vec::new();
        mapping.image_segment(
            &EuclideanR4,
            &eye,
            &pose,
            &relative,
            [Vec4::X, Vec4::Y],
            &mut |_, point| samples.push(point.expect("the arc stays away from the pole")),
        );

        assert_eq!(samples.len(), STEREOGRAPHIC_EDGE_SAMPLES + 1);
        let midpoint = Vec3::from_array(samples[STEREOGRAPHIC_EDGE_SAMPLES / 2]);
        let expected = Vec3::new(
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
            0.0,
        );
        assert!((midpoint - expected).length() < 1e-5);

        samples.clear();
        mapping.image_segment(
            &EuclideanR4,
            &eye,
            &pose,
            &relative,
            [Vec4::W, Vec4::X],
            &mut |_, point| samples.extend(point),
        );
        let lengths = samples
            .iter()
            .map(|point| Vec3::from_array(*point).length());
        let furthest = lengths.fold(0.0_f32, f32::max);
        assert!((furthest - STEREOGRAPHIC_CLIP_RADIUS).abs() < 1e-3);
        assert!(samples
            .iter()
            .all(|point| Vec3::from_array(*point).is_finite()));
    }
}
