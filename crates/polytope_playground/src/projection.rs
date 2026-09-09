use glam::{Vec3, Vec4};
use loam_math::{EuclideanR4, Projection, RasterizableSpace};
use loam_runtime::{
    DepthEnvelope, DomainRay, DomainSpace, ImageRay, Pose, SectionCut, ViewMapping,
};
use loam_shape::polytope::Polytope4;
use loam_shape::projection::SchlegelParams;

pub(crate) const STEREOGRAPHIC_POLE: Vec4 = Vec4::W;

pub(crate) const PERSPECTIVE_FOCAL: f32 = 2.0;

const SCHLEGEL_EYE_MARGIN: f32 = 0.5;

const UNBOUNDED: DepthEnvelope = DepthEnvelope {
    near: 0.0,
    far: f32::INFINITY,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Family {
    #[default]
    Perspective,
    Stereographic,
    Schlegel,
}

impl Family {
    pub(crate) const ALL: [Family; 3] =
        [Family::Perspective, Family::Stereographic, Family::Schlegel];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Family::Perspective => "perspective",
            Family::Stereographic => "stereographic",
            Family::Schlegel => "schlegel",
        }
    }

    pub(crate) fn from_token(token: &str) -> Option<Self> {
        Family::ALL
            .into_iter()
            .find(|family| family.name() == token)
    }

    pub(crate) fn mapping(self, subject: Option<Polytope4>, cell: u32, slice: f32) -> Projected {
        let projection = match self {
            Family::Perspective => Projection::Perspective4D {
                focal_distance: PERSPECTIVE_FOCAL,
            },
            Family::Stereographic => Projection::Stereographic {
                pole: STEREOGRAPHIC_POLE,
            },
            Family::Schlegel => subject
                .and_then(|polytope| schlegel_params(polytope, cell))
                .map_or(Projection::Identity, |params| {
                    Projection::schlegel_with_basis(
                        params.cell_normal,
                        params.cell_offset,
                        params.viewpoint_distance,
                        params.cell_basis,
                    )
                }),
        };
        Projected {
            name: self.name(),
            projection,
            slice,
        }
    }
}

pub(crate) fn schlegel_params(polytope: Polytope4, cell: u32) -> Option<SchlegelParams> {
    let selected = cell.min(polytope.cell_count().saturating_sub(1) as u32);
    SchlegelParams::new(polytope, selected, SCHLEGEL_EYE_MARGIN)
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
        EuclideanR4::project_point(point, &self.projection)
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
        let placed = self.project(point - origin) + origin.truncate();
        placed.is_finite().then(|| placed.to_array())
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
        let mapping = family.mapping(Some(Polytope4::Tesseract), 0, 0.0);
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
    fn the_schlegel_map_fixes_the_vertices_of_the_cell_it_projects_onto() {
        let params = schlegel_params(Polytope4::Tesseract, 0).expect("the tesseract has cells");
        let topology = Polytope4::Tesseract.topology();
        for &index in topology.cells[params.cell_index as usize] {
            let vertex = topology.vertices[index as usize];
            let expected = Vec3::new(
                vertex.dot(params.cell_basis[0]),
                vertex.dot(params.cell_basis[1]),
                vertex.dot(params.cell_basis[2]),
            ) + OFFSET.truncate();
            let image = placed(Family::Schlegel, vertex);
            assert!(
                (image - expected).length() < 1e-4,
                "the boundary vertex {vertex} moved to {image} rather than staying at {expected}"
            );
        }
    }
}
