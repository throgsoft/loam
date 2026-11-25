use loam_shape::polytope::Polytope4;

use super::{ChartPoint, ChartTangent, DomainError, DomainSpace, Pose, TypedDomain};
use crate::assets::PreparedGeometry;
use crate::entity::Entity;
use crate::image::{ImageRay, Views};
use crate::view::{self, DomainRay, Pick, Vec4, ViewId};

impl<S: DomainSpace> TypedDomain<S> {
    pub(super) fn pick_view(
        &self,
        view: ViewId,
        ray: &ImageRay,
        views: &Views,
        prepared: &[PreparedGeometry],
    ) -> Option<Pick> {
        let spec = self.views.get(view)?;
        let target = self.views.target(view)?;
        if !spec.style.enabled {
            return None;
        }
        let presented = self.blend.presented();
        let eye = &self.drawn(spec.eye, presented)?;
        let origin = self.space.origin();
        let lifted = spec.style.mapping.lift(eye, ray);
        let mut nearest: Option<Pick> = None;
        for (entity, instance) in self.instances.iter() {
            if spec.subject.is_some_and(|subject| subject != entity) {
                continue;
            }
            let Some(pose) = self.drawn(entity, presented) else {
                continue;
            };
            let Some(geometry) = prepared.get(instance.geometry.index()) else {
                continue;
            };
            let Ok(center) = self.space.place(&self.space.prepare(&pose), origin) else {
                continue;
            };
            let radius = geometry.bounding_radius();
            let hit = match &lifted {
                Some(domain_ray) => {
                    let reach = self.space.chart_reach(center, radius);
                    self.space
                        .hit_ball(domain_ray, center, reach)
                        .and_then(|entry| match geometry {
                            PreparedGeometry::Polytope4 { polytope, scale } => {
                                polytope_entry(&self.space, domain_ray, &pose, *polytope, *scale)
                            }
                            _ => Some(entry),
                        })
                        .map(|t| self.space.exp(domain_ray.origin, domain_ray.direction * t))
                        .and_then(|point| {
                            let image_point = spec.style.mapping.image_point(eye, point)?;
                            Some((image_point, Some(self.space.chart_point(point))))
                        })
                }
                None => spec
                    .style
                    .mapping
                    .image_point(eye, center)
                    .and_then(|image_center| {
                        let image_radius = spec.style.mapping.image_radius(eye, center, radius);
                        let t = view::image_hit(ray, image_center, image_radius)?;
                        Some((ray.at(t), None))
                    }),
            };
            let Some((image_point, hit)) = hit else {
                continue;
            };
            let Some(depth) = views.depth(image_point) else {
                continue;
            };
            if nearest.is_some_and(|best| best.depth >= depth) {
                continue;
            }
            nearest = Some(Pick {
                entity,
                domain: self.id,
                view,
                image: target.image,
                image_point,
                depth,
                hit,
            });
        }
        nearest
    }

    pub(super) fn image_in_view(&self, view: ViewId, entity: Entity) -> Option<[f32; 3]> {
        let spec = self.views.get(view)?;
        if !spec.style.enabled || spec.subject.is_some_and(|subject| subject != entity) {
            return None;
        }
        let eye = self.drawn(spec.eye, self.blend.presented())?;
        let pose = self.drawn(entity, self.blend.presented())?;
        let relative = self.space.relative(&eye, &pose).ok()?;
        spec.style
            .mapping
            .image_local(&self.space, &eye, &pose, &relative, self.space.origin())
    }

    pub(super) fn lift_view_origin(
        &self,
        view: ViewId,
        ray: &ImageRay,
    ) -> Result<ChartPoint, DomainError> {
        let spec = self.views.get(view).ok_or(DomainError::UnknownView(view))?;
        let eye = self
            .drawn(spec.eye, self.blend.presented())
            .ok_or(DomainError::Stale(spec.eye))?;
        let lifted = spec
            .style
            .mapping
            .lift(&eye, ray)
            .ok_or(DomainError::NoLift(spec.style.mapping.name()))?;
        self.space.check(lifted.origin)?;
        Ok(self.space.chart_point(lifted.origin))
    }

    pub(super) fn lift_view_tangent(
        &self,
        view: ViewId,
        at: [f32; 3],
        tangent: [f32; 3],
    ) -> Result<ChartTangent, DomainError> {
        let spec = self.views.get(view).ok_or(DomainError::UnknownView(view))?;
        let eye = self
            .drawn(spec.eye, self.blend.presented())
            .ok_or(DomainError::Stale(spec.eye))?;
        let lifted = spec
            .style
            .mapping
            .lift_tangent(&eye, at, tangent)
            .ok_or(DomainError::NoLift(spec.style.mapping.name()))?;
        Ok(self.space.chart_tangent(lifted))
    }
}

// Haines, "Fast Ray-Convex Polyhedron Intersection", Graphics Gems II (1991).
fn polytope_entry<S: DomainSpace>(
    space: &S,
    ray: &DomainRay<S>,
    pose: &Pose<S>,
    polytope: Polytope4,
    scale: f32,
) -> Option<f32> {
    let canonical = |point| {
        let local = space.local(pose, point).ok()?;
        Some(Vec4::from_array(space.chart_point(local).coordinates) / scale)
    };
    let origin = canonical(ray.origin)?;
    let direction = canonical(space.exp(ray.origin, ray.direction))? - origin;
    let topology = polytope.topology();
    let mut entry = 0.0f32;
    let mut exit = f32::INFINITY;
    for cell in topology.cells {
        let centroid = cell
            .iter()
            .map(|&index| topology.vertices[index as usize])
            .sum::<Vec4>()
            / cell.len() as f32;
        let offset = centroid.length();
        let normal = centroid / offset;
        let along = normal.dot(direction);
        let gap = offset - normal.dot(origin);
        if along == 0.0 {
            if gap < 0.0 {
                return None;
            }
            continue;
        }
        let t = gap / along;
        if along < 0.0 {
            entry = entry.max(t);
        } else {
            exit = exit.min(t);
        }
        if entry > exit {
            return None;
        }
    }
    Some(entry)
}
