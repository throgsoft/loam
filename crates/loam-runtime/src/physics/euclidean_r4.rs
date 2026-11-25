use loam_math::EuclideanR4;
use loam_physics::euclidean_r4::{
    flush_halfspace4_body_r4, register_default_narrowphase, regular_polytope_body_r4,
};
use loam_physics::{BodyDef, EditError};

use super::PhysicsConfig;
use crate::app_state::Stores;
use crate::assets::{PreparedGeometry, PreparedId};
use crate::command::{Dispatch, Rejection, SpawnBundle};
use crate::domain::{DomainHandle, Pose};
use crate::entity::Entity;
use crate::image::Backdrop;
use crate::view::Vec4;

impl Default for PhysicsConfig<EuclideanR4> {
    fn default() -> Self {
        Self::new(register_default_narrowphase)
    }
}

impl<A: Stores> Dispatch<'_, A> {
    /// The uniform-density body of a prepared `Polytope4` at its prepared scale.
    pub fn body_of(
        &self,
        geometry: PreparedId,
        mass: f32,
    ) -> Result<BodyDef<EuclideanR4>, Rejection> {
        match self.prepared(geometry) {
            Some(&PreparedGeometry::Polytope4 { polytope, scale }) => {
                regular_polytope_body_r4(polytope, scale, mass)
                    .ok_or(Rejection::Edit(EditError::InvalidBody))
            }
            _ => Err(Rejection::Unsupported("a body needs polytope geometry")),
        }
    }

    /// Spawns a flush half-space at the backdrop's ground height, then sets the root backdrop.
    pub fn floor(
        &mut self,
        domain: DomainHandle<EuclideanR4>,
        backdrop: Backdrop,
    ) -> Result<Entity, Rejection> {
        let body =
            flush_halfspace4_body_r4(Vec4::Y, backdrop.ground.y).ok_or(EditError::NotFinite)?;
        let floor = self.spawn(
            SpawnBundle::new()
                .at(domain, Pose::at(Vec4::ZERO))
                .body(body),
        )?;
        self.views.root_mut().backdrop = Some(backdrop);
        Ok(floor)
    }
}
