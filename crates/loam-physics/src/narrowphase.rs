use std::collections::HashMap;

use crate::body::RigidBody;
use crate::collider::ColliderKind;
use crate::geometry::GeometryStore;
use crate::integrator::PhysicsSpace;
use crate::response::Contact;

/// Always called with `a.kind()` matching the key's first component.
pub type NarrowphaseFn<S> = fn(
    a: &RigidBody<S>,
    b: &RigidBody<S>,
    geometry: &GeometryStore,
    space: &S,
) -> Option<Contact<S>>;

fn kind_rank(kind: ColliderKind) -> u8 {
    match kind {
        ColliderKind::Sphere => 0,
        ColliderKind::HalfSpace => 1,
        ColliderKind::HalfSpace4D => 2,
        ColliderKind::Box3 => 3,
        ColliderKind::Polygon2D => 4,
        ColliderKind::ConvexPolytope3D => 5,
        ColliderKind::ConvexPolytope4D => 6,
        ColliderKind::HyperSphere4D => 7,
    }
}

pub struct Narrowphase<S: PhysicsSpace> {
    dispatch: HashMap<(ColliderKind, ColliderKind), NarrowphaseFn<S>>,
    order: Vec<(ColliderKind, ColliderKind)>,
}

impl<S: PhysicsSpace> Default for Narrowphase<S> {
    fn default() -> Self {
        Self {
            dispatch: HashMap::new(),
            order: Vec::new(),
        }
    }
}

impl<S: PhysicsSpace> Narrowphase<S> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, a: ColliderKind, b: ColliderKind, f: NarrowphaseFn<S>) {
        if self.dispatch.insert((a, b), f).is_some() {
            return;
        }
        let key = (kind_rank(a), kind_rank(b));
        let at = self
            .order
            .partition_point(|&(x, y)| (kind_rank(x), kind_rank(y)) < key);
        self.order.insert(at, (a, b));
    }

    pub fn registrations(&self) -> &[(ColliderKind, ColliderKind)] {
        &self.order
    }

    pub fn test(
        &self,
        a: &RigidBody<S>,
        b: &RigidBody<S>,
        geometry: &GeometryStore,
        space: &S,
    ) -> Option<Contact<S>>
    where
        S::Vector: std::ops::Mul<f32, Output = S::Vector>,
    {
        let key = (a.collider().kind(), b.collider().kind());
        if let Some(&f) = self.dispatch.get(&key) {
            return f(a, b, geometry, space);
        }
        let reversed = (b.collider().kind(), a.collider().kind());
        if let Some(&f) = self.dispatch.get(&reversed) {
            return f(b, a, geometry, space).map(|c| Contact {
                normal: c.normal * -1.0,
                point: c.point,
                penetration: c.penetration,
                restitution: c.restitution,
            });
        }
        None
    }
}
