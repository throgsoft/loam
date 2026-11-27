//! Narrowphase pairs shared by the flat spaces, where a point and a tangent vector are the same type.

use std::ops::Div;

use loam_math::Space;
use loam_shape::{Shape, ShapeKind};

use crate::body::RigidBody;
use crate::collision::{ContactInfo, SupportFn};
use crate::geometry::GeometryStore;
use crate::narrowphase::Narrowphase;
use crate::response::Contact;
use crate::space::{PhysicsSpace, VectorOps};

pub(crate) trait Euclidean:
    PhysicsSpace
    + Space<Point = <Self as Space>::Vector, Vector: Div<f32, Output = <Self as Space>::Vector>>
{
    const POLYTOPE: ShapeKind;
    const HALFSPACE: ShapeKind;
    const COINCIDENT_NORMAL: Self::Vector;

    type Hull<'a>: SupportFn<Self::Vector>;
    type Ball: SupportFn<Self::Vector>;

    fn polytope(shape: &Shape) -> Option<&[Self::Vector]>;

    fn halfspace(shape: &Shape) -> Option<(Self::Vector, f32)>;

    fn bounding_radius(vertices: &[Self::Vector]) -> f32;

    fn rotate(orientation: Self::Iso, v: Self::Vector) -> Self::Vector;

    fn hull(
        local: &[Self::Vector],
        position: Self::Vector,
        orientation: Self::Iso,
    ) -> Self::Hull<'_>;

    fn ball(center: Self::Vector, radius: f32) -> Self::Ball;

    // `scale` is the sum of the two bounding radii.
    fn penetration<A: SupportFn<Self::Vector>, B: SupportFn<Self::Vector>>(
        a: &A,
        b: &B,
        initial_direction: Self::Vector,
        scale: f32,
    ) -> Option<ContactInfo<Self::Vector>>;
}

pub(crate) fn register_pairs<S: Euclidean>(np: &mut Narrowphase<S>) {
    np.register(ShapeKind::Sphere, ShapeKind::Sphere, sphere_sphere::<S>);
    np.register(ShapeKind::Sphere, S::HALFSPACE, sphere_halfspace::<S>);
    np.register(S::POLYTOPE, S::POLYTOPE, polytope_polytope::<S>);
    np.register(ShapeKind::Sphere, S::POLYTOPE, sphere_polytope::<S>);
    np.register(S::POLYTOPE, S::HALFSPACE, polytope_halfspace::<S>);
}

fn sphere_sphere<S: Euclidean>(
    a: &RigidBody<S>,
    b: &RigidBody<S>,
    geometry: &GeometryStore,
    space: &S,
) -> Option<Contact<S>> {
    let Some(&Shape::Sphere { radius: ra, .. }) = geometry.get(a.collider()) else {
        return None;
    };
    let Some(&Shape::Sphere { radius: rb, .. }) = geometry.get(b.collider()) else {
        return None;
    };

    let d = space.distance(a.position, b.position);
    let combined = ra + rb;
    if d >= combined {
        return None;
    }
    let log = space.log(a.position, b.position);
    let len = log.length();
    let normal = if len > 1e-8 {
        log / len
    } else {
        S::COINCIDENT_NORMAL
    };

    let surface_a = a.position + normal * ra;
    let surface_b = b.position - normal * rb;
    let point = (surface_a + surface_b) * 0.5;

    Some(Contact {
        normal,
        point,
        penetration: combined - d,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

fn sphere_halfspace<S: Euclidean>(
    a: &RigidBody<S>,
    b: &RigidBody<S>,
    geometry: &GeometryStore,
    _space: &S,
) -> Option<Contact<S>> {
    let Some(&Shape::Sphere { radius, .. }) = geometry.get(a.collider()) else {
        return None;
    };
    let (normal, offset) = geometry.get(b.collider()).and_then(S::halfspace)?;
    let normal = S::rotate(b.orientation, normal);
    let offset = offset + normal.dot(b.position);
    let signed = a.position.dot(normal) - offset;
    let penetration = radius - signed;
    if penetration <= 0.0 {
        return None;
    }
    let contact_normal = -normal;
    let point = a.position - normal * radius;
    Some(Contact {
        normal: contact_normal,
        point,
        penetration,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

fn polytope_halfspace<S: Euclidean>(
    a: &RigidBody<S>,
    b: &RigidBody<S>,
    geometry: &GeometryStore,
    _space: &S,
) -> Option<Contact<S>> {
    let va_local = geometry.get(a.collider()).and_then(S::polytope)?;
    let (plane_n, offset) = geometry.get(b.collider()).and_then(S::halfspace)?;
    let plane_n = S::rotate(b.orientation, plane_n);
    let offset = offset + plane_n.dot(b.position);

    let mut deepest = <S::Vector as VectorOps>::zero();
    let mut deepest_depth = 0.0_f32;
    for &v_local in va_local {
        let v_world = S::rotate(a.orientation, v_local) + a.position;
        let signed = v_world.dot(plane_n) - offset;
        let depth = -signed;
        if depth > deepest_depth {
            deepest_depth = depth;
            deepest = v_world;
        }
    }
    if deepest_depth <= 0.0 {
        return None;
    }
    Some(Contact {
        normal: -plane_n,
        point: deepest,
        penetration: deepest_depth,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

fn validate_contact<S: Euclidean>(
    info: &ContactInfo<S::Vector>,
    a: &RigidBody<S>,
    b: &RigidBody<S>,
) -> Option<Contact<S>> {
    if !info.penetration.is_finite()
        || info.penetration <= 0.0
        || !info.normal.is_finite()
        || !info.point.is_finite()
    {
        return None;
    }
    let n2 = info.normal.length_squared();
    if !(0.5..=1.5).contains(&n2) {
        return None;
    }
    Some(Contact {
        normal: info.normal,
        point: info.point,
        penetration: info.penetration,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

fn polytope_polytope<S: Euclidean>(
    a: &RigidBody<S>,
    b: &RigidBody<S>,
    geometry: &GeometryStore,
    _space: &S,
) -> Option<Contact<S>> {
    let va_local = geometry.get(a.collider()).and_then(S::polytope)?;
    let vb_local = geometry.get(b.collider()).and_then(S::polytope)?;

    let ra = S::bounding_radius(va_local);
    let rb = S::bounding_radius(vb_local);
    let center_dist_sq = (b.position - a.position).length_squared();
    let combined = ra + rb;
    if center_dist_sq > combined * combined {
        return None;
    }

    let hull_a = S::hull(va_local, a.position, a.orientation);
    let hull_b = S::hull(vb_local, b.position, b.orientation);
    let initial_dir = b.position - a.position;
    let info = S::penetration(&hull_a, &hull_b, initial_dir, combined)?;
    validate_contact(&info, a, b)
}

fn sphere_polytope<S: Euclidean>(
    a: &RigidBody<S>,
    b: &RigidBody<S>,
    geometry: &GeometryStore,
    _space: &S,
) -> Option<Contact<S>> {
    let Some(&Shape::Sphere { radius, .. }) = geometry.get(a.collider()) else {
        return None;
    };
    let vb_local = geometry.get(b.collider()).and_then(S::polytope)?;

    let rb = S::bounding_radius(vb_local);
    let center_dist_sq = (b.position - a.position).length_squared();
    let combined = radius + rb;
    if center_dist_sq > combined * combined {
        return None;
    }

    let support_a = S::ball(a.position, radius);
    let support_b = S::hull(vb_local, b.position, b.orientation);
    let initial_dir = b.position - a.position;
    let info = S::penetration(&support_a, &support_b, initial_dir, combined)?;
    validate_contact(&info, a, b)
}
