use crate::space::VectorOps;

pub trait SupportFn<V> {
    fn support(&self, direction: V) -> V;
}

pub struct ConvexHull<'a, V> {
    pub vertices: &'a [V],
}

/// `rotation` must be unit.
pub struct PosedHull<'a, V, R> {
    pub local: &'a [V],
    pub position: V,
    pub rotation: R,
}

pub struct Sphere<V> {
    pub center: V,
    pub radius: f32,
}

/// `sa` and `sb` are the contributing support points on A and B.
#[derive(Clone, Copy, Debug)]
pub struct MinkowskiPoint<V> {
    pub point: V,
    pub sa: V,
    pub sb: V,
}

pub fn minkowski_support<V: VectorOps, A: SupportFn<V>, B: SupportFn<V>>(
    a: &A,
    b: &B,
    direction: V,
) -> MinkowskiPoint<V> {
    let sa = a.support(direction);
    let sb = b.support(-direction);
    MinkowskiPoint {
        point: sa - sb,
        sa,
        sb,
    }
}

/// On intersection the simplex spans the space and is the seed EPA expects.
#[derive(Debug)]
pub enum GjkResult<V, const N: usize> {
    Intersecting { simplex: [MinkowskiPoint<V>; N] },
    Separated,
}

#[derive(Clone, Copy, Debug)]
pub struct ContactInfo<V> {
    /// Unit, world, from A toward B.
    pub normal: V,
    pub penetration: f32,
    /// Midpoint of the closest features on A and B, in world space.
    pub point: V,
}
