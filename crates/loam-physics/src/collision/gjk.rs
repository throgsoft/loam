use glam::{Quat, Vec3};
use loam_shape::shape::{convex_polytope3, sphere};

pub trait SupportFn {
    fn support(&self, direction: Vec3) -> Vec3;
}

pub struct ConvexHull<'a> {
    pub vertices: &'a [Vec3],
}

impl<'a> SupportFn for ConvexHull<'a> {
    fn support(&self, direction: Vec3) -> Vec3 {
        convex_polytope3::support(self.vertices, direction)
    }
}

/// `rotation` must be a unit quaternion.
pub struct PosedHull<'a> {
    pub local: &'a [Vec3],
    pub position: Vec3,
    pub rotation: Quat,
}

impl SupportFn for PosedHull<'_> {
    fn support(&self, direction: Vec3) -> Vec3 {
        let local_direction = self.rotation.conjugate() * direction;
        let hull = ConvexHull {
            vertices: self.local,
        };
        self.rotation * hull.support(local_direction) + self.position
    }
}

pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl SupportFn for Sphere {
    fn support(&self, direction: Vec3) -> Vec3 {
        sphere::support(self.center, self.radius, direction)
    }
}

/// `sa` and `sb` are the contributing support points on A and B.
#[derive(Clone, Copy, Debug)]
pub struct MinkowskiPoint {
    pub point: Vec3,
    pub sa: Vec3,
    pub sb: Vec3,
}

pub fn minkowski_support<A: SupportFn, B: SupportFn>(
    a: &A,
    b: &B,
    direction: Vec3,
) -> MinkowskiPoint {
    let sa = a.support(direction);
    let sb = b.support(-direction);
    MinkowskiPoint {
        point: sa - sb,
        sa,
        sb,
    }
}

#[derive(Debug)]
pub enum GjkResult {
    Intersecting { simplex: [MinkowskiPoint; 4] },
    Separated,
}

const GJK_MAX_ITERATIONS: u32 = 32;
const GJK_EPS: f32 = 1e-6;

/// On intersection the returned tetrahedron is the seed EPA expects.
pub fn gjk_intersect<A: SupportFn, B: SupportFn>(
    a: &A,
    b: &B,
    initial_direction: Vec3,
) -> GjkResult {
    let mut dir = if initial_direction.length_squared() > GJK_EPS {
        initial_direction
    } else {
        Vec3::X
    };

    let mut simplex: [MinkowskiPoint; 4] = [MinkowskiPoint {
        point: Vec3::ZERO,
        sa: Vec3::ZERO,
        sb: Vec3::ZERO,
    }; 4];
    simplex[0] = minkowski_support(a, b, dir);
    let mut n = 1usize;
    dir = -simplex[0].point;

    for _ in 0..GJK_MAX_ITERATIONS {
        let new_point = minkowski_support(a, b, dir);
        if new_point.point.dot(dir) < 0.0 {
            return GjkResult::Separated;
        }

        simplex[n] = new_point;
        n += 1;

        let (contains_origin, new_n, new_dir) = do_simplex(&mut simplex, n);
        n = new_n;
        if contains_origin {
            return GjkResult::Intersecting { simplex };
        }
        if new_dir.length_squared() < GJK_EPS {
            if n >= 4 {
                return GjkResult::Intersecting { simplex };
            }
            return GjkResult::Separated;
        }
        dir = new_dir;
    }

    GjkResult::Separated
}

// Newest point at `n-1` on entry; `simplex[0..new_n]` holds survivors on exit.
fn do_simplex(simplex: &mut [MinkowskiPoint; 4], n: usize) -> (bool, usize, Vec3) {
    match n {
        2 => do_line(simplex),
        3 => do_triangle(simplex),
        4 => do_tetrahedron(simplex),
        _ => unreachable!("simplex size {n} out of range"),
    }
}

fn do_line(simplex: &mut [MinkowskiPoint; 4]) -> (bool, usize, Vec3) {
    let a = simplex[1].point;
    let b = simplex[0].point;
    let ab = b - a;
    let ao = -a;

    if ab.dot(ao) > 0.0 {
        let dir = triple_product(ab, ao, ab);
        if dir.length_squared() < 1e-10 {
            (false, 2, any_perpendicular(ab))
        } else {
            (false, 2, dir)
        }
    } else {
        simplex[0] = simplex[1];
        (false, 1, ao)
    }
}

fn any_perpendicular(v: Vec3) -> Vec3 {
    if v.x.abs() <= v.y.abs() && v.x.abs() <= v.z.abs() {
        v.cross(Vec3::X)
    } else if v.y.abs() <= v.z.abs() {
        v.cross(Vec3::Y)
    } else {
        v.cross(Vec3::Z)
    }
}

fn fall_back_to_ab(simplex: &mut [MinkowskiPoint; 4]) -> (bool, usize, Vec3) {
    simplex[0] = simplex[1];
    simplex[1] = simplex[2];
    do_line(simplex)
}

fn do_triangle(simplex: &mut [MinkowskiPoint; 4]) -> (bool, usize, Vec3) {
    let a = simplex[2].point;
    let b = simplex[1].point;
    let c = simplex[0].point;

    let ab = b - a;
    let ac = c - a;
    let ao = -a;
    let abc = ab.cross(ac);

    if abc.cross(ac).dot(ao) > 0.0 {
        if ac.dot(ao) > 0.0 {
            simplex[1] = simplex[2];
            let dir = triple_product(ac, ao, ac);
            return (false, 2, dir);
        }
        return fall_back_to_ab(simplex);
    }

    if ab.cross(abc).dot(ao) > 0.0 {
        return fall_back_to_ab(simplex);
    }

    if abc.dot(ao) > 0.0 {
        (false, 3, abc)
    } else {
        simplex.swap(0, 1);
        (false, 3, -abc)
    }
}

// Face normals use the opposite vertex; simplex pruning can change winding.
fn do_tetrahedron(simplex: &mut [MinkowskiPoint; 4]) -> (bool, usize, Vec3) {
    let a = simplex[3].point;
    let b = simplex[2].point;
    let c = simplex[1].point;
    let d = simplex[0].point;

    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    let ao = -a;

    let mut abc = ab.cross(ac);
    if abc.dot(ad) > 0.0 {
        abc = -abc;
    }
    let mut acd = ac.cross(ad);
    if acd.dot(ab) > 0.0 {
        acd = -acd;
    }
    let mut adb = ad.cross(ab);
    if adb.dot(ac) > 0.0 {
        adb = -adb;
    }

    if abc.dot(ao) > 0.0 {
        simplex[0] = simplex[1];
        simplex[1] = simplex[2];
        simplex[2] = simplex[3];
        return do_triangle(simplex);
    }
    if acd.dot(ao) > 0.0 {
        simplex[2] = simplex[3];
        return do_triangle(simplex);
    }
    if adb.dot(ao) > 0.0 {
        let d_point = simplex[0];
        simplex[0] = simplex[2];
        simplex[1] = d_point;
        simplex[2] = simplex[3];
        return do_triangle(simplex);
    }

    (true, 4, Vec3::ZERO)
}

fn triple_product(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    a.cross(b).cross(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posed_support() {
        let local = [
            Vec3::new(0.6, 0.1, -0.2),
            Vec3::new(-0.4, 0.7, 0.15),
            Vec3::new(0.05, -0.8, 0.45),
        ];
        let position = Vec3::new(1.0, -2.0, 0.5);
        let rotation = Quat::from_rotation_z(0.7) * Quat::from_rotation_x(-0.3);
        let vertices = local.map(|v| rotation * v + position);
        let posed = PosedHull {
            local: &local,
            position,
            rotation,
        };
        let world = ConvexHull {
            vertices: &vertices,
        };
        for direction in [Vec3::X, -Vec3::Y, Vec3::Z, Vec3::new(0.3, -0.7, 0.9)] {
            assert!((posed.support(direction) - world.support(direction)).length() < 1e-6);
        }
    }
}
