//! `R = exp(B/2)`, so a bivector of magnitude θ in plane `e_i ∧ e_j` rotates
//! by θ from `e_i` toward `e_j`. Rotor application is the sandwich
//! `v' = R̃ · v · R`.

use std::ops::{Add, Mul};

use glam::{Vec2, Vec3, Vec4};

pub trait Bivector: Copy + Add<Output = Self> + Mul<f32, Output = Self> {
    type Rotor: Rotor<Bivector = Self>;

    fn zero() -> Self;

    fn exp(self) -> Self::Rotor;

    fn magnitude(self) -> f32;
}

/// Rotor operations require unit elements of Spin(N); public fields do not enforce this.
pub trait Rotor: Copy + Mul<Output = Self> {
    type Bivector: Bivector<Rotor = Self>;

    type Vector: Copy;

    fn identity() -> Self;

    fn inverse(self) -> Self;

    fn apply(&self, v: Self::Vector) -> Self::Vector;

    /// Inverse of [`Bivector::exp`]; [`Rotor4`] takes the shorter rotation, so its result may generate `−self`.
    fn log(self) -> Self::Bivector;
}

#[derive(Copy, Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Bivector2(pub f32);

impl Add for Bivector2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Mul<f32> for Bivector2 {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self(self.0 * k)
    }
}

impl Bivector for Bivector2 {
    type Rotor = Rotor2;

    fn zero() -> Self {
        Self(0.0)
    }

    fn magnitude(self) -> f32 {
        self.0.abs()
    }

    fn exp(self) -> Rotor2 {
        let half = self.0 * 0.5;
        Rotor2 {
            a: half.cos(),
            b: half.sin(),
        }
    }
}

/// Unit complex number `a + b·e1e2` with `a = cos(θ/2)`, `b = sin(θ/2)`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rotor2 {
    pub a: f32,
    pub b: f32,
}

impl Rotor2 {
    pub const IDENTITY: Self = Self { a: 1.0, b: 0.0 };
}

impl Default for Rotor2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mul for Rotor2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            a: self.a * rhs.a - self.b * rhs.b,
            b: self.a * rhs.b + self.b * rhs.a,
        }
    }
}

impl Rotor for Rotor2 {
    type Bivector = Bivector2;
    type Vector = Vec2;

    fn identity() -> Self {
        Self::IDENTITY
    }

    fn inverse(self) -> Self {
        Self {
            a: self.a,
            b: -self.b,
        }
    }

    fn apply(&self, v: Vec2) -> Vec2 {
        let c = self.a * self.a - self.b * self.b;
        let s = 2.0 * self.a * self.b;
        Vec2::new(c * v.x - s * v.y, s * v.x + c * v.y)
    }

    fn log(self) -> Bivector2 {
        Bivector2(2.0 * self.b.atan2(self.a))
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Bivector3 {
    pub xy: f32,
    pub yz: f32,
    pub zx: f32,
}

impl Bivector3 {
    pub const ZERO: Self = Self {
        xy: 0.0,
        yz: 0.0,
        zx: 0.0,
    };

    pub fn new(xy: f32, yz: f32, zx: f32) -> Self {
        Self { xy, yz, zx }
    }

    pub fn magnitude(self) -> f32 {
        (self.xy * self.xy + self.yz * self.yz + self.zx * self.zx).sqrt()
    }
}

impl Add for Bivector3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            xy: self.xy + rhs.xy,
            yz: self.yz + rhs.yz,
            zx: self.zx + rhs.zx,
        }
    }
}

impl Mul<f32> for Bivector3 {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self {
            xy: self.xy * k,
            yz: self.yz * k,
            zx: self.zx * k,
        }
    }
}

impl Bivector for Bivector3 {
    type Rotor = Rotor3;

    fn zero() -> Self {
        Self::ZERO
    }

    fn magnitude(self) -> f32 {
        Bivector3::magnitude(self)
    }

    fn exp(self) -> Rotor3 {
        let mag_sq = self.xy * self.xy + self.yz * self.yz + self.zx * self.zx;

        if mag_sq < 1e-16 {
            return Rotor3 {
                s: 1.0,
                xy: self.xy * 0.5,
                yz: self.yz * 0.5,
                zx: self.zx * 0.5,
            };
        }
        let mag = mag_sq.sqrt();
        let half = mag * 0.5;
        let c = half.cos();
        let k = half.sin() / mag;
        Rotor3 {
            s: c,
            xy: self.xy * k,
            yz: self.yz * k,
            zx: self.zx * k,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rotor3 {
    pub s: f32,
    pub xy: f32,
    pub yz: f32,
    pub zx: f32,
}

impl Rotor3 {
    pub const IDENTITY: Self = Self {
        s: 1.0,
        xy: 0.0,
        yz: 0.0,
        zx: 0.0,
    };
}

impl Default for Rotor3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mul for Rotor3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let (s1, a1, b1, c1) = (self.s, self.xy, self.yz, self.zx);
        let (s2, a2, b2, c2) = (rhs.s, rhs.xy, rhs.yz, rhs.zx);
        Self {
            s: s1 * s2 - a1 * a2 - b1 * b2 - c1 * c2,
            xy: s1 * a2 + s2 * a1 - b1 * c2 + c1 * b2,
            yz: s1 * b2 + s2 * b1 + a1 * c2 - c1 * a2,
            zx: s1 * c2 + s2 * c1 - a1 * b2 + b1 * a2,
        }
    }
}

impl Rotor for Rotor3 {
    type Bivector = Bivector3;
    type Vector = Vec3;

    fn identity() -> Self {
        Self::IDENTITY
    }

    fn inverse(self) -> Self {
        Self {
            s: self.s,
            xy: -self.xy,
            yz: -self.yz,
            zx: -self.zx,
        }
    }

    fn apply(&self, v: Vec3) -> Vec3 {
        let (s, a, b, c) = (self.s, self.xy, self.yz, self.zx);
        let (vx, vy, vz) = (v.x, v.y, v.z);

        let p1 = s * vx - a * vy + c * vz;
        let p2 = s * vy + a * vx - b * vz;
        let p3 = s * vz + b * vy - c * vx;
        let pt = -(a * vz + b * vx + c * vy);

        Vec3::new(
            p1 * s - p2 * a + p3 * c - pt * b,
            p2 * s + p1 * a - p3 * b - pt * c,
            p3 * s + p2 * b - p1 * c - pt * a,
        )
    }

    fn log(self) -> Bivector3 {
        let mag_sq = self.xy * self.xy + self.yz * self.yz + self.zx * self.zx;
        if mag_sq < 1e-16 {
            return Bivector3 {
                xy: self.xy * 2.0,
                yz: self.yz * 2.0,
                zx: self.zx * 2.0,
            };
        }
        let mag = mag_sq.sqrt();
        let theta = 2.0 * mag.atan2(self.s);
        let k = theta / mag;
        Bivector3 {
            xy: self.xy * k,
            yz: self.yz * k,
            zx: self.zx * k,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Bivector4 {
    pub xy: f32,
    pub xz: f32,
    pub xw: f32,
    pub yz: f32,
    pub yw: f32,
    pub zw: f32,
}

impl Bivector4 {
    pub const ZERO: Self = Self {
        xy: 0.0,
        xz: 0.0,
        xw: 0.0,
        yz: 0.0,
        yw: 0.0,
        zw: 0.0,
    };

    pub fn new(xy: f32, xz: f32, xw: f32, yz: f32, yw: f32, zw: f32) -> Self {
        Self {
            xy,
            xz,
            xw,
            yz,
            yw,
            zw,
        }
    }

    /// Panics if `i >= 6`; [`Plane4::unit_bivector`] is the typed form.
    pub fn basis(i: usize) -> Self {
        let mut c = [0.0_f32; 6];
        c[i] = 1.0;
        Self::new(c[0], c[1], c[2], c[3], c[4], c[5])
    }

    pub fn magnitude_squared(self) -> f32 {
        self.xy * self.xy
            + self.xz * self.xz
            + self.xw * self.xw
            + self.yz * self.yz
            + self.yw * self.yw
            + self.zw * self.zw
    }

    /// `sqrt(θ₁² + θ₂²)` over the invariant planes; the angle only when simple.
    pub fn magnitude(self) -> f32 {
        self.magnitude_squared().sqrt()
    }

    pub fn component(self, plane: Plane4) -> f32 {
        match plane {
            Plane4::Xy => self.xy,
            Plane4::Xz => self.xz,
            Plane4::Xw => self.xw,
            Plane4::Yz => self.yz,
            Plane4::Yw => self.yw,
            Plane4::Zw => self.zw,
        }
    }

    pub fn set_component(&mut self, plane: Plane4, value: f32) {
        match plane {
            Plane4::Xy => self.xy = value,
            Plane4::Xz => self.xz = value,
            Plane4::Xw => self.xw = value,
            Plane4::Yz => self.yz = value,
            Plane4::Yw => self.yw = value,
            Plane4::Zw => self.zw = value,
        }
    }

    /// Euclidean product of the six coefficients, not the Clifford scalar part.
    pub fn dot(self, other: Self) -> f32 {
        self.xy * other.xy
            + self.xz * other.xz
            + self.xw * other.xw
            + self.yz * other.yz
            + self.yw * other.yw
            + self.zw * other.zw
    }

    /// Pseudoscalar coefficient of `B ∧ B`; zero iff `B` is simple.
    pub fn wedge_self_coeff(self) -> f32 {
        2.0 * (self.xy * self.zw - self.xz * self.yw + self.xw * self.yz)
    }

    pub fn wedge(u: Vec4, v: Vec4) -> Self {
        Self {
            xy: u.x * v.y - u.y * v.x,
            xz: u.x * v.z - u.z * v.x,
            xw: u.x * v.w - u.w * v.x,
            yz: u.y * v.z - u.z * v.y,
            yw: u.y * v.w - u.w * v.y,
            zw: u.z * v.w - u.w * v.z,
        }
    }

    /// Left contraction `B ⌋ v`, the grade-1 part of `B·v`: `e_xy ⌋ e_x = −e_y`.
    pub fn contract_vec(self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.xy * v.y + self.xz * v.z + self.xw * v.w,
            -self.xy * v.x + self.yz * v.z + self.yw * v.w,
            -self.xz * v.x - self.yz * v.y + self.zw * v.w,
            -self.xw * v.x - self.yw * v.y - self.zw * v.z,
        )
    }

    pub fn dual(self) -> Self {
        Self {
            xy: -self.zw,
            xz: self.yw,
            xw: -self.yz,
            yz: -self.xw,
            yw: self.xz,
            zw: -self.xy,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Plane4 {
    Xy = 0,
    Xz = 1,
    Xw = 2,
    Yz = 3,
    Yw = 4,
    Zw = 5,
}

impl Plane4 {
    pub const ALL: [Self; 6] = [Self::Xy, Self::Xz, Self::Xw, Self::Yz, Self::Yw, Self::Zw];

    /// Lowercase axis pair, such as `"xy"`.
    pub fn label(self) -> &'static str {
        match self {
            Self::Xy => "xy",
            Self::Xz => "xz",
            Self::Xw => "xw",
            Self::Yz => "yz",
            Self::Yw => "yw",
            Self::Zw => "zw",
        }
    }

    pub fn unit_bivector(self) -> Bivector4 {
        Bivector4::basis(self as usize)
    }
}

impl Add for Bivector4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            xy: self.xy + rhs.xy,
            xz: self.xz + rhs.xz,
            xw: self.xw + rhs.xw,
            yz: self.yz + rhs.yz,
            yw: self.yw + rhs.yw,
            zw: self.zw + rhs.zw,
        }
    }
}

impl Mul<f32> for Bivector4 {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self {
            xy: self.xy * k,
            xz: self.xz * k,
            xw: self.xw * k,
            yz: self.yz * k,
            yw: self.yw * k,
            zw: self.zw * k,
        }
    }
}

impl Bivector for Bivector4 {
    type Rotor = Rotor4;

    fn zero() -> Self {
        Self::ZERO
    }

    fn magnitude(self) -> f32 {
        Bivector4::magnitude(self)
    }

    fn exp(self) -> Rotor4 {
        let s = self.magnitude_squared();
        if s < 1e-16 {
            return Rotor4::IDENTITY;
        }
        let delta = self.wedge_self_coeff();

        if delta.abs() < 1e-6 * s.max(1.0) {
            let mag = s.sqrt();
            let half = mag * 0.5;
            let c = half.cos();
            let k = half.sin() / mag;
            return Rotor4 {
                s: c,
                xy: self.xy * k,
                xz: self.xz * k,
                xw: self.xw * k,
                yz: self.yz * k,
                yw: self.yw * k,
                zw: self.zw * k,
                xyzw: 0.0,
            };
        }

        let disc_sq = (s * s - delta * delta).max(0.0);
        let disc = disc_sq.sqrt();

        if disc < 1e-6 * s.max(1.0) {
            let theta_sq = s * 0.5;
            let theta = theta_sq.sqrt();
            let half = theta * 0.5;
            let ch = half.cos();
            let sh = half.sin();
            let sign_i = delta.signum();

            let b_coef = if theta > 1e-8 {
                (theta.sin()) / (2.0 * theta)
            } else {
                0.5
            };
            return Rotor4 {
                s: ch * ch,
                xy: self.xy * b_coef,
                xz: self.xz * b_coef,
                xw: self.xw * b_coef,
                yz: self.yz * b_coef,
                yw: self.yw * b_coef,
                zw: self.zw * b_coef,
                xyzw: sh * sh * sign_i,
            };
        }

        // Press et al., Numerical Recipes, 3rd ed., §5.6.
        let t1 = ((s + disc) * 0.5).max(0.0).sqrt();
        let t2 = delta / (2.0 * t1);

        let half1 = t1 * 0.5;
        let half2 = t2 * 0.5;
        let c1 = half1.cos();
        let s1 = half1.sin();
        let c2 = half2.cos();
        let s2 = half2.sin();

        let s1_t1 = s1 / t1;
        let s2_t2 = s2 / t2;

        let b_coef = (s1_t1 * c2 * (s + disc) - c1 * s2_t2 * (s - disc)) / (2.0 * disc);
        let bstar_coef = delta * (s1_t1 * c2 - c1 * s2_t2) / (2.0 * disc);

        let dual = self.dual();
        Rotor4 {
            s: c1 * c2,
            xy: self.xy * b_coef + dual.xy * bstar_coef,
            xz: self.xz * b_coef + dual.xz * bstar_coef,
            xw: self.xw * b_coef + dual.xw * bstar_coef,
            yz: self.yz * b_coef + dual.yz * bstar_coef,
            yw: self.yw * b_coef + dual.yw * bstar_coef,
            zw: self.zw * b_coef + dual.zw * bstar_coef,
            xyzw: s1 * s2,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Rotor4 {
    pub s: f32,
    pub xy: f32,
    pub xz: f32,
    pub xw: f32,
    pub yz: f32,
    pub yw: f32,
    pub zw: f32,
    pub xyzw: f32,
}

impl Rotor4 {
    /// Shortest simple rotation between unit vectors; antipodes use the least-aligned coordinate axis.
    pub fn from_rotation_arc(from: Vec4, to: Vec4) -> Self {
        let plane = Bivector4::wedge(from, to);
        let magnitude = plane.magnitude();
        if magnitude < ARC_PLANE_EPSILON {
            if from.dot(to) > 0.0 {
                return Self::IDENTITY;
            }
            let axes = [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W];
            let least = (1..4).fold(0, |best, i| {
                if from[i].abs() < from[best].abs() {
                    i
                } else {
                    best
                }
            });
            let axis = (axes[least] - from * from[least]).normalize();
            return (Bivector4::wedge(from, axis) * std::f32::consts::PI)
                .exp()
                .normalize();
        }
        // Kahan, How Futile are Mindless Assessments of Roundoff?, 2006, Mangled Angles.
        let angle = 2.0 * (to - from).length().atan2((to + from).length());
        (plane * (angle / magnitude)).exp().normalize()
    }

    pub const IDENTITY: Self = Self {
        s: 1.0,
        xy: 0.0,
        xz: 0.0,
        xw: 0.0,
        yz: 0.0,
        yw: 0.0,
        zw: 0.0,
        xyzw: 0.0,
    };

    pub fn norm_squared(self) -> f32 {
        self.s * self.s
            + self.xy * self.xy
            + self.xz * self.xz
            + self.xw * self.xw
            + self.yz * self.yz
            + self.yw * self.yw
            + self.zw * self.zw
            + self.xyzw * self.xyzw
    }

    /// Rescales a nonzero scalar multiple of a rotor; arbitrary even elements need not lie in Spin(4).
    pub fn normalize(self) -> Self {
        let n = self.norm_squared().sqrt();
        if n > 0.0 {
            let k = 1.0 / n;
            Self {
                s: self.s * k,
                xy: self.xy * k,
                xz: self.xz * k,
                xw: self.xw * k,
                yz: self.yz * k,
                yw: self.yw * k,
                zw: self.zw * k,
                xyzw: self.xyzw * k,
            }
        } else {
            Self::IDENTITY
        }
    }
}

impl Default for Rotor4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

// Slot order is the GPU uniform ABI.
impl From<Rotor4> for [f32; 8] {
    fn from(r: Rotor4) -> Self {
        [r.s, r.xy, r.xz, r.xw, r.yz, r.yw, r.zw, r.xyzw]
    }
}

impl Rotor4 {
    // Hestenes, New Foundations for Classical Mechanics, 2nd ed., §2.5.
    /// Column-major, matching glam's `Mat4` and WGSL's `mat4x4<f32>`.
    pub fn to_mat4(&self) -> [[f32; 4]; 4] {
        let c0 = <Self as Rotor>::apply(self, Vec4::new(1.0, 0.0, 0.0, 0.0));
        let c1 = <Self as Rotor>::apply(self, Vec4::new(0.0, 1.0, 0.0, 0.0));
        let c2 = <Self as Rotor>::apply(self, Vec4::new(0.0, 0.0, 1.0, 0.0));
        let c3 = <Self as Rotor>::apply(self, Vec4::new(0.0, 0.0, 0.0, 1.0));
        [c0.to_array(), c1.to_array(), c2.to_array(), c3.to_array()]
    }

    /// The rotor whose frame is `columns`, or `None` for a frame no rotor rebuilds within 1e-3, such as a reflection or a skew.
    pub fn from_mat4(columns: &[[f32; 4]; 4]) -> Option<Self> {
        let target = columns.map(Vec4::from_array);
        let axes = [Vec4::X, Vec4::Y, Vec4::Z];
        let mut rotor = Self::IDENTITY;
        for (axis, basis) in axes.into_iter().enumerate() {
            let from = <Self as Rotor>::apply(&rotor, basis);
            rotor = rotor * arc_holding(from, target[axis], &target[..axis]);
        }
        rotor
            .to_mat4()
            .iter()
            .flatten()
            .zip(columns.iter().flatten())
            .all(|(built, given)| (built - given).abs() <= ROTATION_TOLERANCE)
            .then_some(rotor)
    }
}

const ARC_PLANE_EPSILON: f32 = 1e-6;
const ROTATION_TOLERANCE: f32 = 1e-3;

// The plane is built orthogonal to `held`, since from ^ to is rounding noise near an antipode.
fn arc_holding(from: Vec4, to: Vec4, held: &[Vec4]) -> Rotor4 {
    let toward = beside(to, from, held);
    let reach = toward.length();
    let angle = reach.atan2(to.dot(from));
    let direction = if reach >= ARC_PLANE_EPSILON {
        toward
    } else {
        [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W]
            .map(|axis| beside(axis, from, held))
            .into_iter()
            .fold(Vec4::ZERO, |free, residual| {
                if residual.length_squared() > free.length_squared() {
                    residual
                } else {
                    free
                }
            })
    };
    let plane = Bivector4::wedge(from, direction);
    (plane * (angle / plane.magnitude())).exp().normalize()
}

// Parlett, The Symmetric Eigenvalue Problem, 1980, §6.9: Gram-Schmidt run twice is enough.
fn beside(v: Vec4, from: Vec4, held: &[Vec4]) -> Vec4 {
    let mut residual = v;
    for _ in 0..2 {
        residual -= from * (residual.dot(from) / from.length_squared());
        for fixed in held {
            residual -= *fixed * residual.dot(*fixed);
        }
    }
    residual
}

impl Mul for Rotor4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let (a0, a12, a13, a14, a23, a24, a34, a_i) = (
            self.s, self.xy, self.xz, self.xw, self.yz, self.yw, self.zw, self.xyzw,
        );
        let (b0, b12, b13, b14, b23, b24, b34, b_i) = (
            rhs.s, rhs.xy, rhs.xz, rhs.xw, rhs.yz, rhs.yw, rhs.zw, rhs.xyzw,
        );

        let s = a0 * b0 - a12 * b12 - a13 * b13 - a14 * b14 - a23 * b23 - a24 * b24 - a34 * b34
            + a_i * b_i;

        let xy = a0 * b12 + a12 * b0 - a13 * b23 + a23 * b13 - a14 * b24 + a24 * b14
            - a34 * b_i
            - a_i * b34;
        let xz = a0 * b13 + a13 * b0 + a12 * b23 - a14 * b34 - a23 * b12
            + a34 * b14
            + a24 * b_i
            + a_i * b24;
        let xw = a0 * b14 + a14 * b0 + a12 * b24 + a13 * b34
            - a24 * b12
            - a34 * b13
            - a23 * b_i
            - a_i * b23;
        let yz = a0 * b23 + a23 * b0 - a12 * b13 + a13 * b12 - a24 * b34 + a34 * b24
            - a14 * b_i
            - a_i * b14;
        let yw = a0 * b24 + a24 * b0 - a12 * b14 + a14 * b12 + a23 * b34 - a34 * b23
            + a13 * b_i
            + a_i * b13;
        let zw = a0 * b34 + a34 * b0 - a13 * b14 + a14 * b13 - a23 * b24 + a24 * b23
            - a12 * b_i
            - a_i * b12;

        let xyzw = a0 * b_i + a_i * b0 + a12 * b34 + a34 * b12 - a13 * b24 - a24 * b13
            + a14 * b23
            + a23 * b14;

        Self {
            s,
            xy,
            xz,
            xw,
            yz,
            yw,
            zw,
            xyzw,
        }
    }
}

impl Rotor for Rotor4 {
    type Bivector = Bivector4;
    type Vector = Vec4;

    fn identity() -> Self {
        Self::IDENTITY
    }

    fn inverse(self) -> Self {
        Self {
            s: self.s,
            xy: -self.xy,
            xz: -self.xz,
            xw: -self.xw,
            yz: -self.yz,
            yw: -self.yw,
            zw: -self.zw,
            xyzw: self.xyzw,
        }
    }

    fn apply(&self, v: Vec4) -> Vec4 {
        let (vx, vy, vz, vw) = (v.x, v.y, v.z, v.w);
        let rs = self.s;
        let rxy = self.xy;
        let rxz = self.xz;
        let rxw = self.xw;
        let ryz = self.yz;
        let ryw = self.yw;
        let rzw = self.zw;
        let r_i = self.xyzw;

        let p1 = rs * vx - rxy * vy - rxz * vz - rxw * vw;
        let p2 = rs * vy + rxy * vx - ryz * vz - ryw * vw;
        let p3 = rs * vz + rxz * vx + ryz * vy - rzw * vw;
        let p4 = rs * vw + rxw * vx + ryw * vy + rzw * vz;

        let t123 = -rxy * vz + rxz * vy - ryz * vx + r_i * vw;
        let t124 = -rxy * vw + rxw * vy - ryw * vx - r_i * vz;
        let t134 = -rxz * vw + rxw * vz - rzw * vx + r_i * vy;
        let t234 = -ryz * vw + ryw * vz - rzw * vy - r_i * vx;

        let q1 = rs * p1 - rxy * p2 - rxz * p3 - rxw * p4 - ryz * t123 - ryw * t124 - rzw * t134
            + r_i * t234;
        let q2 = rs * p2 + rxy * p1 - ryz * p3 - ryw * p4 + rxz * t123 + rxw * t124
            - rzw * t234
            - r_i * t134;
        let q3 = rs * p3 + rxz * p1 + ryz * p2 - rzw * p4 - rxy * t123
            + rxw * t134
            + ryw * t234
            + r_i * t124;
        let q4 = rs * p4 + rxw * p1 + ryw * p2 + rzw * p3
            - rxy * t124
            - rxz * t134
            - ryz * t234
            - r_i * t123;

        Vec4::new(q1, q2, q3, q4)
    }

    fn log(self) -> Bivector4 {
        // Kahan, How Futile are Mindless Assessments of Roundoff?, 2006, Mangled Angles.
        let branch = if self.s < 0.0 { -1.0 } else { 1.0 };
        let c = branch * self.s;
        let p = branch * self.xyzw;
        let sum_part = Vec3::new(self.xy + self.zw, self.xz - self.yw, self.xw + self.yz) * branch;
        let diff_part = Vec3::new(self.xy - self.zw, self.xz + self.yw, self.xw - self.yz) * branch;
        let sin_sum = sum_part.length();
        let sin_diff = diff_part.length();

        if sum_part == Vec3::ZERO && diff_part == Vec3::ZERO && p.abs() > c {
            let half_turn = std::f32::consts::PI;
            return Bivector4::new(half_turn, 0.0, 0.0, 0.0, 0.0, half_turn * p.signum());
        }

        let k_sum = if sin_sum > 0.0 {
            sin_sum.atan2(c - p) / sin_sum
        } else {
            1.0
        };
        let k_diff = if sin_diff > 0.0 {
            sin_diff.atan2(c + p) / sin_diff
        } else {
            1.0
        };

        Bivector4 {
            xy: k_sum * sum_part.x + k_diff * diff_part.x,
            xz: k_sum * sum_part.y + k_diff * diff_part.y,
            xw: k_sum * sum_part.z + k_diff * diff_part.z,
            yz: k_sum * sum_part.z - k_diff * diff_part.z,
            yw: k_diff * diff_part.y - k_sum * sum_part.y,
            zw: k_sum * sum_part.x - k_diff * diff_part.x,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI, SQRT_2, TAU};

    fn assert_close(a: f32, b: f32) {
        assert!(
            (a - b).abs() <= 1e-5,
            "expected {a} close to {b} (diff {})",
            (a - b).abs()
        );
    }

    fn assert_vec2_close(a: Vec2, b: Vec2) {
        assert!((a - b).length() <= 1e-5, "expected {a:?} close to {b:?}");
    }

    fn assert_vec3_close(a: Vec3, b: Vec3) {
        assert!((a - b).length() <= 1e-5, "expected {a:?} close to {b:?}");
    }

    fn assert_vec4_close_tol(a: Vec4, b: Vec4, tol: f32) {
        assert!(
            (a - b).length() <= tol,
            "expected {a:?} close to {b:?} (diff {})",
            (a - b).length()
        );
    }

    fn assert_vec4_close(a: Vec4, b: Vec4) {
        assert_vec4_close_tol(a, b, 1e-4);
    }

    #[test]
    fn quarter_turn() {
        let q = FRAC_PI_2;

        let r = Bivector2(q).exp();
        assert_vec2_close(r.apply(Vec2::X), Vec2::Y);
        assert_vec2_close(r.apply(Vec2::Y), -Vec2::X);

        for (plane, from, to, fixed) in [
            (Bivector3::new(q, 0.0, 0.0), Vec3::X, Vec3::Y, Vec3::Z),
            (Bivector3::new(0.0, q, 0.0), Vec3::Y, Vec3::Z, Vec3::X),
            (Bivector3::new(0.0, 0.0, q), Vec3::Z, Vec3::X, Vec3::Y),
        ] {
            let r = plane.exp();
            assert_vec3_close(r.apply(from), to);
            assert_vec3_close(r.apply(to), -from);
            assert_vec3_close(r.apply(fixed), fixed);
        }

        for (plane, from, to) in [
            (Bivector4::new(q, 0.0, 0.0, 0.0, 0.0, 0.0), Vec4::X, Vec4::Y),
            (Bivector4::new(0.0, q, 0.0, 0.0, 0.0, 0.0), Vec4::X, Vec4::Z),
            (Bivector4::new(0.0, 0.0, q, 0.0, 0.0, 0.0), Vec4::X, Vec4::W),
            (Bivector4::new(0.0, 0.0, 0.0, q, 0.0, 0.0), Vec4::Y, Vec4::Z),
            (Bivector4::new(0.0, 0.0, 0.0, 0.0, q, 0.0), Vec4::Y, Vec4::W),
            (Bivector4::new(0.0, 0.0, 0.0, 0.0, 0.0, q), Vec4::Z, Vec4::W),
        ] {
            let r = plane.exp();
            assert_vec4_close(r.apply(from), to);
            assert_vec4_close(r.apply(to), -from);
            for fixed in [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W] {
                if fixed != from && fixed != to {
                    assert_vec4_close(r.apply(fixed), fixed);
                }
            }
        }

        let r = Bivector4::new(q, 0.0, 0.0, 0.0, 0.0, q).exp();
        assert_vec4_close(r.apply(Vec4::X), Vec4::Y);
        assert_vec4_close(r.apply(Vec4::Y), -Vec4::X);
        assert_vec4_close(r.apply(Vec4::Z), Vec4::W);
        assert_vec4_close(r.apply(Vec4::W), -Vec4::Z);
    }

    #[test]
    fn product_order_and_drift() {
        let ra = Bivector4::new(0.4, 0.0, 0.0, 0.0, 0.0, 0.0).exp();
        let rb = Bivector4::new(0.0, 0.0, 0.0, 0.5, 0.0, 0.0).exp();
        let v = Vec4::new(0.7, -0.3, 1.1, 0.2);
        assert_vec4_close((ra * rb).apply(v), rb.apply(ra.apply(v)));

        let r = Bivector4::new(0.3, -0.2, 0.4, 0.1, -0.5, 0.25).exp();
        let id = r * r.inverse();
        assert_close(id.s, 1.0);
        assert_close(id.xy, 0.0);
        assert_close(id.xz, 0.0);
        assert_close(id.xw, 0.0);
        assert_close(id.yz, 0.0);
        assert_close(id.yw, 0.0);
        assert_close(id.zw, 0.0);
        assert_close(id.xyzw, 0.0);

        let omega = Bivector4::new(1.0, 1.0, 1.0, 1.0, 0.0, 0.0);
        let dt = 1.0 / 60.0;
        let delta = (omega * dt).exp();

        let mut integrated = Rotor4::IDENTITY;
        for _ in 0..60 {
            integrated = delta * integrated;
        }
        let closed_form = (omega * (dt * 60.0)).exp();

        for v in [
            Vec4::X,
            Vec4::Y,
            Vec4::Z,
            Vec4::W,
            Vec4::new(1.0, 2.0, 3.0, 4.0),
            Vec4::new(-0.5, 0.5, -0.5, 0.5),
        ] {
            let via_integrated = integrated.apply(v);
            let via_closed = closed_form.apply(v);
            let diff = (via_integrated - via_closed).length();
            assert!(
                diff < 1e-5,
                "compound xy+xz+xw+yz integration drift: v={v:?} \
                 integrated={via_integrated:?} closed={via_closed:?} \
                 diff={diff}",
            );
        }

        for _ in 60..900 {
            integrated = delta * integrated;
        }
        let n2 = integrated.norm_squared();
        assert!(
            (n2 - 1.0).abs() < 1e-5,
            "rotor norm drifted after 900 compositions: |R|^2 = {n2}",
        );
    }

    #[test]
    fn rotation_arc_parallel_and_antipodal() {
        let cases = [
            (Vec4::X, Vec4::Y),
            (Vec4::W, -Vec4::Y),
            (Vec4::Y, Vec4::Y),
            (Vec4::Y, -Vec4::Y),
            (Vec4::new(0.5, 0.5, 0.5, 0.5), -Vec4::Y),
            (
                Vec4::new(0.5, -0.5, 0.5, -0.5),
                Vec4::new(1.0, 0.0, 0.0, 0.0),
            ),
        ];
        for (from, to) in cases {
            let turned = Rotor4::from_rotation_arc(from, to).apply(from);
            assert!(
                (turned - to).length() < 1e-5,
                "{from} turned to {turned}, not {to}"
            );
        }
    }

    #[test]
    fn mat4_round_trip() {
        let r = Bivector4::new(0.7, 0.0, 0.0, 0.0, 0.0, 0.3).exp();
        let m = r.to_mat4();

        let v = Vec4::new(0.6, -0.4, 0.2, 0.9);
        let row = |k: usize| m[0][k] * v.x + m[1][k] * v.y + m[2][k] * v.z + m[3][k] * v.w;
        let by_matrix = Vec4::new(row(0), row(1), row(2), row(3));
        let by_rotor = r.apply(v);
        assert!(
            (by_matrix - by_rotor).length() < 1e-5,
            "matrix application {by_matrix:?} should match rotor application {by_rotor:?}"
        );

        for rotor in [
            Rotor4::IDENTITY,
            Bivector4::new(0.7, 0.0, 0.0, 0.0, 0.0, 0.3).exp(),
            Bivector4::new(FRAC_PI_2, 0.0, 0.0, 0.0, 0.0, FRAC_PI_2).exp(),
            Bivector4::new(0.0, 0.0, 0.0, PI, 0.0, 0.0).exp(),
            Bivector4::new(0.0, 0.0, 0.0, 0.0, 0.0, PI).exp(),
        ] {
            let frame = rotor.to_mat4();
            let rebuilt = Rotor4::from_mat4(&frame)
                .unwrap_or_else(|| panic!("{rotor:?} gave a frame no rotor rebuilds"))
                .to_mat4();
            for (slot, (built, given)) in rebuilt
                .iter()
                .flatten()
                .zip(frame.iter().flatten())
                .enumerate()
            {
                assert!(
                    (built - given).abs() < 1e-5,
                    "{rotor:?} slot {slot}: rebuilt {built} against {given}"
                );
            }
        }
    }

    fn invariant_plane_pairs() -> [(Bivector4, Bivector4); 4] {
        let root_half = 0.5_f32.sqrt();
        let u1 = Vec4::new(root_half, root_half, 0.0, 0.0);
        let v1 = Vec4::new(0.0, 0.0, root_half, root_half);
        let u2 = Vec4::new(root_half, -root_half, 0.0, 0.0);
        let v2 = Vec4::new(0.0, 0.0, root_half, -root_half);
        let [oblique, orthogonal] = nondegenerate_plane_pairs();
        [
            (Bivector4::basis(0), Bivector4::basis(5)),
            (Bivector4::wedge(u1, v1), Bivector4::wedge(v2, u2)),
            oblique,
            orthogonal,
        ]
    }

    fn plane_pair_from_eigenparts(sum: Vec3, diff: Vec3) -> (Bivector4, Bivector4) {
        let plane = |d: Vec3| Bivector4 {
            xy: 0.5 * (sum.x + d.x),
            xz: 0.5 * (sum.y + d.y),
            xw: 0.5 * (sum.z + d.z),
            yz: 0.5 * (sum.z - d.z),
            yw: 0.5 * (d.y - sum.y),
            zw: 0.5 * (sum.x - d.x),
        };
        (plane(diff), plane(-diff))
    }

    fn nondegenerate_plane_pairs() -> [(Bivector4, Bivector4); 2] {
        let sum = Vec3::new(2.0, 3.0, 6.0) / 7.0;
        let oblique_diff = Vec3::new(9.0, 2.0, 6.0) / 11.0;
        let orthogonal_diff = Vec3::new(3.0, -6.0, 2.0) / 7.0;
        [
            plane_pair_from_eigenparts(sum, oblique_diff),
            plane_pair_from_eigenparts(sum, orthogonal_diff),
        ]
    }

    const HALF_ANGLE_STRESS_PAIRS: [(f32, f32); 10] = [
        (1.0e-3, 1.0e-3),
        (1.0e-3, 5.0e-4),
        (1.0e-3, -1.0e-3),
        (1.0e-3, 0.0),
        (0.7, 0.7),
        (0.7, 0.6999),
        (1.0, 0.999),
        (2.0, 1.999),
        (1.2, 0.3),
        (0.5, -0.4),
    ];

    #[test]
    fn log_angles() {
        for (p1, p2) in invariant_plane_pairs() {
            for (t1, t2) in HALF_ANGLE_STRESS_PAIRS {
                let b = p1 * t1 + p2 * t2;

                let back = ((p1 * t1).exp() * (p2 * t2).exp()).log();
                let diff = (back + b * (-1.0)).magnitude();
                assert!(
                    diff <= 1e-6,
                    "log of the product mismatch at ({t1}, {t2}): {b:?} gave {back:?}"
                );

                let back = b.exp().log();
                let diff = (back + b * (-1.0)).magnitude();
                assert!(
                    diff <= 2e-5 * b.magnitude(),
                    "log of exp mismatch at ({t1}, {t2}): {b:?} gave {back:?}"
                );
            }
        }

        for (t1, t2) in [(3.0_f32, 1.0e-4_f32), (1.0, 1.0e-5), (0.5, -1.0e-4)] {
            let b = Bivector4::new(t1, 0.0, 0.0, 0.0, 0.0, t2);
            let rotor = b.exp();
            assert_close(rotor.norm_squared(), 1.0);
            let back = rotor.log();
            assert!(
                (back + b * (-1.0)).magnitude() <= 1e-6,
                "({t1}, {t2}) round-tripped as {back:?}"
            );
        }
    }

    struct Xorshift(u32);

    impl Xorshift {
        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }

        fn signed(&mut self, span: f32) -> f32 {
            let unit = self.next_u32() as f32 / u32::MAX as f32;
            (unit * 2.0 - 1.0) * span
        }

        fn bivector4(&mut self, span: f32) -> Bivector4 {
            Bivector4::new(
                self.signed(span),
                self.signed(span),
                self.signed(span),
                self.signed(span),
                self.signed(span),
                self.signed(span),
            )
        }
    }

    #[test]
    fn log_shortest_generator() {
        const STEPS: i32 = 47;
        for step in -STEPS..=STEPS {
            let theta = step as f32 * (TAU / STEPS as f32);
            let short = theta - TAU * (theta / TAU).round();
            let back = Bivector4::new(0.0, 0.0, 0.0, theta, 0.0, 0.0).exp().log();
            assert!(
                (back.yz - short).abs() <= 1e-5 && back.magnitude() <= PI,
                "a turn of {theta} logged as {back:?}, not {short}"
            );
        }

        const SAMPLES: usize = 64;
        let probes = [Vec4::X, Vec4::W, Vec4::new(1.0, -0.5, 0.3, 0.7)];
        let mut rng = Xorshift(0x5eed_4104);
        for _ in 0..SAMPLES {
            let rotor = rng.bivector4(TAU).exp();
            let back = rotor.log();
            let sum = Vec3::new(back.xy + back.zw, back.xz - back.yw, back.xw + back.yz);
            let diff = Vec3::new(back.xy - back.zw, back.xz + back.yw, back.xw - back.yz);

            assert!(
                sum.length() + diff.length() <= 2.0 * PI + 1e-5,
                "long way round: {rotor:?} logged as {back:?}"
            );
            assert!(
                back.magnitude() <= PI * SQRT_2 + 1e-5,
                "past the SO(4) diameter: {rotor:?} logged as {back:?}"
            );
            let regenerated = back.exp();
            for v in probes {
                assert_vec4_close_tol(regenerated.apply(v), rotor.apply(v), 1e-3);
            }
        }

        for sign in [-1.0, 1.0] {
            let pseudoscalar = Rotor4 {
                s: 0.0,
                xy: 0.0,
                xz: 0.0,
                xw: 0.0,
                yz: 0.0,
                yw: 0.0,
                zw: 0.0,
                xyzw: sign,
            };
            let recovered = pseudoscalar.log().exp();
            for v in [
                Vec4::X,
                Vec4::Y,
                Vec4::Z,
                Vec4::W,
                Vec4::new(1.0, -0.5, 0.3, 0.7),
            ] {
                assert_vec4_close_tol(pseudoscalar.apply(v), -v, 1e-6);
                assert_vec4_close_tol(recovered.apply(v), -v, 1e-6);
            }
        }

        let near = Bivector4::new(0.99 * PI, 0.0, 0.0, 0.0, 0.0, 0.99 * PI);
        let back = near.exp().log();
        assert!(back.magnitude() <= PI * SQRT_2);
        assert!((back + near * (-1.0)).magnitude() <= 1e-3, "{back:?}");
    }

    #[test]
    fn contract_vec() {
        let b = Bivector4::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_vec4_close_tol(b.contract_vec(Vec4::X), -Vec4::Y, 1e-6);
        assert_vec4_close_tol(b.contract_vec(Vec4::Y), Vec4::X, 1e-6);
        assert_vec4_close_tol(b.contract_vec(Vec4::Z), Vec4::ZERO, 1e-6);
        assert_vec4_close_tol(b.contract_vec(Vec4::W), Vec4::ZERO, 1e-6);
        let b = Bivector4::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        assert_vec4_close_tol(b.contract_vec(Vec4::Z), -Vec4::W, 1e-6);
        assert_vec4_close_tol(b.contract_vec(Vec4::W), Vec4::Z, 1e-6);
    }
}
