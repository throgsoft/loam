#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FieldKind {
    #[default]
    ExactDistance,
    ConservativeBound,
    Implicit,
}

impl FieldKind {
    fn rank(self) -> u8 {
        match self {
            FieldKind::ExactDistance => 0,
            FieldKind::ConservativeBound => 1,
            FieldKind::Implicit => 2,
        }
    }

    pub fn weaker(self, other: Self) -> Self {
        if other.rank() > self.rank() {
            other
        } else {
            self
        }
    }
}

pub const GRADIENT_STEP: f32 = 1.0e-3;

/// Below this a contact query refuses rather than normalizes noise into a normal.
pub const MIN_GRADIENT_NORM: f32 = 1.0e-4;

/// `distance` returns the Euclidean value described by `field_kind` in the first `dimension` coordinates, off by at most `error_at` at the query point.
pub trait DistanceField: Send + Sync {
    fn field_kind(&self) -> FieldKind;

    fn dimension(&self) -> u32;

    fn distance(&self, point: [f32; 4]) -> f32;

    fn error_at(&self, point: [f32; 4]) -> f32;

    fn gradient(&self, point: [f32; 4]) -> [f32; 4] {
        let mut gradient = [0.0; 4];
        for axis in 0..self.dimension().min(gradient.len() as u32) as usize {
            let mut ahead = point;
            let mut behind = point;
            ahead[axis] += GRADIENT_STEP;
            behind[axis] -= GRADIENT_STEP;
            gradient[axis] = (self.distance(ahead) - self.distance(behind)) / (2.0 * GRADIENT_STEP);
        }
        gradient
    }
}
