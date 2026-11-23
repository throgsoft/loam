use crate::polytope::Polytope4;

/// A 4D solid at unit size that a field kernel draws by its distance function.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FieldShape {
    Polytope(Polytope4),
    ThreeSphere,
    Duocylinder,
    CliffordTorus,
    Spherinder,
}

impl FieldShape {
    pub fn polytope4(&self) -> Option<Polytope4> {
        match self {
            FieldShape::Polytope(polytope) => Some(*polytope),
            _ => None,
        }
    }

    /// Distance from the origin to the farthest point of the unit solid.
    pub fn radius(&self) -> f32 {
        match self {
            FieldShape::Polytope(polytope) => polytope
                .topology()
                .vertices
                .iter()
                .map(|vertex| vertex.length())
                .fold(0.0, f32::max),
            // The kernel's smooth solids are built inside the unit ball.
            _ => 1.0,
        }
    }
}
