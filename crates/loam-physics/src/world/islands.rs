use crate::body::{BodyArena, BodyId};
use crate::space::PhysicsSpace;

use super::{dense_pair, PairKey};

#[derive(Default)]
pub(super) struct Islands {
    parent: Vec<u32>,
    labels: Vec<BodyId>,
}

impl Islands {
    pub(super) fn clear(&mut self) {
        self.parent.clear();
        self.labels.clear();
    }

    pub(super) fn fill<S: PhysicsSpace>(
        &mut self,
        bodies: &BodyArena<S>,
        touched: impl Iterator<Item = PairKey>,
    ) {
        let n = bodies.len();
        self.parent.clear();
        self.parent.extend(0..n as u32);
        self.labels.clear();
        self.labels.extend((0..n).map(|dense| bodies.id_at(dense)));

        for key in touched {
            let (i, j) = dense_pair(bodies, key);
            if bodies[i].inv_mass() == 0.0 || bodies[j].inv_mass() == 0.0 {
                continue;
            }
            let (a, b) = (self.root(i), self.root(j));
            if a != b {
                self.parent[a.max(b)] = a.min(b) as u32;
            }
        }

        for dense in 0..n {
            let root = self.root(dense);
            self.labels[root] = self.labels[root].min(bodies.id_at(dense));
        }
        for dense in 0..n {
            let root = self.root(dense);
            let label = self.labels[root];
            self.labels[dense] = label;
        }
    }

    // Tarjan and van Leeuwen, JACM 31(2), 1984, sec. 2.
    fn root(&mut self, mut dense: usize) -> usize {
        let parent = &mut self.parent;
        while parent[dense] as usize != dense {
            parent[dense] = parent[parent[dense] as usize];
            dense = parent[dense] as usize;
        }
        dense
    }

    pub(super) fn constraint_island<S: PhysicsSpace>(
        &self,
        bodies: &BodyArena<S>,
        dense: (usize, usize),
    ) -> BodyId {
        let (i, j) = dense;
        debug_assert!(
            bodies[i].inv_mass() != 0.0 || bodies[j].inv_mass() != 0.0,
            "a contact between two static bodies has no island to solve in",
        );
        if bodies[i].inv_mass() != 0.0 {
            self.labels[i]
        } else {
            self.labels[j]
        }
    }
}
