use crate::body::{BodyArena, BodyId};
use crate::integrator::PhysicsSpace;

#[derive(Default)]
pub(crate) struct DirtyBodies {
    ids: Vec<BodyId>,
    marked: Vec<bool>,
}

impl DirtyBodies {
    pub(crate) fn mark(&mut self, id: BodyId) {
        let slot = id.slot() as usize;
        if slot >= self.marked.len() {
            self.marked.resize(slot + 1, false);
        }
        if self.marked[slot] {
            return;
        }
        self.marked[slot] = true;
        self.ids.push(id);
    }

    pub(crate) fn forget(&mut self, id: BodyId) {
        if let Some(flag) = self.marked.get_mut(id.slot() as usize) {
            *flag = false;
        }
    }

    pub(crate) fn recorded(&self) -> &[BodyId] {
        &self.ids
    }

    pub(crate) fn reset(&mut self, ids: &[BodyId]) {
        self.ids.clear();
        for flag in &mut self.marked {
            *flag = false;
        }
        for &id in ids {
            self.mark(id);
        }
    }

    pub(crate) fn drain<'a, S: PhysicsSpace>(
        &'a mut self,
        bodies: &'a BodyArena<S>,
    ) -> DirtyDrain<'a, S> {
        let Self { ids, marked } = self;
        DirtyDrain {
            bodies,
            marked,
            ids: ids.drain(..),
        }
    }
}

pub struct DirtyDrain<'a, S: PhysicsSpace> {
    bodies: &'a BodyArena<S>,
    marked: &'a mut [bool],
    ids: std::vec::Drain<'a, BodyId>,
}

impl<S: PhysicsSpace> Iterator for DirtyDrain<'_, S> {
    type Item = BodyId;

    fn next(&mut self) -> Option<BodyId> {
        for id in self.ids.by_ref() {
            if let Some(flag) = self.marked.get_mut(id.slot() as usize) {
                *flag = false;
            }
            if self.bodies.dense_index(id).is_some() {
                return Some(id);
            }
        }
        None
    }
}

impl<S: PhysicsSpace> Drop for DirtyDrain<'_, S> {
    fn drop(&mut self) {
        for id in self.ids.by_ref() {
            if let Some(flag) = self.marked.get_mut(id.slot() as usize) {
                *flag = false;
            }
        }
    }
}
