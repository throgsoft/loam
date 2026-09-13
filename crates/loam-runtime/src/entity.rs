use std::sync::atomic::{AtomicU32, Ordering};

use crate::command::Rejection;
use crate::domain::DomainError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RuntimeId(u32);

static NEXT_RUNTIME: AtomicU32 = AtomicU32::new(1);

impl RuntimeId {
    pub(crate) fn allocate() -> Self {
        match NEXT_RUNTIME
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        {
            Ok(id) => Self(id),
            Err(_) => panic!("runtime id space exhausted"),
        }
    }
}

/// Advances on every restore and never rewinds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Epoch(u32);

impl Epoch {
    pub fn advance(self) -> Self {
        match self.0.checked_add(1) {
            Some(epoch) => Self(epoch),
            None => panic!("epoch space exhausted"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SceneId {
    pub(crate) runtime: RuntimeId,
    pub(crate) epoch: Epoch,
}

impl SceneId {
    pub(crate) const UNBOUND: Self = Self {
        runtime: RuntimeId(0),
        epoch: Epoch(0),
    };

    pub fn runtime(self) -> RuntimeId {
        self.runtime
    }

    pub fn epoch(self) -> Epoch {
        self.epoch
    }
}

/// Slot and generation; survives a restore and never leaves the session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntityKey {
    slot: u32,
    generation: u32,
}

impl EntityKey {
    pub fn slot(self) -> u32 {
        self.slot
    }

    pub fn generation(self) -> u32 {
        self.generation
    }
}

/// Resolves to its original object or fails: after a restore, in another session, after despawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Entity {
    scene: SceneId,
    key: EntityKey,
}

impl Entity {
    pub(crate) fn new(scene: SceneId, key: EntityKey) -> Self {
        Self { scene, key }
    }

    pub fn scene(self) -> SceneId {
        self.scene
    }

    pub fn key(self) -> EntityKey {
        self.key
    }
}

#[derive(Clone)]
struct Slot {
    generation: u32,
    live: bool,
    reserved: bool,
}

pub struct Entities {
    scene: SceneId,
    slots: Vec<Slot>,
    free: Vec<u32>,
    live: usize,
}

impl Entities {
    pub(crate) fn new(scene: SceneId) -> Self {
        Self {
            scene,
            slots: Vec::new(),
            free: Vec::new(),
            live: 0,
        }
    }

    pub fn scene(&self) -> SceneId {
        self.scene
    }

    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    pub fn spawn(&mut self) -> Entity {
        let entity = self.reserve();
        self.commit(entity);
        entity
    }

    pub(crate) fn reserve(&mut self) -> Entity {
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None => {
                self.slots.push(Slot {
                    generation: 0,
                    live: false,
                    reserved: false,
                });
                (self.slots.len() - 1) as u32
            }
        };
        let entry = &mut self.slots[slot as usize];
        entry.reserved = true;
        Entity::new(
            self.scene,
            EntityKey {
                slot,
                generation: entry.generation,
            },
        )
    }

    pub(crate) fn commit(&mut self, entity: Entity) {
        if let Some(slot) = self.reserved_index(entity) {
            self.slots[slot].reserved = false;
            self.slots[slot].live = true;
            self.live += 1;
        }
    }

    pub(crate) fn release(&mut self, entity: Entity) {
        if let Some(slot) = self.reserved_index(entity) {
            self.slots[slot].reserved = false;
            self.retire(slot);
        }
    }

    pub fn is_reserved(&self, entity: Entity) -> bool {
        self.reserved_index(entity).is_some()
    }

    pub(crate) fn has_reservations(&self) -> bool {
        self.slots.iter().any(|slot| slot.reserved)
    }

    fn reserved_index(&self, entity: Entity) -> Option<usize> {
        if entity.scene != self.scene {
            return None;
        }
        let key = entity.key;
        let slot = self.slots.get(key.slot as usize)?;
        (slot.reserved && slot.generation == key.generation).then_some(key.slot as usize)
    }

    /// A slot whose generation would wrap is retired instead of recycled.
    pub fn despawn(&mut self, entity: Entity) -> Result<EntityKey, Rejection> {
        let key = self
            .resolve(entity)
            .ok_or(Rejection::Domain(DomainError::Stale(entity)))?;
        self.slots[key.slot as usize].live = false;
        self.live -= 1;
        self.retire(key.slot as usize);
        Ok(key)
    }

    fn retire(&mut self, slot: usize) {
        let entry = &mut self.slots[slot];
        if let Some(next) = entry.generation.checked_add(1) {
            entry.generation = next;
            self.free.push(slot as u32);
        }
    }

    pub fn resolve(&self, entity: Entity) -> Option<EntityKey> {
        if entity.scene != self.scene {
            return None;
        }
        let key = entity.key;
        let slot = self.slots.get(key.slot as usize)?;
        (slot.live && slot.generation == key.generation).then_some(key)
    }

    pub fn snapshot(&self) -> EntitiesSnapshot {
        EntitiesSnapshot {
            slots: self.slots.clone(),
            free: self.free.clone(),
            live: self.live,
        }
    }

    /// Advances the epoch; every handle from before the restore fails.
    pub fn restore(&mut self, from: &EntitiesSnapshot) {
        self.scene.epoch = self.scene.epoch.advance();
        self.slots.clone_from(&from.slots);
        self.free.clone_from(&from.free);
        self.live = from.live;
    }
}

pub struct EntitiesSnapshot {
    slots: Vec<Slot>,
    free: Vec<u32>,
    live: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(runtime: u32) -> SceneId {
        SceneId {
            runtime: RuntimeId(runtime),
            epoch: Epoch::default(),
        }
    }

    #[test]
    fn stale_reset_or_foreign_handle_never_reaches_a_recycled_object() {
        let mut entities = Entities::new(scene(1));
        let doomed = entities.spawn();
        assert!(entities.despawn(doomed).is_ok());
        let recycled = entities.spawn();
        assert_eq!(recycled.key().slot(), doomed.key().slot());
        assert_eq!(entities.resolve(doomed), None);
        assert_eq!(entities.resolve(recycled), Some(recycled.key()));
        assert!(matches!(
            entities.despawn(doomed),
            Err(Rejection::Domain(DomainError::Stale(stale))) if stale == doomed
        ));
        assert_eq!(entities.len(), 1);

        let foreign = Entity::new(scene(2), recycled.key());
        assert_eq!(entities.resolve(foreign), None);

        let snapshot = entities.snapshot();
        entities.restore(&snapshot);
        assert_eq!(entities.resolve(recycled), None);
        let rebased = Entity::new(entities.scene(), recycled.key());
        assert_eq!(entities.resolve(rebased), Some(recycled.key()));
        assert_eq!(entities.scene().epoch(), Epoch::default().advance());
    }
}
