use std::sync::atomic::{AtomicU32, Ordering};

use crate::command::Rejection;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RuntimeId(u32);

static NEXT_RUNTIME: AtomicU32 = AtomicU32::new(1);

impl RuntimeId {
    pub(crate) fn allocate() -> Self {
        Self(NEXT_RUNTIME.fetch_add(1, Ordering::Relaxed))
    }
}

/// Advances on every restore and never rewinds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Epoch(u32);

impl Epoch {
    pub fn advance(self) -> Self {
        Self(self.0.wrapping_add(1))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SceneId {
    pub runtime: RuntimeId,
    pub epoch: Epoch,
}

impl SceneId {
    pub(crate) const UNBOUND: Self = Self {
        runtime: RuntimeId(0),
        epoch: Epoch(0),
    };
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

struct Slot {
    generation: u32,
    live: bool,
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
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None => {
                self.slots.push(Slot {
                    generation: 0,
                    live: false,
                });
                (self.slots.len() - 1) as u32
            }
        };
        let entry = &mut self.slots[slot as usize];
        entry.live = true;
        self.live += 1;
        Entity::new(
            self.scene,
            EntityKey {
                slot,
                generation: entry.generation,
            },
        )
    }

    /// A slot whose generation would wrap is retired instead of recycled.
    pub fn despawn(&mut self, entity: Entity) -> Result<EntityKey, Rejection> {
        let key = self.resolve(entity).ok_or(Rejection::Stale(entity))?;
        let slot = &mut self.slots[key.slot as usize];
        slot.live = false;
        self.live -= 1;
        if let Some(next) = slot.generation.checked_add(1) {
            slot.generation = next;
            self.free.push(key.slot);
        }
        Ok(key)
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
        todo!()
    }
}

pub struct EntitiesSnapshot;
