use crate::entity::{Entities, Entity, SceneId};
use crate::store::{StoreError, StoreField};

/// Resolves to its original link or fails: after unlink, after a restore, in another session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LinkId {
    scene: SceneId,
    slot: u32,
    generation: u32,
}

/// A typed pair of entities with data; endpoints resolve through the sparse index.
#[derive(Clone, Copy, Debug)]
pub struct Link<T> {
    pub(crate) from: Entity,
    pub(crate) to: Entity,
    pub data: T,
}

impl<T> Link<T> {
    pub fn from(&self) -> Entity {
        self.from
    }

    pub fn to(&self) -> Entity {
        self.to
    }
}

#[derive(Clone, Copy)]
struct LinkSlot {
    generation: u32,
    dense: Option<u32>,
}

#[derive(Clone, Copy)]
struct Ends {
    outgoing: u32,
    incoming: u32,
}

#[derive(Default)]
struct Adjacency {
    generation: u32,
    outgoing: Vec<LinkId>,
    incoming: Vec<LinkId>,
}

pub struct Relation<T> {
    scene: SceneId,
    links: Vec<Link<T>>,
    ids: Vec<LinkId>,
    ends: Vec<Ends>,
    slots: Vec<LinkSlot>,
    free: Vec<u32>,
    adjacency: Vec<Adjacency>,
}

impl<T> Default for Relation<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Relation<T> {
    pub fn new() -> Self {
        Self {
            scene: SceneId::UNBOUND,
            links: Vec::new(),
            ids: Vec::new(),
            ends: Vec::new(),
            slots: Vec::new(),
            free: Vec::new(),
            adjacency: Vec::new(),
        }
    }

    pub fn scene(&self) -> SceneId {
        self.scene
    }

    pub fn len(&self) -> usize {
        self.links.len()
    }

    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    pub fn links(&self) -> &[Link<T>] {
        &self.links
    }

    pub fn ids(&self) -> &[LinkId] {
        &self.ids
    }

    /// `Occupied` when the endpoint's slot still lists links of an older generation.
    pub fn link(
        &mut self,
        entities: &Entities,
        from: Entity,
        to: Entity,
        data: T,
    ) -> Result<LinkId, StoreError> {
        for endpoint in [from, to] {
            if endpoint.scene() != entities.scene() {
                return Err(StoreError::Foreign(endpoint));
            }
            if entities.resolve(endpoint).is_none() {
                return Err(StoreError::Stale(endpoint));
            }
        }
        if self.scene == SceneId::UNBOUND {
            self.scene = entities.scene();
        }
        self.link_raw(from, to, data)
    }

    pub(crate) fn link_raw(
        &mut self,
        from: Entity,
        to: Entity,
        data: T,
    ) -> Result<LinkId, StoreError> {
        for endpoint in [from, to] {
            if endpoint.scene() != self.scene {
                return Err(StoreError::Foreign(endpoint));
            }
            self.claim(endpoint)?;
        }
        let dense = u32::try_from(self.links.len()).map_err(|_| StoreError::Capacity)?;
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None => {
                self.slots.push(LinkSlot {
                    generation: 0,
                    dense: None,
                });
                (self.slots.len() - 1) as u32
            }
        };
        self.slots[slot as usize].dense = Some(dense);
        let id = LinkId {
            scene: self.scene,
            slot,
            generation: self.slots[slot as usize].generation,
        };
        let outgoing = &mut self.adjacency[from.key().slot() as usize].outgoing;
        let out_position = outgoing.len() as u32;
        outgoing.push(id);
        let incoming = &mut self.adjacency[to.key().slot() as usize].incoming;
        let in_position = incoming.len() as u32;
        incoming.push(id);
        self.links.push(Link { from, to, data });
        self.ids.push(id);
        self.ends.push(Ends {
            outgoing: out_position,
            incoming: in_position,
        });
        Ok(id)
    }

    fn claim(&mut self, endpoint: Entity) -> Result<(), StoreError> {
        let slot = endpoint.key().slot() as usize;
        if slot >= self.adjacency.len() {
            self.adjacency.resize_with(slot + 1, Adjacency::default);
        }
        let adjacency = &mut self.adjacency[slot];
        if adjacency.generation != endpoint.key().generation() {
            if !adjacency.outgoing.is_empty() || !adjacency.incoming.is_empty() {
                return Err(StoreError::Occupied(endpoint));
            }
            adjacency.generation = endpoint.key().generation();
        }
        Ok(())
    }

    pub fn unlink(&mut self, id: LinkId) -> Result<Link<T>, StoreError> {
        let dense = self.dense_index(id).ok_or(StoreError::Unlinked(id))?;
        let ends = self.ends[dense];
        let from_slot = self.links[dense].from.key().slot() as usize;
        let to_slot = self.links[dense].to.key().slot() as usize;
        let outgoing = &mut self.adjacency[from_slot].outgoing;
        outgoing.swap_remove(ends.outgoing as usize);
        let moved_out = outgoing.get(ends.outgoing as usize).copied();
        if let Some(moved) = moved_out.and_then(|moved| self.dense_index(moved)) {
            self.ends[moved].outgoing = ends.outgoing;
        }
        let incoming = &mut self.adjacency[to_slot].incoming;
        incoming.swap_remove(ends.incoming as usize);
        let moved_in = incoming.get(ends.incoming as usize).copied();
        if let Some(moved) = moved_in.and_then(|moved| self.dense_index(moved)) {
            self.ends[moved].incoming = ends.incoming;
        }
        let link = self.links.swap_remove(dense);
        self.ids.swap_remove(dense);
        self.ends.swap_remove(dense);
        if let Some(&moved) = self.ids.get(dense) {
            self.slots[moved.slot as usize].dense = Some(dense as u32);
        }
        let slot = &mut self.slots[id.slot as usize];
        slot.dense = None;
        if let Some(next) = slot.generation.checked_add(1) {
            slot.generation = next;
            self.free.push(id.slot);
        }
        Ok(link)
    }

    fn dense_index(&self, id: LinkId) -> Option<usize> {
        if id.scene != self.scene {
            return None;
        }
        let slot = self.slots.get(id.slot as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.dense.map(|dense| dense as usize)
    }

    pub fn get(&self, id: LinkId) -> Option<&Link<T>> {
        self.dense_index(id).map(|dense| &self.links[dense])
    }

    pub fn get_mut(&mut self, id: LinkId) -> Option<&mut T> {
        let dense = self.dense_index(id)?;
        Some(&mut self.links[dense].data)
    }

    /// O(1) to the first link; no rescan after a swap-remove.
    pub fn outgoing(&self, from: Entity) -> Endpoints<'_> {
        Endpoints {
            ids: self
                .adjacency_of(from)
                .map_or(&[], |adjacency| &adjacency.outgoing),
        }
    }

    pub fn incoming(&self, to: Entity) -> Endpoints<'_> {
        Endpoints {
            ids: self
                .adjacency_of(to)
                .map_or(&[], |adjacency| &adjacency.incoming),
        }
    }

    fn adjacency_of(&self, endpoint: Entity) -> Option<&Adjacency> {
        if endpoint.scene() != self.scene {
            return None;
        }
        let adjacency = self.adjacency.get(endpoint.key().slot() as usize)?;
        (adjacency.generation == endpoint.key().generation()).then_some(adjacency)
    }
}

pub struct Endpoints<'a> {
    ids: &'a [LinkId],
}

impl Iterator for Endpoints<'_> {
    type Item = LinkId;

    fn next(&mut self) -> Option<LinkId> {
        let (&first, rest) = self.ids.split_first()?;
        self.ids = rest;
        Some(first)
    }
}

pub struct RelationSnapshot<T> {
    links: Vec<Link<T>>,
}

impl<T> RelationSnapshot<T> {
    pub fn links(&self) -> &[Link<T>] {
        &self.links
    }
}

impl<T: Clone + Send + 'static> StoreField for Relation<T> {
    type Snapshot = RelationSnapshot<T>;

    fn bind(&mut self, scene: SceneId) {
        self.scene = scene;
    }

    fn snapshot(&self) -> RelationSnapshot<T> {
        RelationSnapshot {
            links: self.links.clone(),
        }
    }

    fn restore(&mut self, from: &RelationSnapshot<T>, scene: SceneId) {
        self.scene = scene;
        self.links.clear();
        self.ids.clear();
        self.ends.clear();
        self.slots.clear();
        self.free.clear();
        self.adjacency.clear();
        for link in &from.links {
            let from = Entity::new(scene, link.from.key());
            let to = Entity::new(scene, link.to.key());
            let _ = self.link_raw(from, to, link.data.clone());
        }
    }

    fn release(&mut self, entity: Entity) {
        while let Some(id) = self
            .outgoing(entity)
            .next()
            .or_else(|| self.incoming(entity).next())
        {
            if self.unlink(id).is_err() {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::{Entities, Epoch, RuntimeId};

    fn data(relation: &Relation<&'static str>, ids: Endpoints<'_>) -> Vec<&'static str> {
        let mut data: Vec<&str> = ids.map(|id| relation.get(id).unwrap().data).collect();
        data.sort_unstable();
        data
    }

    #[test]
    fn endpoints_resolve_after_a_swap_remove() {
        let mut entities = Entities::new(SceneId {
            runtime: RuntimeId::allocate(),
            epoch: Epoch::default(),
        });
        let mut relation = Relation::new();
        relation.bind(entities.scene());
        let [a, b, c] = [(); 3].map(|()| entities.spawn());
        let ab = relation.link(&entities, a, b, "ab").unwrap();
        let ac = relation.link(&entities, a, c, "ac").unwrap();
        let bc = relation.link(&entities, b, c, "bc").unwrap();

        assert_eq!(relation.unlink(ab).map(|link| link.data), Ok("ab"));
        assert_eq!(
            relation.unlink(ab).map(|link| link.data),
            Err(StoreError::Unlinked(ab))
        );
        assert_eq!(relation.len(), 2);
        assert!(relation.get(ab).is_none());
        assert_eq!(relation.get(bc).map(|link| link.data), Some("bc"));
        assert_eq!(data(&relation, relation.outgoing(a)), ["ac"]);
        assert_eq!(data(&relation, relation.incoming(c)), ["ac", "bc"]);
        assert!(data(&relation, relation.incoming(b)).is_empty());

        assert_eq!(relation.unlink(bc).map(|link| link.data), Ok("bc"));
        assert_eq!(data(&relation, relation.incoming(c)), ["ac"]);
        assert_eq!(relation.get(ac).map(Link::to), Some(c));
        let relinked = relation.link(&entities, b, c, "bc2").unwrap();
        assert_ne!(relinked, bc);
        assert!(relation.get(bc).is_none());
        assert_eq!(data(&relation, relation.incoming(c)), ["ac", "bc2"]);
    }

    #[test]
    fn mutable_link_data_keeps_adjacency_endpoints_sealed() {
        let mut entities = Entities::new(SceneId {
            runtime: RuntimeId::allocate(),
            epoch: Epoch::default(),
        });
        let mut relation = Relation::new();
        relation.bind(entities.scene());
        let [a, b, c] = [(); 3].map(|()| entities.spawn());
        let ab = relation.link(&entities, a, b, "before").unwrap();

        *relation.get_mut(ab).unwrap() = "after";

        assert_eq!(relation.get(ab).map(Link::from), Some(a));
        assert_eq!(relation.get(ab).map(Link::to), Some(b));
        assert_eq!(data(&relation, relation.outgoing(a)), ["after"]);
        assert_eq!(data(&relation, relation.incoming(b)), ["after"]);
        assert!(relation.incoming(c).next().is_none());
    }
}
