use crate::entity::{Entity, SceneId};
use crate::store::{ErasedStore, SchemaId, StoreError, StoreField};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LinkId(u32);

/// A typed pair of entities with data; endpoints resolve through the sparse index.
#[derive(Clone, Copy, Debug)]
pub struct Link<T> {
    pub from: Entity,
    pub to: Entity,
    pub data: T,
}

pub struct Relation<T> {
    scene: SceneId,
    links: Vec<Link<T>>,
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

    pub fn link(&mut self, _from: Entity, _to: Entity, _data: T) -> Result<LinkId, StoreError> {
        todo!()
    }

    pub fn unlink(&mut self, _id: LinkId) -> Result<Link<T>, StoreError> {
        todo!()
    }

    pub fn get(&self, _id: LinkId) -> Option<&Link<T>> {
        todo!()
    }

    pub fn get_mut(&mut self, _id: LinkId) -> Option<&mut Link<T>> {
        todo!()
    }

    /// O(1) to the first link; no rescan after a swap-remove.
    pub fn outgoing(&self, _from: Entity) -> Endpoints<'_> {
        todo!()
    }

    pub fn incoming(&self, _to: Entity) -> Endpoints<'_> {
        todo!()
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
        todo!()
    }

    fn restore(&mut self, _from: &RelationSnapshot<T>, _scene: SceneId) {
        todo!()
    }

    fn erased(&mut self) -> &mut dyn ErasedStore {
        self
    }
}

impl<T: Send + 'static> ErasedStore for Relation<T> {
    fn schema(&self) -> SchemaId {
        SchemaId::of::<T>()
    }

    fn len(&self) -> usize {
        self.links.len()
    }

    fn is_tracked(&self) -> bool {
        false
    }
}
