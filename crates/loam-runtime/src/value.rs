use crate::entity::SceneId;
use crate::store::{Owner, StoreField, Version};

/// One row with no entity; app state that snapshots like a store.
pub struct Value<T> {
    value: T,
    version: Version,
}

impl<T> Value<T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            version: Version::default(),
        }
    }

    pub fn get(&self) -> &T {
        &self.value
    }

    pub fn get_mut(&mut self) -> &mut T {
        self.version = self.version.bump();
        &mut self.value
    }

    pub fn set(&mut self, value: T) {
        *self.get_mut() = value;
    }

    pub fn version(&self) -> Version {
        self.version
    }
}

impl<T: Default> Default for Value<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: Clone + Send + 'static> StoreField for Value<T> {
    type Snapshot = T;

    fn bind(&mut self, _scene: SceneId, _owner: Owner) {}

    fn snapshot(&self) -> T {
        self.value.clone()
    }

    fn restore(&mut self, from: &T, _scene: SceneId, _owner: Owner) {
        self.set(from.clone());
    }
}
