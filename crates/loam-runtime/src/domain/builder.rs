use super::blend::Blend;
use super::field_table::Fields;
use super::view_table::ViewTable;
use super::{DomainId, DomainSpace, Facility, TypedDomain};
use crate::entity::SceneId;
use crate::store::{LogCapacity, Owner, Store};

pub struct DomainBuilder<S: DomainSpace> {
    name: &'static str,
    pub(crate) space: S,
    tracking: Option<LogCapacity>,
    fields: bool,
    facilities: Vec<Box<dyn Facility<S>>>,
}

impl<S: DomainSpace> DomainBuilder<S> {
    pub fn new(name: &'static str, space: S) -> Self {
        Self {
            name,
            space,
            tracking: Some(LogCapacity::default()),
            fields: false,
            facilities: Vec::new(),
        }
    }

    pub fn tracked(mut self, capacity: LogCapacity) -> Self {
        self.tracking = Some(capacity);
        self
    }

    pub fn untracked(mut self) -> Self {
        self.tracking = None;
        self
    }

    pub fn fields(mut self) -> Self {
        self.fields = true;
        self
    }

    pub fn facility(mut self, facility: impl Facility<S>) -> Self {
        self.facilities.push(Box::new(facility));
        self
    }

    pub(crate) fn build(self, id: DomainId, scene: SceneId) -> TypedDomain<S> {
        fn store<T>(tracking: Option<LogCapacity>, scene: SceneId) -> Store<T> {
            let mut store = match tracking {
                Some(capacity) => Store::tracked(capacity),
                None => Store::untracked(),
            };
            store.bind(scene);
            store
        }
        let mut facilities = self.facilities;
        for facility in &mut facilities {
            facility.bind(scene, Owner::new());
        }
        TypedDomain {
            id,
            scene,
            name: self.name,
            space: self.space,
            poses: store(self.tracking, scene),
            instances: store(self.tracking, scene),
            fields: Fields::new(self.fields.then(|| store(self.tracking, scene))),
            views: ViewTable::new(),
            facilities,
            blend: Blend::new(),
        }
    }
}
