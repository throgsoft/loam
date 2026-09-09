use crate::collider::{Collider, ColliderKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GeometryId(u32);

impl GeometryId {
    pub fn index(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GeometryRef {
    id: GeometryId,
    version: u32,
}

impl GeometryRef {
    pub fn id(self) -> GeometryId {
        self.id
    }

    pub fn version(self) -> u32 {
        self.version
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColliderRef {
    kind: ColliderKind,
    geometry: GeometryRef,
}

impl ColliderRef {
    pub fn kind(self) -> ColliderKind {
        self.kind
    }

    pub fn geometry(self) -> GeometryRef {
        self.geometry
    }
}

#[derive(Clone)]
struct Entry {
    version: u32,
    uses: u32,
    shape: Option<Collider>,
}

const RELEASED_BUFFERS: usize = 2;

#[derive(Clone, Default)]
pub struct GeometryStore {
    entries: Vec<Entry>,
    free: Vec<u32>,
    released: Vec<Collider>,
}

impl GeometryStore {
    pub fn prepare(&mut self, shape: Collider) -> ColliderRef {
        let kind = shape.kind();
        let shared = self.entries.iter().position(|entry| {
            entry
                .shape
                .as_ref()
                .is_some_and(|prepared| same_shape(prepared, &shape))
        });
        if let Some(slot) = shared {
            let entry = &mut self.entries[slot];
            entry.uses += 1;
            let version = entry.version;
            self.stash(shape);
            return ColliderRef {
                kind,
                geometry: GeometryRef {
                    id: GeometryId(slot as u32),
                    version,
                },
            };
        }

        let slot = match self.free.pop() {
            Some(slot) => {
                let entry = &mut self.entries[slot as usize];
                entry.uses = 1;
                entry.shape = Some(shape);
                slot
            }
            None => {
                self.entries.push(Entry {
                    version: 0,
                    uses: 1,
                    shape: Some(shape),
                });
                (self.entries.len() - 1) as u32
            }
        };
        ColliderRef {
            kind,
            geometry: GeometryRef {
                id: GeometryId(slot),
                version: self.entries[slot as usize].version,
            },
        }
    }

    pub fn release(&mut self, collider: ColliderRef) {
        let slot = collider.geometry.id.0 as usize;
        let Some(entry) = self.entries.get_mut(slot) else {
            return;
        };
        if entry.version != collider.geometry.version || entry.uses == 0 {
            return;
        }
        entry.uses -= 1;
        if entry.uses > 0 {
            return;
        }
        let freed = entry.shape.take();
        if let Some(next) = entry.version.checked_add(1) {
            entry.version = next;
            self.free.push(slot as u32);
        }
        if let Some(freed) = freed {
            self.stash(freed);
        }
    }

    pub(crate) fn stash(&mut self, shape: Collider) {
        if self.released.len() == RELEASED_BUFFERS {
            self.released.remove(0);
        }
        self.released.push(shape);
    }

    pub fn take_released(&mut self) -> Option<Collider> {
        self.released.pop()
    }

    pub fn get(&self, collider: ColliderRef) -> Option<&Collider> {
        let entry = self.entries.get(collider.geometry.id.0 as usize)?;
        if entry.version != collider.geometry.version {
            return None;
        }
        entry.shape.as_ref()
    }

    pub fn prepared(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.shape.is_some())
            .count()
    }
}

fn same_shape(a: &Collider, b: &Collider) -> bool {
    match (a, b) {
        (
            Collider::Sphere {
                center: ca,
                radius: ra,
            },
            Collider::Sphere {
                center: cb,
                radius: rb,
            },
        ) => ca == cb && ra == rb,
        (
            Collider::HyperSphere4D {
                center: ca,
                radius: ra,
            },
            Collider::HyperSphere4D {
                center: cb,
                radius: rb,
            },
        ) => ca == cb && ra == rb,
        (
            Collider::HalfSpace {
                normal: na,
                offset: oa,
            },
            Collider::HalfSpace {
                normal: nb,
                offset: ob,
            },
        ) => na == nb && oa == ob,
        (
            Collider::HalfSpace4D {
                normal: na,
                offset: oa,
            },
            Collider::HalfSpace4D {
                normal: nb,
                offset: ob,
            },
        ) => na == nb && oa == ob,
        (Collider::Box3 { half_extents: a }, Collider::Box3 { half_extents: b }) => a == b,
        (Collider::Polygon2D { vertices: a }, Collider::Polygon2D { vertices: b }) => a == b,
        (
            Collider::ConvexPolytope3D { vertices: a },
            Collider::ConvexPolytope3D { vertices: b },
        ) => a == b,
        (
            Collider::ConvexPolytope4D { vertices: a },
            Collider::ConvexPolytope4D { vertices: b },
        ) => a == b,
        _ => false,
    }
}
