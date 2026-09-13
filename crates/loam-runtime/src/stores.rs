use crate::entity::{Entity, SceneId};
use crate::relation::Relation;
use crate::store::Store;

/// The application's storage contract; only `Session` calls its lifetime hooks.
pub trait Stores: Send + 'static {
    type Snapshot: Send + 'static;

    fn bind(&mut self, scene: SceneId);

    fn boundary(&mut self);

    fn release(&mut self, entity: Entity);

    fn snapshot(&self) -> Self::Snapshot;

    fn restore(&mut self, from: &Self::Snapshot, scene: SceneId);
}

pub trait HasStore<T>: Stores {
    fn store(&self) -> &Store<T>;

    fn store_mut(&mut self) -> &mut Store<T>;
}

pub trait HasRelation<T>: Stores {
    fn relation(&self) -> &Relation<T>;

    fn relation_mut(&mut self) -> &mut Relation<T>;
}

#[doc(hidden)]
#[macro_export]
macro_rules! __stores_field {
    (Store, $name:ident, $field:ident, $row:ty) => {
        impl $crate::HasStore<$row> for $name {
            fn store(&self) -> &$crate::Store<$row> {
                &self.$field
            }

            fn store_mut(&mut self) -> &mut $crate::Store<$row> {
                &mut self.$field
            }
        }
    };
    (Relation, $name:ident, $field:ident, $row:ty) => {
        impl $crate::HasRelation<$row> for $name {
            fn relation(&self) -> &$crate::Relation<$row> {
                &self.$field
            }

            fn relation_mut(&mut self) -> &mut $crate::Relation<$row> {
                &mut self.$field
            }
        }
    };
    (Value, $name:ident, $field:ident, $row:ty) => {};
}

/// Declares the store struct and implements `Stores`; its records and snapshot types are `<A as Stores>::Records` and `::Snapshot`.
#[macro_export]
macro_rules! stores {
    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident {
            $( $field:ident : $kind:ident < $row:ty > ),* $(,)?
        }
    ) => {
        $(#[$attr])*
        $vis struct $name {
            $( pub $field : $crate::$kind<$row>, )*
        }

        const _: () = {
            pub struct Snapshot {
                $( pub $field : <$crate::$kind<$row> as $crate::StoreField>::Snapshot, )*
            }

            impl $crate::Stores for $name {
                type Snapshot = Snapshot;

                fn bind(&mut self, _scene: $crate::SceneId) {
                    $( $crate::StoreField::bind(&mut self.$field, _scene); )*
                }

                fn boundary(&mut self) {
                    $( $crate::StoreField::boundary(&mut self.$field); )*
                }

                fn release(&mut self, _entity: $crate::Entity) {
                    $( $crate::StoreField::release(&mut self.$field, _entity); )*
                }

                fn snapshot(&self) -> Snapshot {
                    Snapshot {
                        $( $field : $crate::StoreField::snapshot(&self.$field), )*
                    }
                }

                fn restore(&mut self, _from: &Snapshot, _scene: $crate::SceneId) {
                    $( $crate::StoreField::restore(&mut self.$field, &_from.$field, _scene); )*
                }
            }

            $( $crate::__stores_field!($kind, $name, $field, $row); )*
        };
    };
}
