use crate::entity::SceneId;
use crate::relation::Relation;
use crate::session::Stamp;
use crate::store::{ErasedStore, Store};

/// The application's store struct; `stores!` writes the impl.
pub trait Stores: Send + 'static {
    type Records: Default + Send + 'static;

    type Snapshot: Send + 'static;

    fn bind(&mut self, scene: SceneId);

    fn publish(&self, into: &mut Self::Records, stamp: Stamp);

    fn snapshot(&self) -> Self::Snapshot;

    fn restore(&mut self, from: &Self::Snapshot, scene: SceneId);

    fn erased(&mut self, visit: &mut dyn FnMut(&'static str, &mut dyn ErasedStore));
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

/// Declares a store struct, its records struct, and its snapshot struct, and implements `Stores`.
#[macro_export]
macro_rules! stores {
    (
        $vis:vis struct $name:ident {
            $( $field:ident : $kind:ident < $row:ty > ),* $(,)?
        }
        $rvis:vis struct $records:ident {
            $( $published:ident : $record:ty ),* $(,)?
        }
        $svis:vis struct $snapshot:ident ;
    ) => {
        $vis struct $name {
            $( pub $field : $crate::$kind<$row>, )*
        }

        #[derive(Default)]
        $rvis struct $records {
            $( pub $published : $crate::RecordBuffer<$record>, )*
        }

        $svis struct $snapshot {
            $( pub $field : <$crate::$kind<$row> as $crate::StoreField>::Snapshot, )*
        }

        impl $crate::Stores for $name {
            type Records = $records;

            type Snapshot = $snapshot;

            fn bind(&mut self, _scene: $crate::SceneId) {
                $( $crate::StoreField::bind(&mut self.$field, _scene); )*
            }

            fn publish(&self, _into: &mut $records, _stamp: $crate::Stamp) {
                $( self.$published.publish(&mut _into.$published, _stamp); )*
            }

            fn snapshot(&self) -> $snapshot {
                $snapshot {
                    $( $field : $crate::StoreField::snapshot(&self.$field), )*
                }
            }

            fn restore(&mut self, _from: &$snapshot, _scene: $crate::SceneId) {
                $( $crate::StoreField::restore(&mut self.$field, &_from.$field, _scene); )*
            }

            fn erased(
                &mut self,
                _visit: &mut dyn FnMut(&'static str, &mut dyn $crate::ErasedStore),
            ) {
                $( _visit(stringify!($field), $crate::StoreField::erased(&mut self.$field)); )*
            }
        }

        $( $crate::__stores_field!($kind, $name, $field, $row); )*
    };
}
