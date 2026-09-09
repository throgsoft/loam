use crate::entity::SceneId;
use crate::relation::Relation;
use crate::session::Stamp;
use crate::store::{ErasedStore, Store};

/// The application's store struct; `stores!` writes the impl and its two companion types.
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
    (Published, $name:ident, $field:ident, $row:ty) => {
        $crate::__stores_field!(Store, $name, $field, $row);
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

#[doc(hidden)]
#[macro_export]
macro_rules! __stores_records {
    (@emit [$($acc:tt)*]) => {
        #[derive(Default)]
        pub struct Records {
            $($acc)*
        }
    };
    (@acc [$($acc:tt)*] Published $field:ident : $row:ty ; $($rest:tt)*) => {
        $crate::__stores_records!(
            @acc [$($acc)* pub $field : $crate::RecordBuffer<<$row as $crate::Publish>::Record>,]
            $($rest)*
        )
    };
    (@acc [$($acc:tt)*] $kind:ident $field:ident : $row:ty ; $($rest:tt)*) => {
        $crate::__stores_records!(@acc [$($acc)*] $($rest)*)
    };
    (@acc [$($acc:tt)*]) => {
        $crate::__stores_records!(@emit [$($acc)*])
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __stores_publish {
    (Published, $store:expr, $buffer:expr, $stamp:expr) => {
        $crate::Store::publish($store, $buffer, $stamp);
    };
    ($kind:ident, $store:expr, $buffer:expr, $stamp:expr) => {};
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
            $crate::__stores_records!(@acc [] $( $kind $field : $row ; )*);

            pub struct Snapshot {
                $( pub $field : <$crate::$kind<$row> as $crate::StoreField>::Snapshot, )*
            }

            impl $crate::Stores for $name {
                type Records = Records;

                type Snapshot = Snapshot;

                fn bind(&mut self, _scene: $crate::SceneId) {
                    $( $crate::StoreField::bind(&mut self.$field, _scene); )*
                }

                fn publish(&self, _into: &mut Records, _stamp: $crate::Stamp) {
                    $( $crate::__stores_publish!($kind, &self.$field, &mut _into.$field, _stamp); )*
                }

                fn snapshot(&self) -> Snapshot {
                    Snapshot {
                        $( $field : $crate::StoreField::snapshot(&self.$field), )*
                    }
                }

                fn restore(&mut self, _from: &Snapshot, _scene: $crate::SceneId) {
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
    };
}
