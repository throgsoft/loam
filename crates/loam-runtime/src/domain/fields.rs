use loam_shape::field::{DistanceField, FieldKind};

use super::{put_row, DomainError, DomainSpace, Pose};
use crate::command::Rejection;
use crate::entity::{Entity, SceneId};
use crate::field::{
    self, FieldCompiler, FieldCost, FieldError, FieldNode, FieldOp, FieldPrimitive,
};
use crate::store::{Owner, Store, StoreError, StoreField, StoreSnapshot};

#[derive(Clone, Debug)]
pub struct Field {
    pub kind: FieldKind,
    pub op: FieldOp,
    pub operands: Vec<Entity>,
}

#[derive(Clone, Debug, Default)]
pub struct FieldProgram {
    pub(crate) primitives: Vec<FieldPrimitive>,
    pub(crate) program: Vec<u32>,
    pub(crate) nodes: Vec<FieldNode>,
    pub(crate) stack: u32,
    pub(crate) dimension: u32,
    pub(crate) kind: FieldKind,
}

impl FieldProgram {
    pub fn evaluate(&self, point: [f32; 4]) -> Result<(f32, FieldKind), FieldError> {
        field::evaluate(&self.program, &self.primitives, point).map(|value| (value, self.kind))
    }

    pub fn evaluate_bounded(
        &self,
        point: [f32; 4],
        tolerance: f32,
    ) -> Result<(f32, FieldKind), FieldError> {
        field::evaluate_bounded(
            &self.program,
            &self.primitives,
            &self.nodes,
            point,
            tolerance,
        )
        .map(|value| (value, self.kind))
    }
}

impl DistanceField for FieldProgram {
    fn field_kind(&self) -> FieldKind {
        self.kind
    }

    fn dimension(&self) -> u32 {
        self.dimension
    }

    fn distance(&self, point: [f32; 4]) -> f32 {
        field::evaluate_bounded(&self.program, &self.primitives, &self.nodes, point, 0.0)
            .unwrap_or(field::FIELD_FAR)
    }

    fn error_at(&self, point: [f32; 4]) -> f32 {
        field::evaluate_error(
            &self.program,
            &self.primitives,
            &self.nodes,
            point,
            self.kind,
        )
        .unwrap_or(f32::INFINITY)
    }
}

pub(super) struct Fields {
    store: Option<Store<Field>>,
    compiler: FieldCompiler,
}

impl Fields {
    pub(super) fn new(store: Option<Store<Field>>) -> Self {
        Self {
            store,
            compiler: FieldCompiler::new(),
        }
    }

    pub(super) fn enabled(&self) -> bool {
        self.store.is_some()
    }

    pub(super) fn store(&self) -> Option<&Store<Field>> {
        self.store.as_ref()
    }

    pub(super) fn get_mut(&mut self, entity: Entity) -> Option<&mut Field> {
        self.store.as_mut()?.get_mut(entity)
    }

    pub(super) fn remove(&mut self, entity: Entity) -> Result<Field, StoreError> {
        self.store
            .as_mut()
            .ok_or(StoreError::Missing(entity))?
            .remove(entity)
    }

    pub(super) fn attach(&mut self, entity: Entity, field: Field) -> Result<(), Rejection> {
        let store = self
            .store
            .as_mut()
            .ok_or(Rejection::Unsupported("fields"))?;
        put_row(store, entity, field)
    }

    pub(super) fn compile<S: DomainSpace>(
        &mut self,
        space: &S,
        poses: &Store<Pose<S>>,
    ) -> Result<FieldCost, DomainError> {
        let store = self
            .store
            .as_ref()
            .ok_or(DomainError::Unsupported("fields"))?;
        self.compiler.compile(space, store, poses)
    }

    pub(super) fn program(&self) -> &FieldProgram {
        self.compiler.program()
    }

    pub(super) fn boundary(&mut self) {
        if let Some(store) = &mut self.store {
            store.boundary();
        }
    }

    pub(super) fn release(&mut self, entity: Entity) {
        if let Some(store) = &mut self.store {
            StoreField::release(store, entity, Owner::new());
        }
    }

    pub(super) fn snapshot(&self) -> Option<StoreSnapshot<Field>> {
        self.store.as_ref().map(StoreField::snapshot)
    }

    pub(super) fn restore(
        &mut self,
        snapshot: Option<&StoreSnapshot<Field>>,
        from: SceneId,
        scene: SceneId,
    ) {
        if let (Some(store), Some(snapshot)) = (&mut self.store, snapshot) {
            StoreField::restore(store, snapshot, scene, Owner::new());
            for field in store.rows_mut_untracked() {
                for operand in &mut field.operands {
                    if operand.scene() == from {
                        *operand = Entity::new(scene, operand.key());
                    }
                }
            }
        }
        self.compiler.invalidate();
    }
}
