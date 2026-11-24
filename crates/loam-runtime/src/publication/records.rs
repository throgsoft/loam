use bytemuck::{Pod, Zeroable};
use loam_math::Rotor4;
use loam_shape::polytope::SectionScratch;
use loam_shape::{FieldShape, LineMesh, TriangleMesh};
use std::ops::Range;

use super::{RecordCache, Stamp};
use crate::assets::{MaterialId, PreparedId};
use crate::domain::{ChartPose, DomainError};
use crate::entity::Entity;
use crate::view::Vec4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InstanceRecord {
    pub entity: Entity,
    pub geometry: PreparedId,
    pub material: MaterialId,
    pub pose: ChartPose,
    pub image_point: [f32; 3],
}

/// Instance vertex layout of `loam-render`'s `line_raster.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct SegmentRecord {
    pub start: [f32; 3],
    pub _pad0: f32,
    pub end: [f32; 3],
    pub _pad1: f32,
    pub start_color: [f32; 4],
    pub end_color: [f32; 4],
    pub width_px: f32,
    pub _pad2: [f32; 3],
}

/// One point of a view in its image space with a radius in pixels; `loam-render`'s `PointPass` builds its mesh from these.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct PointRecord {
    pub position: [f32; 3],
    pub radius_px: f32,
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct TriangleRecord {
    pub vertices: [[f32; 3]; 3],
    pub color: [f32; 4],
}

/// Screen text in logical points, with the origin measured from the top-left of the area the host's UI leaves free.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextRecord {
    pub origin_pt: [f32; 2],
    pub size_pt: f32,
    pub color: [f32; 4],
}

/// Lines and points in the root image space and screen text that publication systems draw each frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Overlay {
    pub lines: Vec<SegmentRecord>,
    pub points: Vec<PointRecord>,
    text: String,
    labels: Vec<(Range<usize>, TextRecord)>,
}

impl Overlay {
    /// Later text paints over earlier text.
    pub fn push_text(&mut self, text: &str, record: TextRecord) {
        let start = self.text.len();
        self.text.push_str(text);
        self.labels.push((start..self.text.len(), record));
    }

    pub fn text(&self) -> impl Iterator<Item = (&str, &TextRecord)> {
        self.labels
            .iter()
            .map(|(range, record)| (&self.text[range.clone()], record))
    }

    pub(crate) fn clear(&mut self) {
        self.lines.clear();
        self.points.clear();
        self.text.clear();
        self.labels.clear();
    }
}

/// A solid in its view's eye frame, which a field kernel cuts at [`ViewRecords::field_cut`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldRecord {
    pub entity: Entity,
    pub shape: FieldShape,
    pub size: f32,
    pub material: MaterialId,
    pub color: [f32; 4],
    pub position: [f32; 4],
    pub frame: Rotor4,
}

#[derive(Default)]
pub(crate) struct SectionScratchpad {
    pub(crate) rotated: Vec<Vec4>,
    pub(crate) cut: SectionScratch,
    pub(crate) faces: TriangleMesh<3>,
    pub(crate) perimeter: LineMesh<3>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefusalSource {
    Instance,
    Segment,
    Section,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewRefusal {
    pub entity: Entity,
    pub error: DomainError,
    pub source: RefusalSource,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ViewRefusals {
    pub count: u32,
    pub first: Option<ViewRefusal>,
    pub last: Option<ViewRefusal>,
}

impl ViewRefusals {
    pub(crate) fn record(&mut self, entity: Entity, error: DomainError, source: RefusalSource) {
        let refusal = ViewRefusal {
            entity,
            error,
            source,
        };
        self.count = self.count.saturating_add(1);
        self.first.get_or_insert(refusal);
        self.last = Some(refusal);
    }
}

#[derive(Default)]
pub struct ViewRecords {
    pub instances: RecordBuffer<InstanceRecord>,
    pub(crate) segments: Vec<SegmentRecord>,
    pub(crate) triangles: Vec<TriangleRecord>,
    pub(crate) fields: Vec<FieldRecord>,
    pub(crate) field_cut: Option<f32>,
    pub(crate) refusals: ViewRefusals,
    pub(crate) scratch: SectionScratchpad,
    pub(crate) cache: RecordCache,
}

impl ViewRecords {
    pub fn segments(&self) -> &[SegmentRecord] {
        &self.segments
    }

    pub fn triangles(&self) -> &[TriangleRecord] {
        &self.triangles
    }

    pub fn fields(&self) -> &[FieldRecord] {
        &self.fields
    }

    /// `None` when the view draws no fields.
    pub fn field_cut(&self) -> Option<f32> {
        self.field_cut
    }

    pub fn refusals(&self) -> ViewRefusals {
        self.refusals
    }

    /// The publication that last rebuilt these records; a skipped rebuild keeps it.
    pub fn built(&self) -> Stamp {
        self.cache.built()
    }
}

const NO_RECORD: u32 = u32::MAX;

pub struct RecordBuffer<R> {
    rows: Vec<R>,
    entities: Vec<Entity>,
    positions: Vec<u32>,
}

impl<R> RecordBuffer<R> {
    pub fn rows(&self) -> &[R] {
        &self.rows
    }

    fn position(&self, entity: Entity) -> Option<usize> {
        let position = *self.positions.get(entity.key().slot() as usize)?;
        (position != NO_RECORD && self.entities[position as usize] == entity)
            .then_some(position as usize)
    }

    fn upsert(&mut self, entity: Entity, record: R) {
        if let Some(position) = self.position(entity) {
            self.rows[position] = record;
            return;
        }
        let slot = entity.key().slot() as usize;
        if slot >= self.positions.len() {
            self.positions.resize(slot + 1, NO_RECORD);
        }
        self.positions[slot] = self.rows.len() as u32;
        self.rows.push(record);
        self.entities.push(entity);
    }

    fn clear(&mut self) {
        self.rows.clear();
        self.entities.clear();
        self.positions.fill(NO_RECORD);
    }

    pub(crate) fn update(&mut self, entity: Entity, record: R) -> bool {
        let Some(position) = self.position(entity) else {
            return false;
        };
        self.rows[position] = record;
        true
    }

    pub(crate) fn replace(&mut self, records: impl Iterator<Item = (Entity, R)>) {
        self.clear();
        for (entity, record) in records {
            self.upsert(entity, record);
        }
    }
}

impl<R> Default for RecordBuffer<R> {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            entities: Vec::new(),
            positions: Vec::new(),
        }
    }
}
