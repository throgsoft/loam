use std::ops::Range;

use super::{FieldRecord, SegmentRecord, TriangleRecord};
use crate::domain::{DomainSpace, Instance, Pose};
use crate::entity::Entity;
use crate::publication::Stamp;
use crate::store::{Change, Cursor, Store};

const NO_OUTPUT: u32 = u32::MAX;

#[derive(Clone)]
pub(crate) struct EntityOutput {
    pub(crate) entity: Entity,
    pub(crate) section_segments: Range<usize>,
    pub(crate) line_segments: Range<usize>,
    pub(crate) triangles: Range<usize>,
    pub(crate) fields: Range<usize>,
    pub(crate) instance: bool,
}

#[derive(Default)]
pub(crate) struct ViewOutputCache {
    entries: Vec<EntityOutput>,
    positions: Vec<u32>,
}

impl ViewOutputCache {
    fn clear(&mut self) {
        self.entries.clear();
        self.positions.fill(NO_OUTPUT);
    }

    pub(crate) fn push(&mut self, output: EntityOutput) {
        let slot = output.entity.key().slot() as usize;
        if slot >= self.positions.len() {
            self.positions.resize(slot + 1, NO_OUTPUT);
        }
        self.positions[slot] = self.entries.len() as u32;
        self.entries.push(output);
    }

    pub(crate) fn get(&self, entity: Entity) -> Option<&EntityOutput> {
        let position = *self.positions.get(entity.key().slot() as usize)?;
        if position == NO_OUTPUT || self.entries[position as usize].entity != entity {
            return None;
        }
        Some(&self.entries[position as usize])
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut EntityOutput> {
        self.entries.iter_mut()
    }
}

#[derive(Clone, Copy, Default)]
enum PendingBuild {
    #[default]
    Current,
    Patch {
        poses: Cursor,
        attachments: Cursor,
    },
    Rebuild,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Planned {
    Current,
    Patch(usize),
    Rebuild,
}

pub(crate) struct Motion<'a> {
    pub(crate) generation: u64,
    pub(crate) moving: &'a [Entity],
}

pub(crate) struct ViewState {
    pub(crate) revision: u32,
    pub(crate) eye: Entity,
    pub(crate) patchable: bool,
}

pub(crate) struct PatchScratch<'a> {
    pub(crate) changed: &'a [Entity],
    pub(crate) output: &'a ViewOutputCache,
    pub(crate) segments: &'a mut Vec<SegmentRecord>,
    pub(crate) triangles: &'a mut Vec<TriangleRecord>,
    pub(crate) fields: &'a mut Vec<FieldRecord>,
}

#[derive(Default)]
pub(crate) struct RecordCache {
    output: ViewOutputCache,
    patch_segments: Vec<SegmentRecord>,
    patch_triangles: Vec<TriangleRecord>,
    patch_fields: Vec<FieldRecord>,
    changed: Vec<Entity>,
    poses: Cursor,
    attachments: Cursor,
    pending: PendingBuild,
    revision: u32,
    built: Stamp,
    alpha: f32,
    merged: Vec<Entity>,
    generation: u64,
}

impl RecordCache {
    pub(crate) fn built(&self) -> Stamp {
        self.built
    }

    // A plan with nothing to build commits at once.
    pub(crate) fn plan<S: DomainSpace>(
        &mut self,
        poses: &Store<Pose<S>>,
        instances: &Store<Instance>,
        motion: Option<Motion<'_>>,
        alpha: f32,
        view: ViewState,
    ) -> Planned {
        let mut pose_cursor = self.poses;
        let pose_changes = poses.changes(&mut pose_cursor);
        let pose_resync = pose_changes.is_resync();
        self.changed.clear();
        let mut pose_removed = false;
        for change in pose_changes {
            match change {
                Change::Row(entity, _) => self.changed.push(entity),
                Change::Removed(_) => pose_removed = true,
            }
        }
        if let Some(motion) = motion {
            let logged = !self.changed.is_empty();
            let ticked = self.generation != motion.generation;
            if ticked {
                self.changed.extend_from_slice(&self.merged);
                self.merged.clear();
                self.merged.extend_from_slice(motion.moving);
                self.generation = motion.generation;
            }
            if ticked || alpha != self.alpha {
                self.changed.extend_from_slice(motion.moving);
                if logged || ticked {
                    self.changed.sort_unstable();
                    self.changed.dedup();
                }
            }
        }
        self.alpha = alpha;
        let mut attachment_cursor = self.attachments;
        let mut attachment_changes = instances.changes(&mut attachment_cursor);
        let attachment_resync = attachment_changes.is_resync();
        let attached = attachment_changes.next().is_some();
        if !pose_resync
            && !pose_removed
            && !attachment_resync
            && !attached
            && self.revision == view.revision
            && !self.changed.contains(&view.eye)
        {
            if self.changed.is_empty() {
                self.poses = pose_cursor;
                self.attachments = attachment_cursor;
                self.pending = PendingBuild::Current;
                return Planned::Current;
            }
            if view.patchable {
                self.pending = PendingBuild::Patch {
                    poses: pose_cursor,
                    attachments: attachment_cursor,
                };
                return Planned::Patch(self.changed.len());
            }
        }
        self.pending = PendingBuild::Rebuild;
        Planned::Rebuild
    }

    pub(crate) fn pending(&self) -> Planned {
        match self.pending {
            PendingBuild::Current => Planned::Current,
            PendingBuild::Patch { .. } => Planned::Patch(self.changed.len()),
            PendingBuild::Rebuild => Planned::Rebuild,
        }
    }

    pub(crate) fn patch(&mut self) -> PatchScratch<'_> {
        PatchScratch {
            changed: &self.changed,
            output: &self.output,
            segments: &mut self.patch_segments,
            triangles: &mut self.patch_triangles,
            fields: &mut self.patch_fields,
        }
    }

    pub(crate) fn commit_patch(&mut self, stamp: Stamp, patched: bool) {
        let PendingBuild::Patch { poses, attachments } = self.pending else {
            unreachable!("a patch commits only after plan chose a patch");
        };
        self.poses = poses;
        self.attachments = attachments;
        if patched {
            self.built = stamp;
        }
    }

    pub(crate) fn reset(&mut self) -> &mut ViewOutputCache {
        self.output.clear();
        &mut self.output
    }

    pub(crate) fn commit_rebuild<S: DomainSpace>(
        &mut self,
        poses: &Store<Pose<S>>,
        instances: &Store<Instance>,
        revision: u32,
        stamp: Stamp,
    ) {
        poses.catch_up(&mut self.poses);
        instances.catch_up(&mut self.attachments);
        self.revision = revision;
        self.built = stamp;
    }
}
