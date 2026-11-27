use super::{DomainError, DomainSpace, Pose};
use crate::entity::{Entity, SceneId};
use crate::store::Store;
use crate::view::{ViewId, ViewSpec, ViewStyle, ViewTarget};

pub(super) struct ViewEntry<S: DomainSpace> {
    pub(super) eye: Entity,
    pub(super) subject: Option<Entity>,
    pub(super) style: ViewStyle<S>,
}

impl<S: DomainSpace> Clone for ViewEntry<S> {
    fn clone(&self) -> Self {
        Self {
            eye: self.eye,
            subject: self.subject,
            style: self.style.clone(),
        }
    }
}

fn same_style<S: DomainSpace>(a: &ViewStyle<S>, b: &ViewStyle<S>) -> bool {
    let ViewStyle {
        enabled,
        edges,
        section_edges,
        section_faces,
        fields,
        mapping,
    } = a;
    let ViewStyle {
        enabled: other_enabled,
        edges: other_edges,
        section_edges: other_section_edges,
        section_faces: other_section_faces,
        fields: other_fields,
        mapping: other_mapping,
    } = b;
    enabled == other_enabled
        && edges == other_edges
        && section_edges == other_section_edges
        && section_faces == other_section_faces
        && fields == other_fields
        && std::sync::Arc::ptr_eq(mapping, other_mapping)
}

// Indexed by view: a revision changes whenever what the view draws from changes, so publication rebuilds it.
pub(super) struct ViewTable<S: DomainSpace> {
    entries: Vec<ViewEntry<S>>,
    targets: Vec<ViewTarget>,
    revisions: Vec<u32>,
    stamps: Vec<ViewStyle<S>>,
}

impl<S: DomainSpace> ViewTable<S> {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            targets: Vec::new(),
            revisions: Vec::new(),
            stamps: Vec::new(),
        }
    }

    pub(super) fn entries(&self) -> &[ViewEntry<S>] {
        &self.entries
    }

    pub(super) fn targets(&self) -> &[ViewTarget] {
        &self.targets
    }

    pub(super) fn get(&self, id: ViewId) -> Option<&ViewEntry<S>> {
        self.entries.get(id.index())
    }

    pub(super) fn target(&self, id: ViewId) -> Option<&ViewTarget> {
        self.targets.get(id.index())
    }

    pub(super) fn revision(&self, id: ViewId) -> Option<u32> {
        self.revisions.get(id.index()).copied()
    }

    pub(super) fn style_mut(&mut self, id: ViewId) -> Option<&mut ViewStyle<S>> {
        let spec = self.entries.get_mut(id.index())?;
        Some(&mut spec.style)
    }

    pub(super) fn add(
        &mut self,
        spec: ViewSpec<S>,
        poses: &Store<Pose<S>>,
    ) -> Result<ViewId, DomainError> {
        if !poses.contains(spec.eye) {
            return Err(DomainError::Stale(spec.eye));
        }
        if let Some(subject) = spec.subject {
            if !poses.contains(subject) {
                return Err(DomainError::Stale(subject));
            }
        }
        let id = ViewId::new(self.entries.len());
        self.targets.push(ViewTarget {
            view: id,
            image: spec.image,
        });
        self.revisions.push(0);
        self.stamps.push(spec.style.clone());
        self.entries.push(ViewEntry {
            eye: spec.eye,
            subject: spec.subject,
            style: spec.style,
        });
        Ok(id)
    }

    pub(super) fn set_eye(
        &mut self,
        id: ViewId,
        eye: Entity,
        poses: &Store<Pose<S>>,
    ) -> Result<(), DomainError> {
        let index = id.index();
        if self.entries.get(index).is_none() {
            return Err(DomainError::UnknownView(id));
        }
        if !poses.contains(eye) {
            return Err(DomainError::Stale(eye));
        }
        if self.entries[index].eye == eye {
            return Ok(());
        }
        self.entries[index].eye = eye;
        self.revisions[index] = self.revisions[index].wrapping_add(1);
        Ok(())
    }

    pub(super) fn set_subject(
        &mut self,
        id: ViewId,
        subject: Option<Entity>,
        poses: &Store<Pose<S>>,
    ) -> Result<(), DomainError> {
        let index = id.index();
        if self.entries.get(index).is_none() {
            return Err(DomainError::UnknownView(id));
        }
        if let Some(subject) = subject {
            if !poses.contains(subject) {
                return Err(DomainError::Stale(subject));
            }
        }
        if self.entries[index].subject == subject {
            return Ok(());
        }
        self.entries[index].subject = subject;
        self.revisions[index] = self.revisions[index].wrapping_add(1);
        Ok(())
    }

    pub(super) fn synchronize(&mut self) {
        for (index, view) in self.entries.iter().enumerate() {
            let Some(stamp) = self.stamps.get_mut(index) else {
                continue;
            };
            if same_style(&view.style, stamp) {
                continue;
            }
            *stamp = view.style.clone();
            if let Some(revision) = self.revisions.get_mut(index) {
                *revision = revision.wrapping_add(1);
            }
        }
    }

    pub(super) fn restore(
        &mut self,
        entries: &[ViewEntry<S>],
        targets: &[ViewTarget],
        from: SceneId,
        scene: SceneId,
    ) {
        let mut views = entries.to_vec();
        let mut revisions = Vec::with_capacity(views.len());
        for (index, view) in views.iter_mut().enumerate() {
            if view.eye.scene() == from {
                view.eye = Entity::new(scene, view.eye.key());
            }
            view.subject = view.subject.map(|subject| {
                if subject.scene() == from {
                    Entity::new(scene, subject.key())
                } else {
                    subject
                }
            });
            revisions.push(
                self.revisions
                    .get(index)
                    .copied()
                    .unwrap_or(0)
                    .wrapping_add(1),
            );
        }
        self.entries = views;
        self.targets.clear();
        self.targets.extend_from_slice(targets);
        self.revisions = revisions;
        self.stamps = self.entries.iter().map(|view| view.style.clone()).collect();
    }
}
