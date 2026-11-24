use super::{Domain, DomainError, DomainHandle, DomainId, DomainOwner, DomainSpace, TypedDomain};
use crate::assets::PreparedGeometry;
use crate::entity::RuntimeId;
use crate::image::Views;
use crate::view::Pick;

pub struct Domains {
    runtime: RuntimeId,
    list: Vec<Box<dyn DomainOwner>>,
}

impl Domains {
    pub(crate) fn new(runtime: RuntimeId) -> Self {
        Self {
            runtime,
            list: Vec::new(),
        }
    }

    pub(crate) fn next_id(&self) -> DomainId {
        DomainId::new(self.list.len())
    }

    pub(crate) fn push(&mut self, domain: Box<dyn DomainOwner>) {
        self.list.push(domain);
    }

    pub fn runtime(&self) -> RuntimeId {
        self.runtime
    }

    pub fn len(&self) -> usize {
        self.list.len()
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    pub fn get(&self, id: DomainId) -> Option<&dyn Domain> {
        self.list.get(id.index()).map(|domain| domain.as_ref() as _)
    }

    pub(crate) fn owner(&self, id: DomainId) -> Option<&dyn DomainOwner> {
        self.list.get(id.index()).map(|domain| domain.as_ref())
    }

    pub(crate) fn facade(&mut self, id: DomainId) -> Option<&mut dyn DomainOwner> {
        self.list.get_mut(id.index()).map(|domain| domain.as_mut())
    }

    pub fn iter(&self) -> impl Iterator<Item = &dyn Domain> {
        self.list.iter().map(|domain| domain.as_ref() as _)
    }

    pub(crate) fn owned(&self) -> impl Iterator<Item = &dyn DomainOwner> {
        self.list.iter().map(|domain| domain.as_ref())
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut dyn DomainOwner> {
        self.list.iter_mut().map(|domain| domain.as_mut())
    }

    pub(crate) fn synchronize(&mut self) {
        for domain in &mut self.list {
            domain.synchronize();
        }
    }

    pub fn typed<S: DomainSpace>(
        &mut self,
        handle: DomainHandle<S>,
    ) -> Result<&mut TypedDomain<S>, DomainError> {
        if handle.runtime != self.runtime {
            return Err(DomainError::ForeignRuntime);
        }
        self.typed_at(handle.id)
    }

    pub(crate) fn typed_at<S: DomainSpace>(
        &mut self,
        id: DomainId,
    ) -> Result<&mut TypedDomain<S>, DomainError> {
        self.list
            .get_mut(id.index())
            .ok_or(DomainError::UnknownDomain(id))?
            .as_any_mut()
            .downcast_mut::<TypedDomain<S>>()
            .ok_or(DomainError::SpaceMismatch(id))
    }

    pub fn named<S: DomainSpace>(
        &self,
        name: &'static str,
    ) -> Result<DomainHandle<S>, DomainError> {
        let mut matched = self.list.iter().filter(|domain| domain.name() == name);
        let Some(domain) = matched.next() else {
            return Err(DomainError::UnknownDomainName(name));
        };
        if matched.next().is_some() {
            return Err(DomainError::AmbiguousDomainName(name));
        }
        domain
            .as_any()
            .downcast_ref::<TypedDomain<S>>()
            .map(TypedDomain::handle)
            .ok_or(DomainError::SpaceMismatch(domain.id()))
    }

    pub fn read<S: DomainSpace>(
        &self,
        handle: DomainHandle<S>,
    ) -> Result<&TypedDomain<S>, DomainError> {
        if handle.runtime != self.runtime {
            return Err(DomainError::ForeignRuntime);
        }
        let domain = self
            .list
            .get(handle.id.index())
            .ok_or(DomainError::UnknownDomain(handle.id))?;
        domain
            .as_any()
            .downcast_ref::<TypedDomain<S>>()
            .ok_or(DomainError::SpaceMismatch(handle.id))
    }

    pub fn pick(
        &self,
        views: &Views,
        prepared: &[PreparedGeometry],
        ndc: [f32; 2],
    ) -> Option<Pick> {
        self.nearest(views, prepared, ndc, false)
    }

    pub fn pick_lifted(
        &self,
        views: &Views,
        prepared: &[PreparedGeometry],
        ndc: [f32; 2],
    ) -> Option<Pick> {
        self.nearest(views, prepared, ndc, true)
    }

    fn nearest(
        &self,
        views: &Views,
        prepared: &[PreparedGeometry],
        ndc: [f32; 2],
        lifted_only: bool,
    ) -> Option<Pick> {
        let mut nearest: Option<Pick> = None;
        for domain in self.iter() {
            for target in domain.views() {
                if lifted_only
                    && !domain
                        .view(target.view)
                        .is_some_and(|summary| summary.ray_lift)
                {
                    continue;
                }
                let Some(ray) = views.ray(target.image, ndc) else {
                    continue;
                };
                let Some(pick) = domain.pick(target.view, &ray, views, prepared) else {
                    continue;
                };
                if nearest.is_none_or(|best| pick.depth > best.depth) {
                    nearest = Some(pick);
                }
            }
        }
        nearest
    }
}
