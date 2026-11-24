use crate::app_state::Stores;
use crate::assets::PreparedGeometry;
use crate::command::{Command, Commands};
use crate::domain::{ChartCommand, ChartPoint, DomainError, DomainId, Domains};
use crate::entity::Entity;
use crate::image::{ImageRay, ImageSpaceId, Views};
use crate::input::{Input, PointerButton, PointerPhase};
use crate::view::{Pick, Vec3, ViewId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragError {
    NoPick,
    NotGrabbed,
    /// The pointer ray runs parallel to the drag plane, so no point on it is nearer than another.
    Ambiguous(&'static str),
    Domain(DomainError),
}

impl From<DomainError> for DragError {
    fn from(error: DomainError) -> Self {
        Self::Domain(error)
    }
}

pub(crate) const VELOCITY_SAMPLES: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drag {
    pub entity: Entity,
    pub domain: DomainId,
    pub view: ViewId,
    pub image: ImageSpaceId,
    pub(crate) plane: [f32; 3],
    pub(crate) normal: [f32; 3],
    pub(crate) depth: f32,
    pub(crate) center: [f32; 3],
    pub(crate) at: [f32; 3],
    pub(crate) samples: [([f32; 3], f64); VELOCITY_SAMPLES],
    pub(crate) sampled: usize,
    pub(crate) opened: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DragRelease {
    pub entity: Entity,
    pub domain: DomainId,
    pub image: ImageSpaceId,
    /// In the view's image space, per second.
    pub velocity: [f32; 3],
}

const PLANE_EPSILON: f32 = 1e-6;
const VELOCITY_WINDOW_SECONDS: f64 = 0.1;
// Keeps the ring from filling with one frame of high-rate pointer events.
const SAMPLE_SPACING_SECONDS: f64 = 0.01;
const RELEASE_STALE_SECONDS: f64 = 0.12;

impl Drag {
    pub(crate) fn plane_at(&self, eye: [f32; 3], forward: [f32; 3]) -> [f32; 3] {
        (Vec3::from(eye) + Vec3::from(forward) * self.depth).to_array()
    }

    pub(crate) fn meet(ray: &ImageRay, plane: [f32; 3], normal: [f32; 3]) -> Option<[f32; 3]> {
        let normal = Vec3::from(normal);
        let along = Vec3::from(ray.direction).dot(normal);
        if along.abs() <= PLANE_EPSILON {
            return None;
        }
        Some(ray.at((Vec3::from(plane) - Vec3::from(ray.origin)).dot(normal) / along))
    }

    pub(crate) fn moved(&self, at: [f32; 3]) -> [f32; 3] {
        (Vec3::from(self.center) + Vec3::from(at) - Vec3::from(self.plane)).to_array()
    }

    pub(crate) fn sample(&mut self, at: [f32; 3], time: f64) {
        if time < self.newest_time() {
            self.at = at;
            return;
        }
        if time - self.opened < SAMPLE_SPACING_SECONDS {
            self.samples[self.sampled.saturating_sub(1) % VELOCITY_SAMPLES] = (at, time);
        } else {
            self.samples[self.sampled % VELOCITY_SAMPLES] = (at, time);
            self.sampled += 1;
            self.opened = time;
        }
        self.at = at;
    }

    pub(crate) fn newest_time(&self) -> f64 {
        self.samples[self.sampled.saturating_sub(1) % VELOCITY_SAMPLES].1
    }

    pub(crate) fn velocity(&self) -> [f32; 3] {
        let held = self.sampled.min(VELOCITY_SAMPLES);
        if held < 2 {
            return [0.0; 3];
        }
        let newest = self.samples[(self.sampled - 1) % VELOCITY_SAMPLES];
        let mut oldest = newest;
        for back in 1..held {
            let sample = self.samples[(self.sampled - 1 - back) % VELOCITY_SAMPLES];
            if back > 1 && newest.1 - sample.1 > VELOCITY_WINDOW_SECONDS {
                break;
            }
            oldest = sample;
        }
        let elapsed = (newest.1 - oldest.1) as f32;
        if elapsed <= 0.0 {
            return [0.0; 3];
        }
        ((Vec3::from(newest.0) - Vec3::from(oldest.0)) / elapsed).to_array()
    }

    pub(crate) fn released(&self, velocity: [f32; 3]) -> DragRelease {
        DragRelease {
            entity: self.entity,
            domain: self.domain,
            image: self.image,
            velocity,
        }
    }
}

#[derive(Default)]
pub(crate) struct Manipulation {
    drag: Option<Drag>,
    pointer: Option<[f32; 2]>,
}

impl Manipulation {
    pub(crate) fn dragging(&self) -> Option<Drag> {
        self.drag
    }

    pub(crate) fn grab<A: Stores>(
        &mut self,
        domains: &Domains,
        views: &Views,
        prepared: &[PreparedGeometry],
        commands: &mut Commands<A>,
        ndc: [f32; 2],
        time: f64,
    ) -> Result<Pick, DragError> {
        let pick = domains
            .pick_lifted(views, prepared, ndc)
            .ok_or(DragError::NoPick)?;
        self.grab_pick(domains, views, commands, pick, time)
    }

    pub(crate) fn grab_pick<A: Stores>(
        &mut self,
        domains: &Domains,
        views: &Views,
        commands: &mut Commands<A>,
        pick: Pick,
        time: f64,
    ) -> Result<Pick, DragError> {
        let domain = domains
            .get(pick.domain)
            .ok_or(DomainError::UnknownDomain(pick.domain))?;
        let center = domain
            .image_of(pick.view, pick.entity)
            .ok_or(DomainError::Stale(pick.entity))?;
        let eye = views.root().eye;
        let plane = pick.image_point;
        let normal = eye.forward;
        let depth = (Vec3::from(plane) - Vec3::from(eye.position)).dot(Vec3::from(normal));
        let hit = pick.hit.ok_or(DomainError::Unsupported("view ray lift"))?;
        if let Some(held) = self.drag.take() {
            commands.submit(Command::Chart(
                held.domain,
                ChartCommand::Release {
                    entity: held.entity,
                },
            ));
        }
        self.drag = Some(Drag {
            entity: pick.entity,
            domain: pick.domain,
            view: pick.view,
            image: pick.image,
            plane,
            normal,
            depth,
            center,
            at: plane,
            samples: [(plane, time); VELOCITY_SAMPLES],
            sampled: 1,
            opened: time,
        });
        commands.submit(Command::Chart(
            pick.domain,
            ChartCommand::Grab {
                entity: pick.entity,
                point: hit,
            },
        ));
        Ok(pick)
    }

    pub(crate) fn drag<A: Stores>(
        &mut self,
        domains: &Domains,
        views: &Views,
        commands: &mut Commands<A>,
        ndc: [f32; 2],
        time: f64,
    ) -> Result<ChartPoint, DragError> {
        let mut drag = self.drag.ok_or(DragError::NotGrabbed)?;
        let domain = domains
            .get(drag.domain)
            .ok_or(DomainError::UnknownDomain(drag.domain))?;
        let name = domain
            .view(drag.view)
            .ok_or(DomainError::UnknownView(drag.view))?
            .name;
        let ray = views
            .ray(drag.image, ndc)
            .ok_or(DomainError::Unsupported("image space"))?;
        let eye = views.root().eye;
        let forward = eye.forward;
        let plane = drag.plane_at(eye.position, forward);
        let at = Drag::meet(&ray, plane, forward).ok_or(DragError::Ambiguous(name))?;
        let point = domain.lift_origin(
            drag.view,
            &ImageRay {
                origin: drag.moved(at),
                direction: drag.normal,
            },
        )?;
        drag.sample(at, time);
        self.drag = Some(drag);
        commands.submit(Command::Chart(
            drag.domain,
            ChartCommand::Move {
                entity: drag.entity,
                point,
            },
        ));
        Ok(point)
    }

    fn finish<A: Stores>(commands: &mut Commands<A>, drag: Drag) {
        commands.submit(Command::Chart(
            drag.domain,
            ChartCommand::Release {
                entity: drag.entity,
            },
        ));
    }

    fn throw<A: Stores>(
        domains: &Domains,
        commands: &mut Commands<A>,
        drag: Drag,
        velocity: [f32; 3],
    ) -> DragRelease {
        Self::finish(commands, drag);
        let lifted = domains.get(drag.domain).and_then(|domain| {
            domain
                .lift_tangent(drag.view, drag.moved(drag.at), velocity)
                .ok()
        });
        if let Some(lifted) = lifted {
            commands.submit(Command::Chart(
                drag.domain,
                ChartCommand::Throw {
                    entity: drag.entity,
                    view: drag.view,
                    velocity: lifted,
                },
            ));
        }
        drag.released(velocity)
    }

    pub(crate) fn release<A: Stores>(
        &mut self,
        domains: &Domains,
        commands: &mut Commands<A>,
    ) -> Option<DragRelease> {
        let drag = self.drag.take()?;
        Some(Self::throw(domains, commands, drag, drag.velocity()))
    }

    pub(crate) fn release_at<A: Stores>(
        &mut self,
        domains: &Domains,
        commands: &mut Commands<A>,
        time: f64,
    ) -> Option<DragRelease> {
        let drag = self.drag.take()?;
        let age = time - drag.newest_time();
        let velocity = if (0.0..=RELEASE_STALE_SECONDS).contains(&age) {
            drag.velocity()
        } else {
            [0.0; 3]
        };
        Some(Self::throw(domains, commands, drag, velocity))
    }

    pub(crate) fn cancel<A: Stores>(&mut self, commands: &mut Commands<A>) -> Option<DragRelease> {
        let drag = self.drag.take()?;
        Self::finish(commands, drag);
        Some(drag.released([0.0; 3]))
    }

    pub(crate) fn carry<A: Stores>(
        &mut self,
        domains: &Domains,
        views: &Views,
        commands: &mut Commands<A>,
        input: &Input,
        button: PointerButton,
        mut pick: impl FnMut([f32; 2]) -> Option<Pick>,
    ) -> Option<DragRelease> {
        if let Some(pointer) = input.pointers.last() {
            self.pointer = Some(pointer.ndc);
        }
        let mut steered = false;
        let mut released = None;
        for pointer in input
            .pointers
            .iter()
            .filter(|pointer| pointer.button == Some(button))
        {
            match pointer.phase {
                PointerPhase::Began => {
                    steered = true;
                    if let Some(pick) = pick(pointer.ndc) {
                        let _ = self.grab_pick(domains, views, commands, pick, pointer.time);
                    }
                }
                PointerPhase::Moved if self.drag.is_some() => {
                    steered = true;
                    if self
                        .drag(domains, views, commands, pointer.ndc, pointer.time)
                        .is_err()
                    {
                        self.cancel(commands);
                    }
                }
                PointerPhase::Ended if self.drag.is_some() => {
                    released = self.release_at(domains, commands, pointer.time);
                }
                PointerPhase::Cancelled => {
                    self.cancel(commands);
                }
                _ => {}
            }
        }
        if let Some(ndc) = self.pointer.filter(|_| !steered && self.drag.is_some()) {
            if self
                .drag(domains, views, commands, ndc, input.time)
                .is_err()
            {
                self.cancel(commands);
            }
        }
        released
    }

    pub(crate) fn clear(&mut self) {
        self.drag = None;
        self.pointer = None;
    }
}
