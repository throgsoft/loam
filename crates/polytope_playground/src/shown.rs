use loam::prelude::*;

use crate::color::ColorMode;
use crate::display::Display;
use crate::mode::{Mode, Spin};
use crate::projection::Family;
use crate::strip::Strip;
use crate::toy::{self, DepthBand};
use crate::{Playground, Slot};

#[derive(Clone, Copy)]
pub(crate) struct Callout {
    pub(crate) index: usize,
    pub(crate) label: &'static str,
    pub(crate) ndc: [f32; 2],
}

#[derive(Default)]
pub struct Shown {
    pub(crate) mode: Mode,
    pub(crate) spin: Spin,
    pub(crate) active: usize,
    pub(crate) slice: f32,
    pub(crate) projection: Family,
    pub(crate) color: ColorMode,
    pub(crate) strip: Strip,
    pub(crate) display: Display,
    pub(crate) time: f32,
    pub(crate) angles: [f32; 6],
    pub(crate) controls: bool,
    pub(crate) formula: bool,
    pub(crate) labels: bool,
    pub(crate) guides: bool,
    pub(crate) rope: bool,
    pub(crate) slots: Vec<(Entity, Slot)>,
    pub(crate) callouts: Vec<Callout>,
    pub(crate) subject: Rotor4,
    pub(crate) turn: Bivector4,
    pub(crate) depths: Vec<DepthBand>,
}

impl Shown {
    pub(crate) fn write(
        &mut self,
        app: &Playground,
        domains: &Domains,
        views: &Views,
        domain: DomainHandle<EuclideanR4>,
        alpha: f32,
    ) {
        self.mode = *app.mode.get();
        self.spin = *app.spin.get();
        self.active = *app.active.get();
        self.slice = *app.slice.get();
        self.projection = *app.projection.get();
        self.color = *app.color.get();
        self.strip = *app.strip.get();
        self.display = *app.display.get();
        self.time = *app.time.get();
        self.angles = *app.angles.get();
        self.controls = *app.controls.get();
        self.formula = *app.formula.get();
        self.labels = *app.labels.get();
        self.guides = *app.guides.get();
        self.rope = *app.rope.get();
        self.slots.clear();
        self.slots
            .extend(app.slots.iter().map(|(entity, slot)| (entity, *slot)));
        self.turn = match self.mode {
            Mode::Rotate => self.spin.omega(),
            Mode::Toybox => Bivector4::ZERO,
        };
        self.follow(app, domains, views, domain, alpha);
        toy::fill_depth_bands(app, domains, domain, alpha, &mut self.depths);
    }

    fn follow(
        &mut self,
        app: &Playground,
        domains: &Domains,
        views: &Views,
        domain: DomainHandle<EuclideanR4>,
        alpha: f32,
    ) {
        let r4 = domains.read(domain).ok();
        self.callouts.clear();
        self.subject = Rotor4::IDENTITY;
        for (entity, slot) in app.slots.iter() {
            if self.display.single && slot.index != self.active {
                continue;
            }
            let pose = r4.and_then(|r4| r4.drawn_pose(entity, alpha));
            if let Some(pose) = pose.filter(|_| slot.index == self.active) {
                self.subject = pose.frame;
            }
            let center = pose.map_or(glam::Vec3::ZERO, |pose| pose.point.truncate());
            if let Some(ndc) = views.ndc(center.to_array()) {
                self.callouts.push(Callout {
                    index: slot.index,
                    label: slot.entry.label,
                    ndc,
                });
            }
        }
    }
}
