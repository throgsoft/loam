use crate::assets::Library;
use crate::domain::{DomainError, DomainId, Domains};
use crate::entity::SceneId;
use crate::image::{Backdrop, Views};
use crate::phase::Tick;
use crate::view::ViewTarget;

mod cache;
mod records;

pub(crate) use cache::{
    EntityOutput, Motion, PatchCursors, PatchScratch, Planned, RecordCache, ViewState,
};
pub(crate) use records::SectionScratchpad;
pub use records::{
    FieldRecord, InstanceRecord, Overlay, PointRecord, RecordBuffer, RefusalSource, SegmentRecord,
    TextRecord, TriangleRecord, ViewRecords, ViewRefusal, ViewRefusals,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stamp {
    pub tick: Tick,
    pub sequence: u64,
}

pub struct PublishedView {
    pub domain: DomainId,
    pub target: ViewTarget,
    pub records: ViewRecords,
}

#[derive(Default)]
pub struct Publication<V> {
    pub views: Vec<PublishedView>,
    pub overlay: Overlay,
    pub backdrop: Option<Backdrop>,
    pub stamp: Stamp,
    /// The clamped alpha the records were drawn at.
    pub alpha: f32,
    /// Written in place by the session's value system, so its buffers keep their capacity between publishes.
    pub value: V,
    source: Option<SceneId>,
    content: u64,
}

impl<V> Publication<V> {
    /// Changes only when a publish alters what a frame draws; a restamp alone leaves it as it was.
    pub fn content(&self) -> u64 {
        self.content
    }
}

impl<V: Default> Publication<V> {
    pub(crate) fn claim(&mut self, scene: SceneId) {
        if self.source != Some(scene) {
            *self = Self::default();
        }
    }

    // Builds at the stamp and alpha the caller has already written.
    pub(crate) fn reconcile(
        &mut self,
        scene: SceneId,
        domains: &Domains,
        views: &Views,
        library: Library<'_>,
        overlay: &Overlay,
    ) -> Result<(), DomainError> {
        let built = self.draw(scene, domains, views, library, overlay);
        match built {
            Ok(()) => self.source = Some(scene),
            Err(_) => *self = Self::default(),
        }
        built
    }
}

impl<V> Publication<V> {
    fn draw(
        &mut self,
        scene: SceneId,
        domains: &Domains,
        views: &Views,
        library: Library<'_>,
        overlay: &Overlay,
    ) -> Result<(), DomainError> {
        let (stamp, alpha) = (self.stamp, self.alpha);
        // A buffer that claim cleared has no source until this publish succeeds.
        let mut changed = self.source != Some(scene);
        let mut count = 0;
        for domain in domains.owned() {
            for &target in domain.views() {
                if target.image != views.root_id() {
                    continue;
                }
                let held = self
                    .views
                    .iter()
                    .position(|view| view.domain == domain.id() && view.target == target);
                match held {
                    Some(index) if index != count => {
                        self.views.swap(index, count);
                        changed = true;
                    }
                    Some(_) => {}
                    None => {
                        self.views.insert(
                            count,
                            PublishedView {
                                domain: domain.id(),
                                target,
                                records: ViewRecords::default(),
                            },
                        );
                        changed = true;
                    }
                }
                count += 1;
            }
        }
        changed |= self.views.len() != count;
        self.views.truncate(count);
        for view in &mut self.views {
            domains
                .owner(view.domain)
                .expect("a published view names a domain of this session")
                .publish(view.target.view, library, &mut view.records, stamp, alpha)?;
            changed |= view.records.built() == stamp;
        }
        changed |= self.overlay != *overlay;
        self.overlay.clone_from(overlay);
        let backdrop = views.root().backdrop;
        changed |= self.backdrop != backdrop;
        self.backdrop = backdrop;
        if changed {
            self.content = stamp.sequence;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::view::Vec4;
    use loam_math::{EuclideanR4, Space};
    use loam_time::alloc::bytes_allocated_by;
    use loam_time::par::Sequential;

    use super::*;
    use crate::assets::{Assets, Material, MaterialId, PreparedGeometry, PreparedId};
    use crate::command::{Rejection, SpawnBundle};
    use crate::domain::{DomainBuilder, Instance, Pose};
    use crate::image::ImageRay;
    use crate::input::Input;
    use crate::phase::{Ctx, Phase};
    use crate::session::{Session, SimConfig};
    use crate::store::{LogCapacity, Store};
    use crate::view::{DepthEnvelope, DomainRay, ViewMapping, ViewSpec};

    crate::stores! {
        #[derive(Default)]
        pub struct Quiet {}
    }

    fn shaded_segments(
        shading: crate::domain::EdgeShading,
        at_w: f32,
        line_style: Option<(f32, f32)>,
        sectioned: bool,
        prepare: impl FnOnce(&mut Session<Quiet>) -> (PreparedId, MaterialId),
    ) -> Vec<SegmentRecord> {
        use crate::image::Eye;
        use crate::view::{Section4, Vec4};

        let mut session = Session::new(Quiet::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let (geometry, material) = prepare(&mut session);
        let root = session.views().root_id();
        session
            .dispatch(|d| -> Result<(), Rejection> {
                let eye = d.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))?;
                let mut instance = Instance::new(geometry, material).shaded(shading);
                if let Some((width_px, opacity)) = line_style {
                    instance = instance.line_style(width_px, opacity);
                }
                if sectioned {
                    instance = instance.sectioned(material);
                }
                d.spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::new(0.0, 0.0, -4.0, at_w)))
                        .instance(instance),
                )?;
                d.domains
                    .typed(r4)?
                    .add_view(ViewSpec::new(root, eye, Section4 { w: at_w }))?;
                Ok(())
            })
            .expect("the view registered");
        session.views_mut().root_mut().eye = Eye::default();
        let mut publication = Publication::default();
        session.publish(&mut publication, 1.0).expect("published");
        publication.views[0].records.segments().to_vec()
    }

    #[test]
    fn tesseract_cut_at_w_zero() {
        use crate::image::Eye;
        use crate::view::{Section4, Vec4, ViewId};

        const SIZE: f32 = 0.7;
        const LIFT: f32 = 0.2;
        let mut session = Session::new(Quiet::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let geometry = session.prepare(PreparedGeometry::Polytope4 {
            polytope: loam_shape::polytope::Polytope4::Tesseract,
            scale: SIZE,
        });
        let body = session.add_material(Material::lines([1.0; 4], 1.0));
        let cut = session.add_material(Material::lines([1.0, 0.85, 0.35, 1.0], 2.0));
        let root = session.views().root_id();
        let view = session
            .dispatch(|d| -> Result<ViewId, Rejection> {
                let eye = d.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))?;
                d.spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::new(0.0, 0.0, -4.0, 0.0)))
                        .instance(Instance::new(geometry, body).sectioned(cut)),
                )?;
                d.domains
                    .typed(r4)?
                    .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))
                    .map_err(Rejection::from)
            })
            .expect("the view registered");
        session.views_mut().root_mut().eye = Eye::default();

        let mut publication = Publication::default();
        let read = |session: &mut Session<Quiet>, publication: &mut Publication<()>| {
            session.publish(publication, 1.0).expect("published");
            let view = &publication.views[0];
            (
                view.records.triangles().len(),
                view.records.segments().len(),
                view.records.built(),
            )
        };

        let (triangles, segments, built) = read(&mut session, &mut publication);
        assert_eq!(
            triangles, 24,
            "the six cells that straddle w = 0 each cut to a square, and each square fans around its centroid into four triangles, not {triangles}"
        );
        assert_eq!(
            segments,
            32 + 24,
            "the wireframe's 32 edges and the cut cube's 24 perimeter segments came to {segments}"
        );
        assert_eq!(
            read(&mut session, &mut publication).2,
            built,
            "an unchanged source and view rebuilt the section"
        );

        session
            .dispatch(|d| -> Result<(), Rejection> {
                d.domains
                    .typed(r4)?
                    .view_mut(view)
                    .ok_or(Rejection::Unsupported("view"))?
                    .set_mapping(Section4 { w: LIFT });
                Ok(())
            })
            .expect("the slice moved");
        let (lifted, _, moved) = read(&mut session, &mut publication);
        assert_ne!(
            moved, built,
            "a new slice left the section at its old build"
        );
        assert_eq!(
            lifted, 24,
            "the cut at w = {LIFT} still meets six cells, not {lifted}"
        );
    }

    #[test]
    fn segment_instance_style() {
        use crate::domain::EdgeShading;

        const COLORS: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ];
        const MATERIAL: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
        let two_edges = vec![
            [[0.0; 4], [0.5, 0.0, 0.0, 0.0]],
            [[0.0; 4], [0.0, 0.5, 0.0, 0.0]],
        ];

        let painted = shaded_segments(EdgeShading::Material, 0.0, None, false, |session| {
            (
                session.prepare(PreparedGeometry::Lines4 {
                    segments: two_edges.clone(),
                }),
                session.add_material(Material::lines(MATERIAL, 1.0)),
            )
        });
        assert_eq!(
            painted.iter().map(|s| s.start_color).collect::<Vec<_>>(),
            [MATERIAL, MATERIAL],
            "an instance with no shading lost the material's line color"
        );

        let first = Assets::default().add_palette(Vec::new());
        let mut id = None;
        let painted = shaded_segments(
            EdgeShading::Palette(first),
            0.0,
            None,
            false,
            |session: &mut Session<Quiet>| {
                id = Some(session.add_palette(COLORS.to_vec()));
                (
                    session.prepare(PreparedGeometry::Lines4 {
                        segments: two_edges,
                    }),
                    session.add_material(Material::lines(MATERIAL, 1.0)),
                )
            },
        );
        assert_eq!(id, Some(first));
        assert_eq!(
            painted
                .iter()
                .flat_map(|s| [s.start_color, s.end_color])
                .collect::<Vec<_>>(),
            COLORS,
            "the palette did not follow the prepared segment endpoints in order"
        );

        const EXTENT: f32 = 0.5;
        const BACK: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
        const FRONT: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        let shading = EdgeShading::Depth {
            back: BACK,
            front: FRONT,
            extent: EXTENT,
        };
        let prepare = |session: &mut Session<Quiet>| {
            (
                session.prepare(PreparedGeometry::Lines4 {
                    segments: vec![[[0.0; 4], [0.0, 0.0, 0.0, EXTENT]]],
                }),
                session.add_material(Material::lines([1.0; 4], 1.0)),
            )
        };

        for lifted in [0.0, 3.0] {
            let segments = shaded_segments(shading, lifted, None, false, prepare);
            assert_eq!(
                segments[0].start_color,
                [0.5, 0.0, 0.5, 1.0],
                "the midpoint of the depth ramp is wrong for a body at w {lifted}"
            );
            assert_eq!(
                segments[0].end_color, FRONT,
                "the far endpoint of the depth ramp is wrong for a body at w {lifted}"
            );
        }

        let [segment] = shaded_segments(
            crate::domain::EdgeShading::Material,
            0.0,
            Some((2.5, 0.4)),
            false,
            |session| {
                (
                    session.prepare(PreparedGeometry::Lines4 {
                        segments: vec![[[0.0; 4], [0.5, 0.0, 0.0, 0.0]]],
                    }),
                    session.add_material(Material::lines([0.2, 0.4, 0.6, 0.8], 1.0)),
                )
            },
        )[..] else {
            panic!("the line did not publish as one segment");
        };
        assert_eq!(segment.width_px, 2.5);
        assert_eq!(segment.start_color, [0.2, 0.4, 0.6, 0.4]);
        assert_eq!(segment.end_color, [0.2, 0.4, 0.6, 0.4]);

        let sectioned = shaded_segments(
            crate::domain::EdgeShading::Material,
            0.0,
            Some((2.5, 0.4)),
            true,
            |session| {
                (
                    session.prepare(PreparedGeometry::Polytope4 {
                        polytope: loam_shape::polytope::Polytope4::Tesseract,
                        scale: 0.7,
                    }),
                    session.add_material(Material::lines([0.2, 0.4, 0.6, 0.8], 1.0)),
                )
            },
        );
        assert!(sectioned.len() > 32, "the section edge did not publish");
        assert!(sectioned[..sectioned.len() - 32].iter().all(|segment| {
            segment.width_px == 2.5
                && segment.start_color == [0.2, 0.4, 0.6, 0.4]
                && segment.end_color == [0.2, 0.4, 0.6, 0.4]
        }));
    }

    crate::stores! {
        pub struct Shown {
            scores: Store<u32>,
        }
        publishes Vec<u32>;
    }

    struct Flat;

    impl ViewMapping<EuclideanR4> for Flat {
        fn name(&self) -> &'static str {
            "flat"
        }

        fn image_point(
            &self,
            eye: &Pose<EuclideanR4>,
            point: <EuclideanR4 as Space>::Point,
        ) -> Option<[f32; 3]> {
            let relative = point - eye.point;
            Some([relative.x, relative.y, relative.z])
        }

        fn lift(
            &self,
            _eye: &Pose<EuclideanR4>,
            _ray: &ImageRay,
        ) -> Option<DomainRay<EuclideanR4>> {
            None
        }

        fn depth_envelope(&self) -> DepthEnvelope {
            DepthEnvelope {
                near: 0.0,
                far: 1.0,
            }
        }
    }

    #[test]
    fn warm_no_allocation() {
        let shown = Shown {
            scores: Store::tracked(LogCapacity::default()),
        };
        let mut session = Session::new(shown, SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let geometry = session.prepare(PreparedGeometry::Lines4 {
            segments: Vec::new(),
        });
        let material = session.add_material(Material::flat([1.0; 4]));
        let root = session.views().root_id();
        session.dispatch(|d| {
            let eye = d
                .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
                .unwrap();
            d.domains
                .typed(r4)
                .unwrap()
                .add_view(ViewSpec::new(root, eye, Flat))
                .unwrap();
            for value in 0..8 {
                d.spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::ZERO))
                        .instance(Instance::new(geometry, material))
                        .row(value),
                )
                .unwrap();
            }
        });
        session.system(Phase::Simulation, "churn", move |ctx: Ctx<'_, Shown>| {
            let domain = ctx.domains.typed(r4).unwrap();
            for (entity, score) in ctx.app.scores.iter_mut() {
                *score += 1;
                let mut point = domain.poses().get(entity).unwrap().point;
                point.x += ctx.step.dt;
                domain.set_point(entity, point).unwrap();
            }
            Ok(())
        });
        session.publish_value("scores", |ctx, shown: &mut Vec<u32>| {
            shown.clear();
            shown.extend(ctx.app.scores.iter().map(|(_, score)| *score));
            Ok(())
        });
        let mut publication = Publication::default();
        let cycle = |session: &mut Session<Shown>, publication: &mut Publication<Vec<u32>>| {
            session.tick(&Sequential).unwrap();
            session.boundary(Input::default()).unwrap();
            session.publish(publication, 1.0).unwrap();
        };
        for _ in 0..4 {
            cycle(&mut session, &mut publication);
        }

        let bytes = bytes_allocated_by(|| {
            for _ in 0..16 {
                cycle(&mut session, &mut publication);
            }
        })
        .expect("the counting allocator is installed");
        assert_eq!(
            bytes, 0,
            "16 warmed publishes asked the allocator for {bytes} bytes"
        );
        assert_eq!(publication.views[0].records.instances.rows().len(), 8);
        assert_eq!(publication.value.len(), 8);
    }
}
