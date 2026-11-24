use loam_math::WPlane;
use loam_shape::polytope::{polytope_section_faces_append, polytope_section_perimeter_append};
use loam_shape::FieldShape;

use super::views::ViewEntry;
use super::{blend, DomainError, DomainSpace, EdgeShading, Instance, Pose, TypedDomain};
use crate::assets::{Library, PreparedGeometry};
use crate::entity::Entity;
use crate::publication::Stamp;
use crate::store::Change;
use crate::view::{
    self, EntityOutput, FieldRecord, InstanceRecord, PendingBuild, RefusalSource, SegmentRecord,
    TriangleRecord, Vec4, ViewId, ViewMapping, ViewRecords, ViewRefusals,
};

fn lerp_color(back: [f32; 4], front: [f32; 4], t: f32) -> [f32; 4] {
    let mut mixed = [0.0; 4];
    for (channel, value) in mixed.iter_mut().enumerate() {
        *value = back[channel] + (front[channel] - back[channel]) * t;
    }
    mixed
}

struct SegmentProjection<'a, S: DomainSpace> {
    space: &'a S,
    mapping: &'a dyn ViewMapping<S>,
    eye: &'a Pose<S>,
    pose: &'a Pose<S>,
    relative: &'a S::Relative,
}

fn push_segments<S: DomainSpace>(
    projection: SegmentProjection<'_, S>,
    library: &Library<'_>,
    instance: &Instance,
    entity: Entity,
    segments: &mut Vec<SegmentRecord>,
    refusals: &mut ViewRefusals,
) {
    let SegmentProjection {
        space,
        mapping,
        eye,
        pose,
        relative,
    } = projection;
    let Some(geometry) = library.geometry.get(instance.geometry.index()) else {
        return;
    };
    let (color, material_width_px) = library.line_style(instance.material);
    let width_px = instance.line_width_px.unwrap_or(material_width_px);
    let opacity = instance.line_opacity;
    let palette = match instance.shading {
        EdgeShading::Palette(id) => library.palette(id),
        _ => &[],
    };
    let placement = space.prepare(pose);
    let origin = match space.place(&placement, space.origin()) {
        Ok(origin) => origin,
        Err(error) => {
            refusals.record(entity, error, RefusalSource::Segment);
            return;
        }
    };
    let origin_depth = space.chart_point(origin).coordinates[3];
    let depth_color = |point: S::Point| match instance.shading {
        EdgeShading::Depth {
            back,
            front,
            extent,
        } => {
            let depth = space.chart_point(point).coordinates[3] - origin_depth;
            Some(lerp_color(
                back,
                front,
                (depth / extent.max(1e-6) * 0.5 + 0.5).clamp(0.0, 1.0),
            ))
        }
        _ => None,
    };
    let mut push = |index: usize, a: [f32; 4], b: [f32; 4]| {
        let start = space.local_point(a);
        let end = space.local_point(b);
        let start_point = match space.place(&placement, start) {
            Ok(point) => point,
            Err(error) => {
                refusals.record(entity, error, RefusalSource::Segment);
                return;
            }
        };
        let end_point = match space.place(&placement, end) {
            Ok(point) => point,
            Err(error) => {
                refusals.record(entity, error, RefusalSource::Segment);
                return;
            }
        };
        let painted = |at: usize| palette.get(at).copied().unwrap_or(color);
        let with_opacity = |mut color: [f32; 4]| {
            if let Some(opacity) = opacity {
                color[3] = opacity;
            }
            color
        };
        let start_color =
            with_opacity(depth_color(start_point).unwrap_or_else(|| painted(index * 2)));
        let end_color =
            with_opacity(depth_color(end_point).unwrap_or_else(|| painted(index * 2 + 1)));
        let mut previous = None;
        mapping.image_segment(space, eye, pose, relative, [start, end], &mut |t, point| {
            let Some(point) = point else {
                previous = None;
                return;
            };
            if let Some((previous_t, previous_point)) = previous {
                segments.push(SegmentRecord {
                    start: previous_point,
                    _pad0: 0.0,
                    end: point,
                    _pad1: 0.0,
                    start_color: lerp_color(start_color, end_color, previous_t),
                    end_color: lerp_color(start_color, end_color, t),
                    width_px,
                    _pad2: [0.0; 3],
                });
            }
            previous = Some((t, point));
        });
    };
    match geometry {
        PreparedGeometry::Lines4 { segments } => {
            for (index, &[a, b]) in segments.iter().enumerate() {
                push(index, a, b);
            }
        }
        PreparedGeometry::Lines3 { segments } => {
            for (index, &[a, b]) in segments.iter().enumerate() {
                push(index, [a[0], a[1], a[2], 0.0], [b[0], b[1], b[2], 0.0]);
            }
        }
        PreparedGeometry::Polytope4 { polytope, scale } => {
            let topology = polytope.topology();
            for (index, &[a, b]) in topology.edges.iter().enumerate() {
                push(
                    index,
                    (topology.vertices[a as usize] * *scale).to_array(),
                    (topology.vertices[b as usize] * *scale).to_array(),
                );
            }
        }
        PreparedGeometry::Field { .. } => {}
    }
}

struct ViewProjection<'a, S: DomainSpace> {
    space: &'a S,
    spec: &'a ViewEntry<S>,
    eye: &'a Pose<S>,
}

impl<'a, S: DomainSpace> ViewProjection<'a, S> {
    fn push_section(
        &self,
        pose: &Pose<S>,
        library: &Library<'_>,
        instance: &Instance,
        scratch: &mut view::SectionScratchpad,
        segments: &mut Vec<SegmentRecord>,
        triangles: &mut Vec<TriangleRecord>,
    ) -> Result<(), DomainError> {
        let space = self.space;
        let spec = self.spec;
        let eye = self.eye;
        let mapping = spec.style.mapping.as_ref();
        let Some(PreparedGeometry::Polytope4 { polytope, scale }) =
            library.geometry.get(instance.geometry.index())
        else {
            return Ok(());
        };
        let (Some(perimeter_material), Some(cut)) = (instance.section, mapping.section(eye, pose))
        else {
            return Ok(());
        };
        let relative = space.relative(eye, pose)?;
        let Some(place) = mapping.image_local(space, eye, pose, &relative, space.origin()) else {
            return Ok(());
        };
        let origin = space.place_relative(&relative, space.origin())?;
        let base = Vec4::from(space.chart_point(origin).coordinates);
        let topology = polytope.topology();
        scratch.rotated.clear();
        for vertex in topology.vertices.iter() {
            let local = space.local_point((*vertex * *scale).to_array());
            let placed = match space.place_relative(&relative, local) {
                Ok(placed) => placed,
                Err(error) => {
                    scratch.rotated.clear();
                    return Err(error);
                }
            };
            scratch
                .rotated
                .push(Vec4::from(space.chart_point(placed).coordinates) - base);
        }
        let plane = WPlane::new(cut.offset);
        let placed = |point: [f32; 3]| {
            [
                point[0] * cut.scale + place[0],
                point[1] * cut.scale + place[1],
                point[2] * cut.scale + place[2],
            ]
        };

        if spec.style.section_faces {
            let (fill_color, _) = library.line_style(instance.material);
            scratch.faces.vertices.clear();
            scratch.faces.colors.clear();
            scratch.faces.indices.clear();
            polytope_section_faces_append(
                topology.edges,
                topology.cells,
                &scratch.rotated,
                plane,
                fill_color,
                &mut scratch.cut,
                &mut scratch.faces,
            );
            if spec.style.section_edges {
                let (mut edge_color, material_width_px) = library.line_style(perimeter_material);
                let width_px = instance.line_width_px.unwrap_or(material_width_px);
                if let Some(opacity) = instance.line_opacity {
                    edge_color[3] = opacity;
                }
                for [_, start, end] in &scratch.faces.indices {
                    segments.push(SegmentRecord {
                        start: placed(scratch.faces.vertices[*start as usize]),
                        _pad0: 0.0,
                        end: placed(scratch.faces.vertices[*end as usize]),
                        _pad1: 0.0,
                        start_color: edge_color,
                        end_color: edge_color,
                        width_px,
                        _pad2: [0.0; 3],
                    });
                }
            }
            for [a, b, c] in &scratch.faces.indices {
                triangles.push(TriangleRecord {
                    vertices: [
                        placed(scratch.faces.vertices[*a as usize]),
                        placed(scratch.faces.vertices[*b as usize]),
                        placed(scratch.faces.vertices[*c as usize]),
                    ],
                    color: fill_color,
                });
            }
        } else if spec.style.section_edges {
            let (mut edge_color, material_width_px) = library.line_style(perimeter_material);
            let width_px = instance.line_width_px.unwrap_or(material_width_px);
            if let Some(opacity) = instance.line_opacity {
                edge_color[3] = opacity;
            }
            scratch.perimeter.segments.clear();
            scratch.perimeter.colors.clear();
            scratch.perimeter.widths.clear();
            polytope_section_perimeter_append(
                topology.edges,
                topology.cells,
                &scratch.rotated,
                plane,
                &mut scratch.cut,
                &mut scratch.perimeter,
            );
            for (start, end) in &scratch.perimeter.segments {
                segments.push(SegmentRecord {
                    start: placed(*start),
                    _pad0: 0.0,
                    end: placed(*end),
                    _pad1: 0.0,
                    start_color: edge_color,
                    end_color: edge_color,
                    width_px,
                    _pad2: [0.0; 3],
                });
            }
        }
        Ok(())
    }

    fn project_instance(
        &self,
        pose: &Pose<S>,
        library: &Library<'_>,
        instance: &Instance,
        entity: Entity,
        segments: &mut Vec<SegmentRecord>,
        refusals: &mut ViewRefusals,
    ) -> Option<InstanceRecord> {
        let space = self.space;
        let spec = self.spec;
        let eye = self.eye;
        let relative = match space.relative(eye, pose) {
            Ok(relative) => relative,
            Err(error) => {
                refusals.record(entity, error, RefusalSource::Instance);
                return None;
            }
        };
        let image_point =
            spec.style
                .mapping
                .image_local(space, eye, pose, &relative, space.origin())?;
        if spec.style.edges {
            push_segments(
                SegmentProjection {
                    space,
                    mapping: spec.style.mapping.as_ref(),
                    eye,
                    pose,
                    relative: &relative,
                },
                library,
                instance,
                entity,
                segments,
                refusals,
            );
        }
        Some(InstanceRecord {
            entity,
            geometry: instance.geometry,
            material: instance.material,
            pose: space.chart_pose(pose),
            image_point,
        })
    }

    fn push_field(
        &self,
        pose: &Pose<S>,
        library: &Library<'_>,
        instance: &Instance,
        entity: Entity,
        fields: &mut Vec<FieldRecord>,
    ) {
        let style = &self.spec.style;
        if !style.fields || style.mapping.field_cut().is_none() {
            return;
        }
        let (shape, size) = match library.geometry.get(instance.geometry.index()) {
            Some(PreparedGeometry::Polytope4 { polytope, scale }) => {
                if style.section_faces && instance.section.is_some() {
                    return;
                }
                (FieldShape::Polytope(*polytope), *scale)
            }
            Some(PreparedGeometry::Field { shape, size }) => (*shape, *size),
            _ => return,
        };
        let Some(pose) = self
            .space
            .relative(self.eye, pose)
            .ok()
            .and_then(|relative| self.space.field_pose(&relative))
        else {
            return;
        };
        let (color, _) = library.line_style(instance.material);
        fields.push(FieldRecord {
            entity,
            shape,
            size,
            material: instance.material,
            color,
            position: pose.point.to_array(),
            frame: pose.frame,
        });
    }
}

impl<S: DomainSpace> TypedDomain<S> {
    pub(super) fn prepare_view(
        &self,
        view: ViewId,
        into: &mut ViewRecords,
        stamp: Stamp,
        alpha: f32,
    ) -> Result<usize, DomainError> {
        let spec = self.views.get(view).ok_or(DomainError::UnknownView(view))?;
        let revision = self
            .views
            .revision(view)
            .ok_or(DomainError::UnknownView(view))?;
        let mut pose_cursor = into.poses;
        let pose_changes = self.poses.changes(&mut pose_cursor);
        let pose_resync = pose_changes.is_resync();
        into.changed.clear();
        let mut pose_removed = false;
        for change in pose_changes {
            match change {
                Change::Row(entity, _) => into.changed.push(entity),
                Change::Removed(_) => pose_removed = true,
            }
        }
        if blend::interpolates(&self.poses) {
            let logged = !into.changed.is_empty();
            let generation = self.blend.generation();
            let ticked = into.generation != generation;
            if ticked {
                into.changed.extend_from_slice(&into.merged);
                into.merged.clear();
                into.merged.extend_from_slice(self.blend.moving());
                into.generation = generation;
            }
            if ticked || alpha != into.alpha {
                into.changed.extend_from_slice(self.blend.moving());
                if logged || ticked {
                    into.changed.sort_unstable();
                    into.changed.dedup();
                }
            }
        }
        into.alpha = alpha;
        let mut attachment_cursor = into.attachments;
        let mut attachment_changes = self.instances.changes(&mut attachment_cursor);
        let attachment_resync = attachment_changes.is_resync();
        let attached = attachment_changes.next().is_some();
        if !pose_resync
            && !pose_removed
            && !attachment_resync
            && !attached
            && into.revision == revision
            && !into.changed.contains(&spec.eye)
        {
            if into.changed.is_empty() {
                into.instances.restamp(stamp);
                into.poses = pose_cursor;
                into.attachments = attachment_cursor;
                into.pending = PendingBuild::Current;
                return Ok(0);
            }
            if spec.style.enabled && into.refusals.count == 0 {
                into.pending = PendingBuild::Patch {
                    poses: pose_cursor,
                    attachments: attachment_cursor,
                };
                return Ok(into.changed.len());
            }
        }
        into.pending = PendingBuild::Rebuild;
        Ok(self.instances.len())
    }

    pub(super) fn build_view(
        &self,
        view: ViewId,
        library: Library<'_>,
        into: &mut ViewRecords,
        stamp: Stamp,
        alpha: f32,
    ) {
        let spec = self
            .views
            .get(view)
            .expect("a prepared view is still registered");
        match into.pending {
            PendingBuild::Current => return,
            PendingBuild::Patch { poses, attachments } => {
                if self.patch_view(spec, library, into, stamp, alpha) {
                    into.poses = poses;
                    into.attachments = attachments;
                    return;
                }
            }
            PendingBuild::Rebuild => {}
        }
        self.poses.catch_up(&mut into.poses);
        self.instances.catch_up(&mut into.attachments);
        into.revision = self
            .views
            .revision(view)
            .expect("a prepared view is still registered");
        self.rebuild_view(spec, library, into, stamp, alpha);
    }

    fn rebuild_view(
        &self,
        spec: &ViewEntry<S>,
        library: Library<'_>,
        into: &mut ViewRecords,
        stamp: Stamp,
        alpha: f32,
    ) {
        let ViewRecords {
            instances,
            segments,
            triangles,
            fields,
            field_cut,
            refusals,
            scratch,
            output,
            built,
            ..
        } = into;
        segments.clear();
        triangles.clear();
        fields.clear();
        *field_cut = None;
        *refusals = ViewRefusals::default();
        output.clear();
        let eye = match self.drawn(spec.eye, alpha) {
            Some(eye) if spec.style.enabled => eye,
            _ => {
                instances.replace(std::iter::empty(), stamp);
                *built = stamp;
                return;
            }
        };
        if spec.style.fields {
            *field_cut = spec.style.mapping.field_cut();
        }
        let projection = ViewProjection {
            space: &self.space,
            spec,
            eye: &eye,
        };
        for (entity, instance) in self.instances.iter() {
            if spec.subject.is_some_and(|subject| subject != entity) {
                continue;
            }
            let section_start = segments.len();
            let triangle_start = triangles.len();
            if let Some(pose) = self.drawn(entity, alpha) {
                if spec.style.section_edges || spec.style.section_faces {
                    if let Err(error) = projection
                        .push_section(&pose, &library, instance, scratch, segments, triangles)
                    {
                        refusals.record(entity, error, RefusalSource::Section);
                    }
                }
            }
            output.push(EntityOutput {
                entity,
                section_segments: section_start..segments.len(),
                line_segments: 0..0,
                triangles: triangle_start..triangles.len(),
                fields: 0..0,
                instance: false,
            });
        }
        let records = self
            .instances
            .iter()
            .filter(|(entity, _)| spec.subject.is_none_or(|subject| subject == *entity))
            .zip(output.iter_mut())
            .filter_map(|((entity, instance), cached)| {
                let line_start = segments.len();
                let field_start = fields.len();
                let mut record_refusals = ViewRefusals::default();
                let record = self.drawn(entity, alpha).and_then(|pose| {
                    let record = projection.project_instance(
                        &pose,
                        &library,
                        instance,
                        entity,
                        segments,
                        &mut record_refusals,
                    )?;
                    projection.push_field(&pose, &library, instance, entity, fields);
                    Some(record)
                });
                cached.line_segments = line_start..segments.len();
                cached.fields = field_start..fields.len();
                cached.instance = record.is_some();
                if record_refusals.count != 0 {
                    refusals.count = refusals.count.saturating_add(record_refusals.count);
                    if refusals.first.is_none() {
                        refusals.first = record_refusals.first;
                    }
                    refusals.last = record_refusals.last;
                }
                record.map(|record| (entity, record))
            });
        instances.replace(records, stamp);
        *built = stamp;
    }

    fn patch_view(
        &self,
        spec: &ViewEntry<S>,
        library: Library<'_>,
        into: &mut ViewRecords,
        stamp: Stamp,
        alpha: f32,
    ) -> bool {
        let Some(eye) = self.drawn(spec.eye, alpha) else {
            return false;
        };
        let projection = ViewProjection {
            space: &self.space,
            spec,
            eye: &eye,
        };
        let ViewRecords {
            instances,
            segments,
            triangles,
            fields,
            scratch,
            output,
            patch_segments,
            patch_triangles,
            patch_fields,
            changed,
            built,
            ..
        } = into;
        let mut patched = false;
        for &entity in changed.iter() {
            if spec.subject.is_some_and(|subject| subject != entity) {
                continue;
            }
            let Some(instance) = self.instances.get(entity) else {
                continue;
            };
            let Some(pose) = self.drawn(entity, alpha) else {
                return false;
            };
            let Some(cached) = output.get(entity).cloned() else {
                return false;
            };
            patch_segments.clear();
            patch_triangles.clear();
            let mut section_refusals = ViewRefusals::default();
            if spec.style.section_edges || spec.style.section_faces {
                if let Err(error) = projection.push_section(
                    &pose,
                    &library,
                    instance,
                    scratch,
                    patch_segments,
                    patch_triangles,
                ) {
                    section_refusals.record(entity, error, RefusalSource::Section);
                }
            }
            if patch_segments.len() != cached.section_segments.end - cached.section_segments.start
                || patch_triangles.len() != cached.triangles.end - cached.triangles.start
                || section_refusals.count != 0
            {
                return false;
            }
            segments[cached.section_segments.clone()].copy_from_slice(patch_segments);
            triangles[cached.triangles.clone()].copy_from_slice(patch_triangles);

            patch_segments.clear();
            patch_fields.clear();
            let mut record_refusals = ViewRefusals::default();
            let record = projection.project_instance(
                &pose,
                &library,
                instance,
                entity,
                patch_segments,
                &mut record_refusals,
            );
            if record.is_some() {
                projection.push_field(&pose, &library, instance, entity, patch_fields);
            }
            if patch_segments.len() != cached.line_segments.end - cached.line_segments.start
                || patch_fields.len() != cached.fields.end - cached.fields.start
                || record.is_some() != cached.instance
                || record_refusals.count != 0
            {
                return false;
            }
            segments[cached.line_segments.clone()].copy_from_slice(patch_segments);
            fields[cached.fields.clone()].copy_from_slice(patch_fields);
            if let Some(record) = record {
                if !instances.update(entity, record) {
                    return false;
                }
            }
            patched = true;
        }
        instances.restamp(stamp);
        if patched {
            *built = stamp;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use loam_math::EuclideanR4;
    use loam_shape::polytope::Polytope4;

    use crate::assets::{Material, PreparedGeometry};
    use crate::command::SpawnBundle;
    use crate::domain::{DomainBuilder, DomainError, Instance, Pose};
    use crate::publication::Publication;
    use crate::session::{Session, SimConfig};
    use crate::store::LogCapacity;
    use crate::view::{Projection4, Section4, Vec4, ViewSpec};

    crate::stores! {
        #[derive(Default)]
        pub struct Probe {
            tags: Store<u8>,
        }
    }

    const SEGMENTS: usize = 3;
    const EYE_AT: Vec4 = Vec4::new(0.0, 0.0, 0.0, 2.0);
    const OBJECT_AT: Vec4 = Vec4::new(0.0, 0.0, -4.0, 0.0);

    fn lines() -> PreparedGeometry {
        PreparedGeometry::Lines4 {
            segments: (0..SEGMENTS)
                .map(|i| {
                    let x = 0.1 + i as f32 * 0.1;
                    [[x, 0.0, 0.0, 0.0], [-x, 0.0, 0.0, 0.0]]
                })
                .collect(),
        }
    }

    #[test]
    fn despawned_view_subject_stays_selected_and_publishes_an_empty_view() {
        let mut session = Session::new(Probe::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let prepared = session.prepare(lines());
        let material = session.add_material(Material::flat([1.0; 4]));
        let root = session.views().root();
        let (subject, view) = session.dispatch(|dispatch| {
            let eye = dispatch
                .spawn(SpawnBundle::new().at(r4, Pose::at(EYE_AT)))
                .unwrap();
            let subject = dispatch
                .spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(OBJECT_AT))
                        .instance(Instance::new(prepared, material)),
                )
                .unwrap();
            dispatch
                .spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::new(1.0, 0.0, -4.0, 0.0)))
                        .instance(Instance::new(prepared, material)),
                )
                .unwrap();
            let view = dispatch
                .domains
                .typed(r4)
                .unwrap()
                .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }).subject(subject))
                .unwrap();
            (subject, view)
        });
        let mut publication = Publication::default();
        session.publish(&mut publication).unwrap();
        assert_eq!(publication.views[0].records.instances.rows().len(), 1);

        session
            .dispatch(|dispatch| dispatch.despawn(subject))
            .unwrap();
        session.publish(&mut publication).unwrap();
        assert!(publication.views[0].records.instances.rows().is_empty());
        assert_eq!(
            session
                .domains_mut()
                .typed(r4)
                .unwrap()
                .set_view_subject(view, Some(subject)),
            Err(DomainError::Stale(subject))
        );
        assert_eq!(
            session.domains().read(r4).unwrap().view_subject(view),
            Some(subject)
        );
    }

    #[test]
    fn cached_output_ranges_match_fresh_publication_across_view_lifecycle_changes() {
        let mut session = Session::new(Probe::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let prepared = session.prepare(lines());
        let section_geometry = session.prepare(PreparedGeometry::Polytope4 {
            polytope: Polytope4::Tesseract,
            scale: 0.25,
        });
        let material = session.add_material(Material::flat([1.0; 4]));
        let root = session.views().root();
        let (edited, removed, alternate_eye, view, section_view) = session.dispatch(|dispatch| {
            let eye = dispatch
                .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
                .unwrap();
            let alternate_eye = dispatch
                .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::new(2.0, 0.0, 0.0, 0.0))))
                .unwrap();
            let instance = || Instance::new(prepared, material);
            let edited = dispatch
                .spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(OBJECT_AT))
                        .instance(Instance::new(section_geometry, material).sectioned(material)),
                )
                .unwrap();
            let removed = dispatch
                .spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::new(1.0, 0.0, -4.0, 0.0)))
                        .instance(instance()),
                )
                .unwrap();
            let domain = dispatch.domains.typed(r4).unwrap();
            let view = domain
                .add_view(ViewSpec::new(root, eye, Projection4 { focal: 2.0 }))
                .unwrap();
            let mut section_spec = ViewSpec::new(root, eye, Section4 { w: 0.0 }).subject(edited);
            section_spec.edges = false;
            let section_view = domain.add_view(section_spec).unwrap();
            (edited, removed, alternate_eye, view, section_view)
        });
        let snapshot = session.snapshot().unwrap();
        let mut publication = Publication::default();
        let publish_and_compare = |session: &mut Session<Probe>, publication: &mut Publication| {
            session.publish(publication).unwrap();
            let mut fresh = Publication::default();
            session.publish(&mut fresh).unwrap();
            assert_eq!(publication.views.len(), fresh.views.len());
            for (cached, fresh) in publication.views.iter().zip(&fresh.views) {
                assert_eq!(cached.domain, fresh.domain);
                assert_eq!(cached.target, fresh.target);
                assert_eq!(cached.placement, fresh.placement);
                assert_eq!(
                    cached.records.instances.rows(),
                    fresh.records.instances.rows()
                );
                assert_eq!(cached.records.segments(), fresh.records.segments());
                assert_eq!(cached.records.triangles(), fresh.records.triangles());
                assert_eq!(cached.records.refusals(), fresh.records.refusals());
            }
        };
        publish_and_compare(&mut session, &mut publication);
        let initial_segments = publication.views[1].records.segments().len();
        let initial_triangles = publication.views[1].records.triangles().len();
        assert!(initial_segments > 0);
        assert!(initial_triangles > 0);

        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .set_pose(edited, Pose::at(Vec4::new(0.5, 0.0, -4.0, 0.0)))
            .unwrap();
        publish_and_compare(&mut session, &mut publication);
        assert_eq!(
            publication.views[1].records.segments().len(),
            initial_segments
        );
        assert_eq!(
            publication.views[1].records.triangles().len(),
            initial_triangles
        );

        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .set_pose(edited, Pose::at(Vec4::new(0.5, 0.0, -4.0, 1.0)))
            .unwrap();
        publish_and_compare(&mut session, &mut publication);
        assert!(publication.views[1].records.segments().is_empty());
        assert!(publication.views[1].records.triangles().is_empty());
        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .set_pose(edited, Pose::at(Vec4::new(0.5, 0.0, -4.0, 0.0)))
            .unwrap();
        publish_and_compare(&mut session, &mut publication);
        assert_eq!(
            publication.views[1].records.segments().len(),
            initial_segments
        );
        assert_eq!(
            publication.views[1].records.triangles().len(),
            initial_triangles
        );

        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .view_mut(section_view)
            .unwrap()
            .set_mapping(Section4 { w: 0.5 });
        publish_and_compare(&mut session, &mut publication);
        assert!(publication.views[1].records.triangles().is_empty());
        assert_ne!(
            publication.views[1].records.triangles().len(),
            initial_triangles
        );

        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .set_pose(edited, Pose::at(Vec4::new(0.5, 0.0, -4.0, 3.0)))
            .unwrap();
        publish_and_compare(&mut session, &mut publication);
        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .set_view_eye(view, alternate_eye)
            .unwrap();
        publish_and_compare(&mut session, &mut publication);
        session
            .dispatch(|dispatch| dispatch.despawn(removed))
            .unwrap();
        publish_and_compare(&mut session, &mut publication);
        assert!(publication.views[0]
            .records
            .instances
            .rows()
            .iter()
            .all(|record| record.entity != removed));
        session.restore(&snapshot).unwrap();
        publish_and_compare(&mut session, &mut publication);
        assert_eq!(
            publication.views[1].records.triangles().len(),
            initial_triangles
        );
    }
}
