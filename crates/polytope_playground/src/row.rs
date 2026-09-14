use loam::math::EuclideanR4;
use loam::runtime::{Dispatch, DomainHandle, EdgeShading, Instance, Pose, Rejection, SpawnBundle};

use crate::catalog::ShapeEntry;
use crate::mode::Mode;
use crate::{display, rest_of, Card, Playground, Slot};

pub(crate) fn add_shape(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
    entry: ShapeEntry,
    card: Card,
) -> Result<(), Rejection> {
    if *dispatch.app.mode.get() != Mode::Rotate {
        return Err(Rejection::Unsupported("shapes can only be added in Rotate"));
    }
    let index = dispatch.app.slots.len();
    let rest = rest_of(index, index + 1);
    let fallback = display::rotor_at(dispatch.app, *dispatch.app.time.get());
    let r4 = dispatch.domains.typed(domain)?;
    let frame = dispatch
        .app
        .slots
        .iter()
        .find_map(|(entity, _)| r4.poses().get(entity).map(|pose| pose.frame))
        .unwrap_or(fallback);
    let mut bundle = SpawnBundle::new()
        .at(domain, Pose { point: rest, frame })
        .row(Slot { index, entry, rest });
    if let Some(geometry) = card.geometry {
        let display = *dispatch.app.display.get();
        let shading = card.shades.map_or(EdgeShading::Material, |shades| {
            shades.of(*dispatch.app.color.get())
        });
        bundle = bundle.instance(
            Instance::new(geometry, card.material)
                .sectioned(card.cut)
                .shaded(shading)
                .line_style(display.wireframe_width_px, display.wireframe_opacity),
        );
    }
    dispatch.spawn(bundle)?;
    reflow(dispatch, domain)?;
    dispatch.app.active.set(index);
    Ok(())
}

pub(crate) fn reorder_shape(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
    from: usize,
    to: usize,
) -> Result<(), Rejection> {
    if *dispatch.app.mode.get() != Mode::Rotate {
        return Err(Rejection::Unsupported(
            "shapes can only be reordered in Rotate",
        ));
    }
    let count = dispatch.app.slots.len();
    if to > count {
        return Err(Rejection::Unsupported("no such row gap"));
    }
    if !dispatch
        .app
        .slots
        .iter()
        .any(|(_, slot)| slot.index == from)
    {
        return Err(Rejection::Unsupported("no such slot"));
    }
    let destination = if to > from { to - 1 } else { to };
    if destination == from {
        return Ok(());
    }
    let active = *dispatch.app.active.get();
    let active_entity = dispatch
        .app
        .slots
        .iter()
        .find(|(_, slot)| slot.index == active)
        .map(|(entity, _)| entity);
    for (_, slot) in dispatch.app.slots.iter_mut() {
        slot.index = reordered_index(slot.index, from, destination);
    }
    reflow(dispatch, domain)?;
    if let Some(slot) = active_entity.and_then(|entity| dispatch.app.slots.get(entity)) {
        dispatch.app.active.set(slot.index);
    }
    Ok(())
}

pub(crate) fn reordered_index(index: usize, from: usize, destination: usize) -> usize {
    if index == from {
        destination
    } else if destination <= index && index < from {
        index + 1
    } else if from < index && index <= destination {
        index - 1
    } else {
        index
    }
}

pub(crate) fn remove_shape(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
    slot: usize,
) -> Result<(), Rejection> {
    if *dispatch.app.mode.get() != Mode::Rotate {
        return Err(Rejection::Unsupported(
            "shapes can only be removed in Rotate",
        ));
    }
    let entity = dispatch
        .app
        .slots
        .iter()
        .find(|(_, held)| held.index == slot)
        .map(|(entity, _)| entity)
        .ok_or(Rejection::Unsupported("no such slot"))?;
    dispatch.despawn(entity)?;
    for (_, held) in dispatch.app.slots.iter_mut() {
        if held.index > slot {
            held.index -= 1;
        }
    }
    reflow(dispatch, domain)?;
    let active = *dispatch.app.active.get();
    dispatch.app.active.set(
        active
            .saturating_sub(usize::from(active > slot))
            .min(dispatch.app.slots.len().saturating_sub(1)),
    );
    Ok(())
}

fn reflow(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
) -> Result<(), Rejection> {
    let count = dispatch.app.slots.len();
    let r4 = dispatch.domains.typed(domain)?;
    for (entity, slot) in dispatch.app.slots.iter_mut() {
        slot.rest = rest_of(slot.index, count);
        if r4.poses().contains(entity) {
            r4.set_point(entity, slot.rest)?;
        }
    }
    Ok(())
}
