use loam::prelude::*;

// The root rig holds one of the orbit and the free camera, and the parked rig holds the other.
pub(crate) fn fly(views: &mut Views, parked: &mut Rig, free: bool) {
    let root = views.root_mut();
    if matches!(root.rig, Rig::Free(_)) == free {
        return;
    }
    std::mem::swap(&mut root.rig, parked);
    if let Rig::Free(camera) = &mut root.rig {
        camera.eye = root.eye;
    }
}

pub(crate) fn is_free(views: &Views) -> bool {
    views
        .get(views.root())
        .is_some_and(|root| matches!(root.rig, Rig::Free(_)))
}

pub(crate) fn orbit_mut<'a>(views: &'a mut Views, parked: &'a mut Rig) -> &'a mut Orbit {
    let orbit = match (&mut views.root_mut().rig, parked) {
        (Rig::Orbit(orbit), _) | (_, Rig::Orbit(orbit)) => Some(orbit),
        _ => None,
    };
    orbit.expect("the orbit is on the root rig or parked")
}

pub(crate) fn free_mut<'a>(views: &'a mut Views, parked: &'a mut Rig) -> &'a mut FreeCamera {
    let free = match (&mut views.root_mut().rig, parked) {
        (Rig::Free(free), _) | (_, Rig::Free(free)) => Some(free),
        _ => None,
    };
    free.expect("the free camera is on the root rig or parked")
}

pub(crate) fn free(views: &Views, parked: &Rig) -> FreeCamera {
    let live = views.get(views.root()).map(|root| root.rig);
    let free = match (live, *parked) {
        (Some(Rig::Free(free)), _) | (_, Rig::Free(free)) => Some(free),
        _ => None,
    };
    free.expect("the free camera is on the root rig or parked")
}
