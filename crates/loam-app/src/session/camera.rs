pub use loam_runtime::camera::{FreeCamera, Orbit};
use loam_runtime::{Eye, Views};

pub fn look(views: &mut Views, eye: Eye) {
    views.root_mut().look(eye);
}
