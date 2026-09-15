use loam::app::session::FreeCamera;
use loam::runtime::ActionId;

pub(crate) const RIGHT: ActionId = ActionId(20);
pub(crate) const LEFT: ActionId = ActionId(21);
pub(crate) const FORWARD: ActionId = ActionId(22);
pub(crate) const BACKWARD: ActionId = ActionId(23);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum CameraMode {
    #[default]
    Orbit,
    Freecam,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Camera {
    pub(crate) mode: CameraMode,
    pub(crate) speed: f32,
    pub(crate) free: FreeCamera,
    pub(crate) cursor_policy: loam::app::CursorPolicy,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            mode: CameraMode::Orbit,
            speed: 4.5,
            free: FreeCamera::default(),
            cursor_policy: loam::app::CursorPolicy::Toggle,
        }
    }
}
