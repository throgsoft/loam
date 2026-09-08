//! Gate pointer controls and hotkeys separately with [`UiCapture`].

mod bivector_matrix;
mod capture;
pub mod console;
pub mod dnd;
mod floating;
mod integration;
pub mod media;
mod slider_edit;
mod world;

pub use bivector_matrix::{bivector_matrix, cell_text as bivector_matrix_cell_text};
pub use capture::UiCapture;
pub use console::{
    cmd, console_echo_enabled, parse_line, render_line, set_console_echo, subcommands, Command,
    Console, ConsoleUi, ConsoleWriter, HistoryLine, Key, LineKind, SubcommandSet,
};
pub use floating::{callout, floating_panel, sticky_menu, CalloutState};
pub use integration::UiIntegration;
pub use slider_edit::{slider_with_edit, SliderInteraction};
pub use world::world_to_screen;

pub use egui;
