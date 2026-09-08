use loam_app::shell::{SceneEntry, SceneRegistry};

pub(crate) struct Playground;

impl SceneRegistry for Playground {
    const SCENES: &'static [SceneEntry] = &[
        SceneEntry {
            slug: "rotate",
            label: "Rotate polytopes",
            build: |ctx, control| Ok(Box::new(crate::RotateScene::new(ctx, control)?)),
        },
        SceneEntry {
            slug: "toybox",
            label: "Toybox",
            build: |ctx, control| Ok(Box::new(crate::toybox::ToyboxScene::new(ctx, control)?)),
        },
    ];
}
