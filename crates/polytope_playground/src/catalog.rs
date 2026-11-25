use loam::prelude::*;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum Category {
    RegularPolychoron,
    SmoothSolid,
}

impl Category {
    const ALL: [Category; 2] = [Category::RegularPolychoron, Category::SmoothSolid];

    fn name(self) -> &'static str {
        match self {
            Category::RegularPolychoron => "Regular polychora",
            Category::SmoothSolid => "Smooth solids",
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub(crate) struct ShapeEntry {
    pub(crate) shape: FieldShape,
    pub(crate) body_color: [f32; 3],
    pub(crate) label: &'static str,
    pub(crate) long_name: &'static str,
    pub(crate) category: Category,
}

impl ShapeEntry {
    pub(crate) fn collider_polytope(&self) -> Option<Polytope4> {
        self.shape.polytope4()
    }
}

pub(crate) const DEFAULT_ROW: &[ShapeEntry] = &[
    ShapeEntry {
        shape: FieldShape::Polytope(Polytope4::Cell24),
        body_color: [0.95, 0.45, 0.85],
        label: "24-cell",
        long_name: "icositetrachoron",
        category: Category::RegularPolychoron,
    },
    ShapeEntry {
        shape: FieldShape::Polytope(Polytope4::Pentatope),
        body_color: [0.95, 0.55, 0.30],
        label: "5-cell",
        long_name: "pentachoron",
        category: Category::RegularPolychoron,
    },
    ShapeEntry {
        shape: FieldShape::Polytope(Polytope4::Cell16),
        body_color: [0.55, 0.95, 0.40],
        label: "16-cell",
        long_name: "hexadecachoron",
        category: Category::RegularPolychoron,
    },
    ShapeEntry {
        shape: FieldShape::Polytope(Polytope4::Tesseract),
        body_color: [0.30, 0.55, 0.95],
        label: "8-cell",
        long_name: "tesseract",
        category: Category::RegularPolychoron,
    },
];

// `body_color` channels pass straight to the WGSL kernel.
pub(crate) const SHAPE_CATALOG: &[ShapeEntry] = &[
    ShapeEntry {
        shape: FieldShape::Polytope(Polytope4::Pentatope),
        body_color: [0.95, 0.55, 0.30],
        label: "5-cell",
        long_name: "pentachoron",
        category: Category::RegularPolychoron,
    },
    ShapeEntry {
        shape: FieldShape::Polytope(Polytope4::Tesseract),
        body_color: [0.30, 0.55, 0.95],
        label: "8-cell",
        long_name: "tesseract",
        category: Category::RegularPolychoron,
    },
    ShapeEntry {
        shape: FieldShape::Polytope(Polytope4::Cell16),
        body_color: [0.55, 0.95, 0.40],
        label: "16-cell",
        long_name: "hexadecachoron",
        category: Category::RegularPolychoron,
    },
    ShapeEntry {
        shape: FieldShape::Polytope(Polytope4::Cell24),
        body_color: [0.95, 0.45, 0.85],
        label: "24-cell",
        long_name: "icositetrachoron",
        category: Category::RegularPolychoron,
    },
    ShapeEntry {
        shape: FieldShape::Polytope(Polytope4::Cell120),
        body_color: [0.40, 0.85, 0.85],
        label: "120-cell",
        long_name: "hecatonicosachoron",
        category: Category::RegularPolychoron,
    },
    ShapeEntry {
        shape: FieldShape::Polytope(Polytope4::Cell600),
        body_color: [0.95, 0.85, 0.40],
        label: "600-cell",
        long_name: "hexacosichoron",
        category: Category::RegularPolychoron,
    },
    ShapeEntry {
        shape: FieldShape::ThreeSphere,
        body_color: [0.85, 0.40, 0.40],
        label: "3-sphere",
        long_name: "hypersphere (4-ball)",
        category: Category::SmoothSolid,
    },
    ShapeEntry {
        shape: FieldShape::Duocylinder,
        body_color: [0.60, 0.45, 0.90],
        label: "duocyl",
        long_name: "duocylinder (D² × D²)",
        category: Category::SmoothSolid,
    },
    ShapeEntry {
        shape: FieldShape::CliffordTorus,
        body_color: [0.70, 0.85, 0.35],
        label: "clifford",
        long_name: "Clifford torus tube",
        category: Category::SmoothSolid,
    },
    ShapeEntry {
        shape: FieldShape::Spherinder,
        body_color: [0.85, 0.55, 0.75],
        label: "spherinder",
        long_name: "spherinder (B³ × interval)",
        category: Category::SmoothSolid,
    },
];

pub(crate) fn shape_catalog_menu(ui: &mut loam::app::egui::Ui, mut choose: impl FnMut(usize)) {
    for category in Category::ALL {
        ui.menu_button(category.name(), |ui| {
            for (card, entry) in SHAPE_CATALOG
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.category == category)
            {
                if ui
                    .button(entry.label)
                    .on_hover_text(entry.long_name)
                    .clicked()
                {
                    choose(card);
                    ui.close_kind(loam::app::egui::UiKind::Menu);
                }
            }
        });
    }
}

fn parse_shape_name(name: &str) -> Option<ShapeEntry> {
    let n = name.to_lowercase();
    let needle: &str = n.as_str();
    for entry in SHAPE_CATALOG {
        if needle == entry.label.to_lowercase() || needle == entry.long_name.to_lowercase() {
            return Some(*entry);
        }
    }
    Some(match needle {
        "5cell" | "pentatope" | "tetrahedron" => SHAPE_CATALOG[0],
        "8cell" | "hypercube" | "cube" => SHAPE_CATALOG[1],
        "16cell" | "octahedron" => SHAPE_CATALOG[2],
        "24cell" | "cuboctahedron" => SHAPE_CATALOG[3],
        "120cell" | "dodecahedron" => SHAPE_CATALOG[4],
        "600cell" | "icosahedron" => SHAPE_CATALOG[5],
        "hypersphere" | "3sphere" | "s3" | "4-ball" => SHAPE_CATALOG[6],
        "duocylinder" => SHAPE_CATALOG[7],
        "clifford" | "clifford-torus" | "torus" => SHAPE_CATALOG[8],
        "spherinder" => SHAPE_CATALOG[9],
        _ => return None,
    })
}

const SHAPE_NAMES: &str = "a shape name: 5-cell, 8-cell, 16-cell, 24-cell, 120-cell, 600-cell, \
     3-sphere, duocyl, clifford, spherinder (plus Platonic aliases: tetrahedron, cube, \
     octahedron, cuboctahedron, dodecahedron, icosahedron)";

pub(crate) fn parse_row(args: &Args) -> Result<Vec<ShapeEntry>, HostError> {
    if args.has_bare_flag("shapes") {
        return Err(HostError::Detached {
            flag: "shapes",
            example: "--shapes=24-cell,8-cell",
        });
    }
    let Some(raw) = args.get("shapes") else {
        return Ok(DEFAULT_ROW.to_vec());
    };
    let names = args.get_many("shapes");
    if names.is_empty() {
        return Err(HostError::Flag {
            flag: "shapes",
            value: raw.to_string(),
            expected: "a comma-separated list of shape names",
        });
    }
    names
        .into_iter()
        .map(|name| {
            parse_shape_name(name).ok_or_else(|| HostError::Flag {
                flag: "shapes",
                value: name.to_string(),
                expected: SHAPE_NAMES,
            })
        })
        .collect()
}
