//! Callers pass WGSL modules in dependency order.

#[derive(Debug, thiserror::Error)]
pub enum WgslValidationError {
    #[error("WGSL parse error: {0}")]
    Parse(#[from] naga::front::wgsl::ParseError),
    #[error("WGSL validation error: {0}")]
    Validate(Box<naga::WithSpan<naga::valid::ValidationError>>),
    #[error("WGSL is missing {stage:?} entry point `{name}`")]
    EntryPoint {
        stage: naga::ShaderStage,
        name: &'static str,
    },
    #[error("WGSL binding {group}:{binding} must use {expected:?}; found {found:?}")]
    Binding {
        group: u32,
        binding: u32,
        expected: naga::AddressSpace,
        found: Option<naga::AddressSpace>,
    },
}

const RAYMARCH_BINDINGS: &[(u32, u32, naga::AddressSpace)] = &[(0, 0, naga::AddressSpace::Uniform)];
const HYPERSLICE_BINDINGS: &[(u32, u32, naga::AddressSpace)] = &[
    (0, 0, naga::AddressSpace::Uniform),
    (
        0,
        1,
        naga::AddressSpace::Storage {
            access: naga::StorageAccess::LOAD,
        },
    ),
];

/// Assemble after the space and scene modules.
pub const GEODESIC_MARCH_KERNEL: &str = concat!(
    include_str!("projective_depth.wgsl"),
    include_str!("kernel.wgsl")
);

pub fn assemble_wgsl(modules: &[&str]) -> String {
    let capacity = modules
        .iter()
        .map(|module| module.len() + usize::from(!module.ends_with('\n')))
        .sum();
    let mut source = String::with_capacity(capacity);
    for module in modules {
        source.push_str(module);
        if !module.ends_with('\n') {
            source.push('\n');
        }
    }
    source
}

pub fn validate_wgsl(source: &str) -> Result<(), WgslValidationError> {
    parse_and_validate(source).map(|_| ())
}

fn parse_and_validate(source: &str) -> Result<naga::Module, WgslValidationError> {
    let module = naga::front::wgsl::parse_str(source)?;
    let flags = naga::valid::ValidationFlags::all();
    let caps = naga::valid::Capabilities::empty();
    naga::valid::Validator::new(flags, caps)
        .validate(&module)
        .map_err(|error| WgslValidationError::Validate(Box::new(error)))?;
    Ok(module)
}

pub(crate) fn validate_raymarch_wgsl(source: &str) -> Result<(), WgslValidationError> {
    validate_render_wgsl(source, &["fs_main"], RAYMARCH_BINDINGS)
}

pub(crate) fn validate_hyperslice_wgsl(source: &str) -> Result<(), WgslValidationError> {
    validate_render_wgsl(source, &["fs_main", "fs_depth"], HYPERSLICE_BINDINGS)
}

fn validate_render_wgsl(
    source: &str,
    fragment_entries: &[&'static str],
    required_bindings: &[(u32, u32, naga::AddressSpace)],
) -> Result<(), WgslValidationError> {
    let module = parse_and_validate(source)?;
    for &(group, binding, expected) in required_bindings {
        let found = module.global_variables.iter().find_map(|(_, variable)| {
            (variable.binding == Some(naga::ResourceBinding { group, binding }))
                .then_some(variable.space)
        });
        if found != Some(expected) {
            return Err(WgslValidationError::Binding {
                group,
                binding,
                expected,
                found,
            });
        }
    }
    for (stage, name) in std::iter::once((naga::ShaderStage::Vertex, "vs_fullscreen")).chain(
        fragment_entries
            .iter()
            .copied()
            .map(|name| (naga::ShaderStage::Fragment, name)),
    ) {
        if !module
            .entry_points
            .iter()
            .any(|entry| entry.stage == stage && entry.name == name)
        {
            return Err(WgslValidationError::EntryPoint { stage, name });
        }
    }
    Ok(())
}
