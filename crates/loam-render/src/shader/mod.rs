use std::fmt;

#[derive(Debug)]
pub enum WgslValidationError {
    Parse(naga::front::wgsl::ParseError),
    Validate(Box<naga::WithSpan<naga::valid::ValidationError>>),
    EntryPoint {
        stage: naga::ShaderStage,
        name: &'static str,
    },
    Binding {
        group: u32,
        binding: u32,
        expected: naga::AddressSpace,
        found: Option<naga::AddressSpace>,
    },
}

impl fmt::Display for WgslValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => write!(f, "WGSL parse error: {error}"),
            Self::Validate(error) => write!(f, "WGSL validation error: {error}"),
            Self::EntryPoint { stage, name } => {
                write!(f, "WGSL is missing {stage:?} entry point `{name}`")
            }
            Self::Binding {
                group,
                binding,
                expected,
                found,
            } => write!(
                f,
                "WGSL binding {group}:{binding} must use {expected:?}; found {found:?}"
            ),
        }
    }
}

impl std::error::Error for WgslValidationError {}

impl From<naga::front::wgsl::ParseError> for WgslValidationError {
    fn from(error: naga::front::wgsl::ParseError) -> Self {
        Self::Parse(error)
    }
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
