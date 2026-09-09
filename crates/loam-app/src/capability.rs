use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capability {
    Native,
    WebGpu,
    WebGpuPhone,
    WebGl2,
    FileWatch,
    Capture,
    PointerLock,
    Touch,
    Threads,
}

impl Capability {
    pub fn name(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::WebGpu => "webgpu",
            Self::WebGpuPhone => "webgpu-phone",
            Self::WebGl2 => "webgl2",
            Self::FileWatch => "file-watch",
            Self::Capture => "capture",
            Self::PointerLock => "pointer-lock",
            Self::Touch => "touch",
            Self::Threads => "threads",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostProfile(u32);

impl HostProfile {
    pub fn host() -> Self {
        #[cfg(target_arch = "wasm32")]
        let profile = Self::default()
            .with(Capability::WebGpu)
            .with(Capability::PointerLock)
            .with(Capability::Touch);
        #[cfg(not(target_arch = "wasm32"))]
        let profile = {
            let profile = Self::default()
                .with(Capability::Native)
                .with(Capability::FileWatch)
                .with(Capability::PointerLock)
                .with(Capability::Touch)
                .with(Capability::Threads);
            #[cfg(feature = "capture")]
            let profile = profile.with(Capability::Capture);
            profile
        };
        profile
    }

    pub fn with(self, capability: Capability) -> Self {
        Self(self.0 | (1 << capability as u32))
    }

    pub fn provides(self, capability: Capability) -> bool {
        self.0 & (1 << capability as u32) != 0
    }

    /// Fails on the first entry of `required` this host does not provide.
    pub fn require(self, required: &[Capability]) -> Result<(), MissingCapability> {
        required
            .iter()
            .copied()
            .find(|capability| !self.provides(*capability))
            .map_or(Ok(()), |capability| Err(MissingCapability(capability)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MissingCapability(pub Capability);

impl fmt::Display for MissingCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "this host does not provide the required capability `{}`",
            self.0.name()
        )
    }
}

impl std::error::Error for MissingCapability {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_names_the_first_absent_required_capability() {
        let host = HostProfile::default()
            .with(Capability::Touch)
            .with(Capability::PointerLock);
        let required = [
            Capability::Touch,
            Capability::FileWatch,
            Capability::Capture,
        ];
        let error = host.require(&required).unwrap_err();
        assert_eq!(error, MissingCapability(Capability::FileWatch));
        assert!(error.to_string().contains("file-watch"));
        assert_eq!(host.require(&required[..1]), Ok(()));
    }
}
