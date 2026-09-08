use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use loam_math::WgslSpace;
use wgpu::{Device, ShaderModule, ShaderModuleDescriptor, ShaderSource};

#[derive(Debug, thiserror::Error)]
pub enum WgslValidationError {
    #[error("WGSL parse error: {0}")]
    Parse(#[from] naga::front::wgsl::ParseError),
    #[error("WGSL validation error: {0}")]
    Validate(Box<naga::WithSpan<naga::valid::ValidationError>>),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderId(usize);

/// One path under two owners is two modules; hot-reload is scoped by owner.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderOwner(u32);

struct Entry {
    path: PathBuf,
    module: ShaderModule,
    scene_source: Option<String>,
    prelude: Cow<'static, str>,
    generation: u64,
}

/// A failed reload or a removed file keeps the last good module.
pub struct ShaderDb {
    device: Device,
    entries: Vec<Entry>,
    path_index: HashMap<ShaderOwner, HashMap<PathBuf, ShaderId>>,
    next_owner: u32,
}

impl ShaderDb {
    pub const ROOT_OWNER: ShaderOwner = ShaderOwner(0);

    pub fn new(device: Device) -> Self {
        Self {
            device,
            entries: Vec::new(),
            path_index: HashMap::new(),
            next_owner: Self::ROOT_OWNER.0 + 1,
        }
    }

    pub fn new_owner(&mut self) -> ShaderOwner {
        let owner = ShaderOwner(self.next_owner);
        self.next_owner += 1;
        owner
    }

    /// The id is stable across reloads for one owner and path.
    pub fn load<S: WgslSpace>(
        &mut self,
        owner: ShaderOwner,
        path: impl AsRef<Path>,
        space: &S,
    ) -> Result<ShaderId> {
        self.load_inner(owner, path, None, space)
    }

    /// The scene source is stored and reused on reloads of the shader file.
    pub fn load_with_scene<S: WgslSpace>(
        &mut self,
        owner: ShaderOwner,
        path: impl AsRef<Path>,
        scene_source: &str,
        space: &S,
    ) -> Result<ShaderId> {
        self.load_inner(owner, path, Some(scene_source), space)
    }

    /// Prelude, scene SDF, [`super::GEODESIC_MARCH_KERNEL`], user shading, in that order.
    pub fn load_geodesic_scene<S: WgslSpace>(
        &mut self,
        owner: ShaderOwner,
        path: impl AsRef<Path>,
        scene_source: &str,
        space: &S,
    ) -> Result<ShaderId> {
        let scene_with_kernel = format!(
            "{scene_source}// ---- loam geodesic march kernel ----\n{}",
            super::GEODESIC_MARCH_KERNEL
        );
        self.load_inner(owner, path, Some(&scene_with_kernel), space)
    }

    fn load_inner<S: WgslSpace>(
        &mut self,
        owner: ShaderOwner,
        path: impl AsRef<Path>,
        scene_source: Option<&str>,
        space: &S,
    ) -> Result<ShaderId> {
        let path = canonicalize(path.as_ref())?;
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("reading shader {}", path.display()))?;
        let prelude = space.wgsl_impl();
        let module = self.compile(&path, &source, scene_source, &prelude)?;

        if let Some(id) = self.lookup(owner, &path) {
            let entry = &mut self.entries[id.0];
            entry.module = module;
            entry.scene_source = scene_source.map(str::to_owned);
            entry.prelude = prelude;
            entry.generation += 1;
            Ok(id)
        } else {
            let id = ShaderId(self.entries.len());
            self.path_index
                .entry(owner)
                .or_default()
                .insert(path.clone(), id);
            self.entries.push(Entry {
                path,
                module,
                scene_source: scene_source.map(str::to_owned),
                prelude,
                generation: 1,
            });
            Ok(id)
        }
    }

    fn lookup(&self, owner: ShaderOwner, path: &Path) -> Option<ShaderId> {
        self.path_index.get(&owner)?.get(path).copied()
    }

    /// The id must come from this database.
    pub fn module(&self, id: ShaderId) -> &ShaderModule {
        &self.entries[id.0].module
    }

    /// Bumps on every successful compile.
    pub fn generation(&self, id: ShaderId) -> u64 {
        self.entries.get(id.0).map(|e| e.generation).unwrap_or(0)
    }

    /// Pass canonical paths for removed files; unknown owner/path returns false and failures preserve the module.
    pub fn reload_path(&mut self, owner: ShaderOwner, path: impl AsRef<Path>) -> Result<bool> {
        let path = path.as_ref();
        let canonical = canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let Some(id) = self.lookup(owner, &canonical) else {
            return Ok(false);
        };
        self.reload(id)?;
        Ok(true)
    }

    fn reload(&mut self, id: ShaderId) -> Result<()> {
        let entry = &self.entries[id.0];
        let source = std::fs::read_to_string(&entry.path)
            .with_context(|| format!("reading shader {}", entry.path.display()))?;
        let module = self.compile(
            &entry.path,
            &source,
            entry.scene_source.as_deref(),
            &entry.prelude,
        )?;
        let entry = &mut self.entries[id.0];
        entry.module = module;
        entry.generation += 1;
        Ok(())
    }

    fn compile(
        &self,
        path: &Path,
        user_source: &str,
        scene_source: Option<&str>,
        prelude: &str,
    ) -> Result<ShaderModule> {
        let full = assemble_source_with_scene(prelude, scene_source, user_source);
        validate_wgsl(&full).with_context(|| format!("validating shader {}", path.display()))?;
        let label = path.file_name().and_then(|n| n.to_str());
        Ok(self.device.create_shader_module(ShaderModuleDescriptor {
            label,
            source: ShaderSource::Wgsl(full.into()),
        }))
    }
}

fn canonicalize(path: &Path) -> Result<PathBuf> {
    path.canonicalize()
        .with_context(|| format!("canonicalizing {}", path.display()))
}

fn assemble_source_with_scene(
    space_wgsl: &str,
    scene_wgsl: Option<&str>,
    user_source: &str,
) -> String {
    let scene_len = scene_wgsl.map(str::len).unwrap_or(0);
    let mut out = String::with_capacity(space_wgsl.len() + scene_len + user_source.len() + 96);
    out.push_str("// ---- loam-math Space prelude ----\n");
    out.push_str(space_wgsl);
    if !space_wgsl.ends_with('\n') {
        out.push('\n');
    }
    if let Some(scene_wgsl) = scene_wgsl {
        out.push_str("// ---- loam-scene scene module ----\n");
        out.push_str(scene_wgsl);
        if !scene_wgsl.ends_with('\n') {
            out.push('\n');
        }
    }
    out.push_str("// ---- user shader ----\n");
    out.push_str(user_source);
    out
}

/// Headless; `wgpu` still validates at module creation.
pub fn validate_wgsl(source: &str) -> std::result::Result<(), WgslValidationError> {
    let module = naga::front::wgsl::parse_str(source)?;
    let flags = naga::valid::ValidationFlags::all();
    let caps = naga::valid::Capabilities::empty();
    naga::valid::Validator::new(flags, caps)
        .validate(&module)
        .map_err(|e| WgslValidationError::Validate(Box::new(e)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::{EuclideanR3, EuclideanR4, HyperbolicH3, SphericalS3};

    const ABI_PROBE: &str = r#"
@compute @workgroup_size(1)
fn main() {
    let a = vec3<f32>(0.1, 0.2, 0.3);
    let b = vec3<f32>(0.2, -0.1, 0.05);
    let v = vec3<f32>(0.01, 0.02, -0.03);
    _ = loam_distance(a, b);
    _ = loam_origin_distance(a);
    _ = loam_exp(a, v);
    _ = loam_log(a, b);
    _ = loam_parallel_transport(a, b, v);
    _ = LOAM_MAX_ARC;
}
"#;

    const ABI_PROBE_VEC4: &str = r#"
@compute @workgroup_size(1)
fn main() {
    let a = vec4<f32>(0.1, 0.2, 0.3, 0.0);
    let b = vec4<f32>(0.2, -0.1, 0.05, 0.4);
    let v = vec4<f32>(0.01, 0.02, -0.03, 0.05);
    _ = loam_distance(a, b);
    _ = loam_origin_distance(a);
    _ = loam_exp(a, v);
    _ = loam_log(a, b);
    _ = loam_parallel_transport(a, b, v);
    _ = LOAM_MAX_ARC;
}
"#;

    fn noop_device() -> Device {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::NOOP,
            backend_options: wgpu::BackendOptions {
                noop: wgpu::NoopBackendOptions { enable: true },
                ..Default::default()
            },
            ..Default::default()
        });
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("noop backend always yields an adapter");
        let (device, _queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("shader-owner-scope-tests"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            }))
            .expect("noop adapter always yields a device");
        device
    }

    fn touch_with_edit(path: &Path) {
        let previous = std::fs::read_to_string(path).unwrap();
        let edited = previous.replace("0.1, 0.2, 0.3", "0.15, 0.25, 0.35");
        assert_ne!(previous, edited, "the edit must actually change the file");
        std::fs::write(path, edited).unwrap();
    }

    #[test]
    fn reload_updates_only_the_requested_owner() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.wgsl");
        std::fs::write(&path, ABI_PROBE).unwrap();

        let mut db = ShaderDb::new(noop_device());
        let spherical = db.new_owner();
        let hyperbolic = db.new_owner();
        let in_s3 = db.load(spherical, &path, &SphericalS3).unwrap();
        let in_h3 = db.load(hyperbolic, &path, &HyperbolicH3).unwrap();

        touch_with_edit(&path);

        assert!(db.reload_path(spherical, &path).unwrap());
        assert_eq!(db.generation(in_s3), 2, "S3's owner asked for this reload");
        assert_eq!(db.generation(in_h3), 1, "S3 reload must leave H3 unchanged",);

        assert!(db.reload_path(hyperbolic, &path).unwrap());
        assert_eq!(
            db.generation(in_s3),
            2,
            "H3's owner must not touch S3's module",
        );
        assert_eq!(db.generation(in_h3), 2);
    }

    #[test]
    fn unrelated_owners_ignore_changed_paths() {
        let dir = tempfile::tempdir().unwrap();
        let s3_path = dir.path().join("spherical.wgsl");
        let h3_path = dir.path().join("hyperbolic.wgsl");
        std::fs::write(&s3_path, ABI_PROBE).unwrap();
        std::fs::write(&h3_path, ABI_PROBE).unwrap();

        let mut db = ShaderDb::new(noop_device());
        let spherical = db.new_owner();
        let hyperbolic = db.new_owner();
        let in_s3 = db.load(spherical, &s3_path, &SphericalS3).unwrap();
        let in_h3 = db.load(hyperbolic, &h3_path, &HyperbolicH3).unwrap();

        touch_with_edit(&s3_path);
        assert!(db.reload_path(spherical, &s3_path).unwrap());
        assert!(!db.reload_path(hyperbolic, &s3_path).unwrap());

        assert_eq!(
            db.generation(in_s3),
            2,
            "the edited shader must recompile once, not once per owner",
        );
        assert_eq!(
            db.generation(in_h3),
            1,
            "H3 has no entry for the changed path"
        );
    }

    #[test]
    fn a_second_load_by_the_same_owner_keeps_the_id_and_recompiles() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.wgsl");
        std::fs::write(&path, ABI_PROBE).unwrap();

        let mut db = ShaderDb::new(noop_device());
        let scene = db.new_owner();
        let first = db.load(scene, &path, &SphericalS3).unwrap();
        let second = db.load(scene, &path, &SphericalS3).unwrap();

        assert_eq!(first, second, "one owner's path must map to one entry");
        assert_eq!(db.generation(first), 2, "the second load must recompile");
    }

    #[test]
    fn a_reload_rebuilds_each_entry_against_the_prelude_it_was_loaded_with() {
        let dir = tempfile::tempdir().unwrap();
        let flat3 = dir.path().join("flat3.wgsl");
        let flat4 = dir.path().join("flat4.wgsl");
        std::fs::write(&flat3, ABI_PROBE).unwrap();
        std::fs::write(&flat4, ABI_PROBE_VEC4).unwrap();

        let mut db = ShaderDb::new(noop_device());
        let owner = ShaderDb::ROOT_OWNER;
        let in_r3 = db.load(owner, &flat3, &EuclideanR3).unwrap();
        let in_r4 = db.load(owner, &flat4, &EuclideanR4).unwrap();

        touch_with_edit(&flat3);
        touch_with_edit(&flat4);
        assert!(db.reload_path(owner, &flat3).unwrap());
        assert!(db.reload_path(owner, &flat4).unwrap());

        assert_eq!(db.generation(in_r3), 2);
        assert_eq!(
            db.generation(in_r4),
            2,
            "the ℝ⁴ entry must reassemble under the vec4 prelude it was loaded with",
        );
    }

    #[test]
    fn re_loading_under_a_new_space_repoints_later_reloads_at_the_new_prelude() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("respecialized.wgsl");
        std::fs::write(&path, ABI_PROBE).unwrap();

        let mut db = ShaderDb::new(noop_device());
        let owner = ShaderDb::ROOT_OWNER;
        let id = db.load(owner, &path, &EuclideanR3).unwrap();

        std::fs::write(&path, ABI_PROBE_VEC4).unwrap();
        assert_eq!(
            db.load(owner, &path, &EuclideanR4).unwrap(),
            id,
            "one owner's path stays one entry across a Space change",
        );
        assert_eq!(db.generation(id), 2);

        touch_with_edit(&path);
        assert!(db.reload_path(owner, &path).unwrap());
        assert_eq!(
            db.generation(id),
            3,
            "the reload must use ℝ⁴'s prelude; ℝ³'s no longer validates this source",
        );
    }
    #[test]
    fn invalid_and_removed_reload_preserve_the_last_good_module() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reload.wgsl");
        std::fs::write(&path, ABI_PROBE).unwrap();
        let path = path.canonicalize().unwrap();
        let mut db = ShaderDb::new(noop_device());
        let owner = ShaderDb::ROOT_OWNER;
        let id = db.load(owner, &path, &EuclideanR3).unwrap();
        std::fs::write(&path, "invalid WGSL").unwrap();
        assert!(db.reload_path(owner, &path).is_err());
        assert_eq!(db.generation(id), 1);
        assert!(db.load(owner, &path, &EuclideanR3).is_err());
        assert_eq!(db.generation(id), 1);
        std::fs::remove_file(&path).unwrap();
        assert!(db.reload_path(owner, &path).is_err());
        assert_eq!(db.generation(id), 1);
        std::fs::write(&path, ABI_PROBE).unwrap();
        assert!(db.reload_path(owner, &path).unwrap());
        assert_eq!(db.generation(id), 2);
    }
}
