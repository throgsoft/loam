use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BlendState, ColorTargetState,
    ColorWrites, CompareFunction, DepthStencilState, FragmentState, MultisampleState,
    PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, RenderPipeline,
    RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, StencilState, TextureFormat,
    VertexBufferLayout, VertexState,
};

use crate::device::{FeatureRequest, GpuContext, MissingGpuCapability};
use crate::{DepthConvention, DepthMode};

pub struct MaterialSpec<'a> {
    pub label: &'a str,
    pub shader: &'a str,
    pub vertex_entry: &'a str,
    pub fragment_entry: &'a str,
    pub vertex_layouts: &'a [VertexBufferLayout<'a>],
    pub resources: &'a [&'a [BindGroupLayoutEntry]],
    pub features: FeatureRequest,
    pub topology: PrimitiveTopology,
    pub blend: Option<BlendState>,
    pub depth: DepthMode,
    pub convention: DepthConvention,
    pub standard_z_compare: CompareFunction,
}

pub struct MaterialPipeline {
    pub pipeline: RenderPipeline,
    pub bind_group_layouts: Vec<BindGroupLayout>,
    convention: DepthConvention,
}

impl MaterialPipeline {
    pub fn convention(&self) -> DepthConvention {
        self.convention
    }

    pub fn group_layout(&self, group: usize) -> Option<&BindGroupLayout> {
        self.bind_group_layouts.get(group)
    }
}

impl MaterialSpec<'_> {
    pub fn build(
        &self,
        gpu: &GpuContext,
        target: TextureFormat,
        sample_count: u32,
    ) -> Result<MaterialPipeline, MissingGpuCapability> {
        let device = &gpu.device;
        self.features.resolve(device.features(), &device.limits())?;

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(self.label),
            source: ShaderSource::Wgsl(self.shader.into()),
        });
        let bind_group_layouts: Vec<BindGroupLayout> = self
            .resources
            .iter()
            .map(|entries| {
                device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some(self.label),
                    entries,
                })
            })
            .collect();
        let borrowed: Vec<&BindGroupLayout> = bind_group_layouts.iter().collect();
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(self.label),
            bind_group_layouts: &borrowed,
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(self.label),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some(self.vertex_entry),
                buffers: self.vertex_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some(self.fragment_entry),
                targets: &[Some(ColorTargetState {
                    format: target,
                    blend: self.blend,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: self.topology,
                ..Default::default()
            },
            depth_stencil: self.depth.format().map(|format| DepthStencilState {
                format,
                depth_write_enabled: self.depth.writes(),
                depth_compare: self.convention.compare(self.standard_z_compare),
                stencil: StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });
        Ok(MaterialPipeline {
            pipeline,
            bind_group_layouts,
            convention: self.convention,
        })
    }
}

#[cfg(test)]
mod tests {
    use wgpu::{BindingType, BufferBindingType, Features, Limits, ShaderStages};

    use super::*;
    use crate::device::noop_context;

    const UNIFORM_AT_ZERO: [BindGroupLayoutEntry; 1] = [BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::VERTEX_FRAGMENT,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];

    const PROBE_WGSL: &str = r#"
@vertex
fn vs_probe(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(vid), 0.0, 0.0, 1.0);
}

@fragment
fn fs_probe() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#;

    fn spec(features: FeatureRequest) -> MaterialSpec<'static> {
        MaterialSpec {
            label: "material probe",
            shader: PROBE_WGSL,
            vertex_entry: "vs_probe",
            fragment_entry: "fs_probe",
            vertex_layouts: &[],
            resources: &[&UNIFORM_AT_ZERO],
            features,
            topology: PrimitiveTopology::TriangleList,
            blend: None,
            depth: DepthMode::Off,
            convention: DepthConvention::StandardZ,
            standard_z_compare: CompareFunction::Less,
        }
    }

    #[test]
    fn a_material_the_device_cannot_serve_is_refused_by_the_feature_it_names() {
        let gpu = noop_context();
        let wanted = Features::SHADER_F16 - gpu.device.features();
        assert!(!wanted.is_empty(), "pick a feature this device lacks");
        let refused = spec(FeatureRequest {
            required_features: wanted,
            optional_features: Features::empty(),
            required_limits: Limits::downlevel_webgl2_defaults(),
        })
        .build(&gpu, TextureFormat::Rgba8Unorm, 1);
        assert_eq!(refused.err(), Some(MissingGpuCapability::Feature(wanted)));
    }
}
