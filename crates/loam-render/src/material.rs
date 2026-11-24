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
    /// One bind group layout per entry, in group order.
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
}

impl MaterialSpec<'_> {
    /// Resolves `features` before creating any GPU object, so a refusal leaves the device untouched.
    pub fn build(
        &self,
        gpu: &GpuContext,
        target: TextureFormat,
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
                count: 1,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });
        Ok(MaterialPipeline {
            pipeline,
        })
    }
}
