use std::fmt::Display;
#[derive(Debug)]
struct HumanSize<T>(T);
impl<T: Into<u64> + Copy> Display for HumanSize<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n: u64 = self.0.into();

        if n < 1024 {
            write!(f, "{} B", n)
        } else if n < 1024 * 1024 {
            write!(f, "{} KiB", n / 1024)
        } else if n < 1024 * 1024 * 1024 {
            write!(f, "{} MiB", n / 1024 / 1024)
        } else {
            write!(f, "{} GiB", n / 1024 / 1024 / 1024)
        }
    }
}

#[derive(Debug)]
pub(crate) struct LimitsCmp {
    max_texture_dimension_1d: [u32; 2],
    max_texture_dimension_2d: [u32; 2],
    max_texture_dimension_3d: [u32; 2],
    max_texture_array_layers: [u32; 2],
    max_bind_groups: [u32; 2],
    max_bindings_per_bind_group: [u32; 2],
    max_dynamic_uniform_buffers_per_pipeline_layout: [u32; 2],
    max_dynamic_storage_buffers_per_pipeline_layout: [u32; 2],
    max_sampled_textures_per_shader_stage: [u32; 2],
    max_samplers_per_shader_stage: [u32; 2],
    max_storage_buffers_per_shader_stage: [u32; 2],
    max_storage_textures_per_shader_stage: [u32; 2],
    max_uniform_buffers_per_shader_stage: [u32; 2],
    max_uniform_buffer_binding_size: [HumanSize<u32>; 2],
    max_storage_buffer_binding_size: [HumanSize<u32>; 2],
    max_vertex_buffers: [u32; 2],
    max_buffer_size: [HumanSize<u64>; 2],
    max_vertex_attributes: [u32; 2],
    max_vertex_buffer_array_stride: [u32; 2],
    min_uniform_buffer_offset_alignment: [u32; 2],
    min_storage_buffer_offset_alignment: [u32; 2],
    max_inter_stage_shader_components: [u32; 2],
    max_compute_workgroup_storage_size: [HumanSize<u32>; 2],
    max_compute_invocations_per_workgroup: [u32; 2],
    max_compute_workgroup_size_x: [u32; 2],
    max_compute_workgroup_size_y: [u32; 2],
    max_compute_workgroup_size_z: [u32; 2],
    max_compute_workgroups_per_dimension: [u32; 2],
    max_push_constant_size: [u32; 2],
    max_non_sampler_bindings: [u32; 2],
}
impl Display for LimitsCmp {
    #[rustfmt::skip]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LimitsCmp {{")?;
        writeln!(f, "max_texture_dimension_1d: {}, {}", self.max_texture_dimension_1d[0], self.max_texture_dimension_1d[1])?;
        writeln!(f, "max_texture_dimension_2d: {}, {}", self.max_texture_dimension_2d[0], self.max_texture_dimension_2d[1])?;
        writeln!(f, "max_texture_dimension_3d: {}, {}", self.max_texture_dimension_3d[0], self.max_texture_dimension_3d[1])?;
        writeln!(f, "max_texture_array_layers: {}, {}", self.max_texture_array_layers[0], self.max_texture_array_layers[1])?;
        writeln!(f, "max_bind_groups: {}, {}", self.max_bind_groups[0], self.max_bind_groups[1])?;
        writeln!(f, "max_bindings_per_bind_group: {}, {}", self.max_bindings_per_bind_group[0], self.max_bindings_per_bind_group[1])?;
        writeln!(f, "max_dynamic_uniform_buffers_per_pipeline_layout: {}, {}", self.max_dynamic_uniform_buffers_per_pipeline_layout[0], self.max_dynamic_uniform_buffers_per_pipeline_layout[1])?;
        writeln!(f, "max_dynamic_storage_buffers_per_pipeline_layout: {}, {}", self.max_dynamic_storage_buffers_per_pipeline_layout[0], self.max_dynamic_storage_buffers_per_pipeline_layout[1])?;
        writeln!(f, "max_sampled_textures_per_shader_stage: {}, {}", self.max_sampled_textures_per_shader_stage[0], self.max_sampled_textures_per_shader_stage[1])?;
        writeln!(f, "max_samplers_per_shader_stage: {}, {}", self.max_samplers_per_shader_stage[0], self.max_samplers_per_shader_stage[1])?;
        writeln!(f, "max_storage_buffers_per_shader_stage: {}, {}", self.max_storage_buffers_per_shader_stage[0], self.max_storage_buffers_per_shader_stage[1])?;
        writeln!(f, "max_storage_textures_per_shader_stage: {}, {}", self.max_storage_textures_per_shader_stage[0], self.max_storage_textures_per_shader_stage[1])?;
        writeln!(f, "max_uniform_buffers_per_shader_stage: {}, {}", self.max_uniform_buffers_per_shader_stage[0], self.max_uniform_buffers_per_shader_stage[1])?;
        writeln!(f, "max_uniform_buffer_binding_size: {}, {}", self.max_uniform_buffer_binding_size[0], self.max_uniform_buffer_binding_size[1])?;
        writeln!(f, "max_storage_buffer_binding_size: {}, {}", self.max_storage_buffer_binding_size[0], self.max_storage_buffer_binding_size[1])?;
        writeln!(f, "max_vertex_buffers: {}, {}", self.max_vertex_buffers[0], self.max_vertex_buffers[1])?;
        writeln!(f, "max_buffer_size: {}, {}", self.max_buffer_size[0], self.max_buffer_size[1])?;
        writeln!(f, "max_vertex_attributes: {}, {}", self.max_vertex_attributes[0], self.max_vertex_attributes[1])?;
        writeln!(f, "max_vertex_buffer_array_stride: {}, {}", self.max_vertex_buffer_array_stride[0], self.max_vertex_buffer_array_stride[1])?;
        writeln!(f, "min_uniform_buffer_offset_alignment: {}, {}", self.min_uniform_buffer_offset_alignment[0], self.min_uniform_buffer_offset_alignment[1])?;
        writeln!(f, "min_storage_buffer_offset_alignment: {}, {}", self.min_storage_buffer_offset_alignment[0], self.min_storage_buffer_offset_alignment[1])?;
        writeln!(f, "max_inter_stage_shader_components: {}, {}", self.max_inter_stage_shader_components[0], self.max_inter_stage_shader_components[1])?;
        writeln!(f, "max_compute_workgroup_storage_size: {}, {}", self.max_compute_workgroup_storage_size[0], self.max_compute_workgroup_storage_size[1])?;
        writeln!(f, "max_compute_invocations_per_workgroup: {}, {}", self.max_compute_invocations_per_workgroup[0], self.max_compute_invocations_per_workgroup[1])?;
        writeln!(f, "max_compute_workgroup_size_x: {}, {}", self.max_compute_workgroup_size_x[0], self.max_compute_workgroup_size_x[1])?;
        writeln!(f, "max_compute_workgroup_size_y: {}, {}", self.max_compute_workgroup_size_y[0], self.max_compute_workgroup_size_y[1])?;
        writeln!(f, "max_compute_workgroup_size_z: {}, {}", self.max_compute_workgroup_size_z[0], self.max_compute_workgroup_size_z[1])?;
        writeln!(f, "max_compute_workgroups_per_dimension: {}, {}", self.max_compute_workgroups_per_dimension[0], self.max_compute_workgroups_per_dimension[1])?;
        writeln!(f, "max_push_constant_size: {}, {}", self.max_push_constant_size[0], self.max_push_constant_size[1])?;
        writeln!(f, "max_non_sampler_bindings: {}, {}", self.max_non_sampler_bindings[0], self.max_non_sampler_bindings[1])?;
        writeln!(f, "}}")?;

        Ok(())
    }
}
impl LimitsCmp {
    #[rustfmt::skip]
    pub(crate) fn new(a: &wgpu::Limits, b: &wgpu::Limits) -> Self {
        Self {
            max_texture_dimension_1d: [a.max_texture_dimension_1d, b.max_texture_dimension_1d],
            max_texture_dimension_2d: [a.max_texture_dimension_2d, b.max_texture_dimension_2d],
            max_texture_dimension_3d: [a.max_texture_dimension_3d, b.max_texture_dimension_3d],
            max_texture_array_layers: [a.max_texture_array_layers, b.max_texture_array_layers],
            max_bind_groups: [a.max_bind_groups, b.max_bind_groups],
            max_bindings_per_bind_group: [a.max_bindings_per_bind_group, b.max_bindings_per_bind_group],
            max_dynamic_uniform_buffers_per_pipeline_layout: [a.max_dynamic_uniform_buffers_per_pipeline_layout, b.max_dynamic_uniform_buffers_per_pipeline_layout],
            max_dynamic_storage_buffers_per_pipeline_layout: [a.max_dynamic_storage_buffers_per_pipeline_layout, b.max_dynamic_storage_buffers_per_pipeline_layout],
            max_sampled_textures_per_shader_stage: [a.max_sampled_textures_per_shader_stage, b.max_sampled_textures_per_shader_stage],
            max_samplers_per_shader_stage: [a.max_samplers_per_shader_stage, b.max_samplers_per_shader_stage],
            max_storage_buffers_per_shader_stage: [a.max_storage_buffers_per_shader_stage, b.max_storage_buffers_per_shader_stage],
            max_storage_textures_per_shader_stage: [a.max_storage_textures_per_shader_stage, b.max_storage_textures_per_shader_stage],
            max_uniform_buffers_per_shader_stage: [a.max_uniform_buffers_per_shader_stage, b.max_uniform_buffers_per_shader_stage],
            max_uniform_buffer_binding_size: [a.max_uniform_buffer_binding_size, b.max_uniform_buffer_binding_size].map(HumanSize),
            max_storage_buffer_binding_size: [a.max_storage_buffer_binding_size, b.max_storage_buffer_binding_size].map(HumanSize),
            max_vertex_buffers: [a.max_vertex_buffers, b.max_vertex_buffers],
            max_buffer_size: [a.max_buffer_size, b.max_buffer_size].map(HumanSize),
            max_vertex_attributes: [a.max_vertex_attributes, b.max_vertex_attributes],
            max_vertex_buffer_array_stride: [a.max_vertex_buffer_array_stride, b.max_vertex_buffer_array_stride],
            min_uniform_buffer_offset_alignment: [a.min_uniform_buffer_offset_alignment, b.min_uniform_buffer_offset_alignment],
            min_storage_buffer_offset_alignment: [a.min_storage_buffer_offset_alignment, b.min_storage_buffer_offset_alignment],
            max_inter_stage_shader_components: [a.max_inter_stage_shader_components, b.max_inter_stage_shader_components],
            max_compute_workgroup_storage_size: [a.max_compute_workgroup_storage_size, b.max_compute_workgroup_storage_size].map(HumanSize),
            max_compute_invocations_per_workgroup: [a.max_compute_invocations_per_workgroup, b.max_compute_invocations_per_workgroup],
            max_compute_workgroup_size_x: [a.max_compute_workgroup_size_x, b.max_compute_workgroup_size_x],
            max_compute_workgroup_size_y: [a.max_compute_workgroup_size_y, b.max_compute_workgroup_size_y],
            max_compute_workgroup_size_z: [a.max_compute_workgroup_size_z, b.max_compute_workgroup_size_z],
            max_compute_workgroups_per_dimension: [a.max_compute_workgroups_per_dimension, b.max_compute_workgroups_per_dimension],
            max_push_constant_size: [a.max_push_constant_size, b.max_push_constant_size],
            max_non_sampler_bindings: [a.max_non_sampler_bindings, b.max_non_sampler_bindings],
        }
    }
}

