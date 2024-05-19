use cgmath::dot;
#[allow(unused)]
use cgmath::{
    AbsDiff, Basis2, Basis3, Decomposed, Deg, Euler, Matrix2, Matrix3, Matrix4, Ortho, Perspective,
    PerspectiveFov, Point1, Point2, Point3, Quaternion, Rad, Relative, Ulps, Vector1, Vector2,
    Vector3, Vector4,
};
#[allow(unused)]
use cgmath::{
    AbsDiffEq, Angle, Array, BaseFloat, BaseNum, Bounded, ElementWise, EuclideanSpace, InnerSpace,
    Matrix, MetricSpace, One, RelativeEq, Rotation, Rotation2, Rotation3, SquareMatrix, Transform,
    Transform2, Transform3, UlpsEq, VectorSpace, Zero,
};
use winit::keyboard::KeyCode;

use pollster::FutureExt;
use wgpu::util::{DeviceExt, RenderEncoder};
use winit::{event::*, event_loop::EventLoop, window::WindowBuilder};

use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    fmt::Display,
    marker::Copy,
    mem::{replace, size_of},
    rc::Rc,
    slice::from_ref,
    sync::{Arc, Mutex},
    time::{Duration, Instant, UNIX_EPOCH},
};

mod marching_cubes;
use marching_cubes::CubeMarch;

include!(concat!(env!("OUT_DIR"), "/compile_time.rs"));

macro_rules! mut_map_i {
    ((&mut $e:expr)) => {
        (wgpu::BufferBindingType::Storage { read_only: false }, &$e)
    };
    ((&$e:expr)) => {
        (wgpu::BufferBindingType::Storage { read_only: true }, &$e)
    };
    (($e:expr)) => {
        (wgpu::BufferBindingType::Uniform, &$e)
    };
}

macro_rules! mut_map {
    ($($t:tt),*) => {
        [$(mut_map_i!($t)),*]
    }
}

#[rustfmt::skip]
macro_rules! source {
    ($filename:expr) => {{
        #[cfg(debug_assertions)]
        { ($filename, &std::fs::read_to_string(concat!("src/",$filename,".wgsl")).unwrap()) }
        #[cfg(not(debug_assertions))]
        { ($filename, include_str!(concat!($filename, ".wgsl"))) }
    }};
}

// MSB                          LSB
// 00000000000000000000000000000000
//   ZZZZZZZZZZYYYYYYYYYYXXXXXXXXXX vertex % 3
// 10 bit => 1024

/*
wgpu::DrawIndirectArgs {
    vertex_count: 3,
    instance_count: memcpy last offset,
    first_vertex: 3 * pos,
    first_instance: write_offset,
}
*/
const MAX_TRIS_PER_CHUNK: u32 = 32 * 32 * 32 * 4;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawIndirectArgs2 {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}
impl From<wgpu::util::DrawIndirectArgs> for DrawIndirectArgs2 {
    fn from(
        wgpu::util::DrawIndirectArgs {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        }: wgpu::util::DrawIndirectArgs,
    ) -> Self {
        Self {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        }
    }
}

fn pos_to_target(pos: Point3<f64>) -> Point3<u32> {
    (pos).map(|f| ((f / 2.0) as i32).max(0).min(1023) as u32)
}

#[derive(Debug)]
struct DataEntry {
    offset: u32,
    count: u32,
    x: u32,
    y: u32,
    z: u32,
}
struct Storage {
    // "copying" GC
    // probably need 128 MiB.
    b0: MetaBuffer,
    //b1: MetaBuffer,
    data: Vec<DataEntry>,
    in_queue: BTreeSet<(u32, u32, u32)>,
    compute_queue: VecDeque<(u32, u32, u32)>,
    count_staging_buffer: MetaBuffer,
    send: std::sync::mpsc::Sender<Result<(), wgpu::BufferAsyncError>>,
    recv: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    awaiting: Option<(u32, u32, u32)>,
    offset: u32,
    capacity: u32,
    trigen: TriGenState,
    data_entries_written: usize,
    indirect_draw_buffer: MetaBuffer,
    culled_indirect_draw_buffer: MetaBuffer,
    cull_kernel: Kernel,
}
impl Storage {
    fn new(device: &wgpu::Device, cube_march: &CubeMarch, camera_buffer: &MetaBuffer) -> Self {
        let (send, recv) = std::sync::mpsc::channel();
        let capacity: u32 = 128 * 1024 * 1024;
        // let capacity: u32 = 511 * 1024 * 1024;
        let b0 = MetaBuffer::new_uninit(device, "b0", STORAGE | VERTEX, (capacity as u64) * 4);
        let data = Vec::new();
        let in_queue = BTreeSet::new();
        let compute_queue = VecDeque::new();
        let count_staging_buffer = MetaBuffer::new_uninit(
            device,
            "staging",
            COPY_DST | MAP_READ,
            size_of::<u32>() as u64,
        );
        let awaiting = None;
        let offset = 0;
        let trigen = TriGenState::new(device, cube_march, &b0);

        let indirect_draw_buffer_size = 32 * 1024 * 1024;
        let indirect_draw_buffer = MetaBuffer::new_uninit(
            device,
            "indirect_draw_buffer",
            INDIRECT | COPY_DST | STORAGE,
            indirect_draw_buffer_size,
        );
        let culled_indirect_draw_buffer = MetaBuffer::new_uninit(
            device,
            "culled_indirect_draw_buffer",
            INDIRECT | STORAGE,
            indirect_draw_buffer_size,
        );

        let cull_kernel = Kernel::new(
            device,
            source!("cull"),
            &mut_map!(
                (camera_buffer),
                (&indirect_draw_buffer),
                (&mut culled_indirect_draw_buffer)
            ),
            0,
        );

        let data_entries_written = 0;
        Self {
            b0,
            data,
            in_queue,
            compute_queue,
            count_staging_buffer,
            send,
            recv,
            awaiting,
            offset,
            trigen,
            capacity,
            data_entries_written,
            indirect_draw_buffer,
            culled_indirect_draw_buffer,
            cull_kernel,
        }
    }
    fn dispatch(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, pos: Point3<u32>) {
        let target: (u32, u32, u32) = (pos.x, pos.y, pos.z);
        let radius = 5;

        if self.capacity < self.offset + MAX_TRIS_PER_CHUNK {
            return;
        }

        if self.compute_queue.len() == 0 {
            for x in target.0.saturating_sub(radius)..(target.0 + radius + 1).min(1023) {
                for y in target.1.saturating_sub(radius)..(target.1 + radius + 1).min(1023) {
                    for z in target.2.saturating_sub(radius)..(target.2 + radius + 1).min(1023) {
                        if self.in_queue.insert((x, y, z)) {
                            self.compute_queue.push_back((x, y, z));
                        }
                    }
                }
            }
        }
        for _ in 0..16 {
            if self.capacity < self.offset + MAX_TRIS_PER_CHUNK {
                break;
            }
            let Some((x, y, z)) = self.compute_queue.pop_front() else {
                break;
            };

            if let Some((x, y, z)) = self.awaiting.take() {
                device.poll(wgpu::Maintain::Wait); // :(
                let _: () = self.recv.recv().unwrap().unwrap();
                let count = bytemuck::cast_slice::<u8, u32>(
                    &self
                        .count_staging_buffer
                        .buffer
                        .slice(..)
                        .get_mapped_range(),
                )[0];
                self.count_staging_buffer.buffer.unmap();

                if count > 0 {
                    let entry = DataEntry {
                        offset: self.offset,
                        count,
                        x,
                        y,
                        z,
                    };
                    self.offset += count;
                    self.data.push(entry);
                }
            }

            let chunk = ChunkUniform {
                x,
                y,
                z,
                offset: self.offset,
            };
            self.trigen.chunk.write(queue, chunk);

            let mut encoder = device.create_command_encoder(&default());
            self.trigen
                .dispatch(&mut encoder, &self.count_staging_buffer);
            queue.submit([encoder.finish()]); // submit needed to map buffer later.

            self.awaiting = Some((x, y, z));
            let sender = self.send.clone();
            self.count_staging_buffer
                .buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |x| sender.send(x).unwrap());
        }

        if self.data_entries_written < self.data.len() {
            let range = &self.data[self.data_entries_written..self.data.len()];
            let to_write: Vec<DrawIndirectArgs2> = range
                .into_iter()
                .map(
                    |&DataEntry {
                         offset,
                         count,
                         x,
                         y,
                         z,
                     }| {
                        let encoded = vertex_encode(x, y, z);
                        wgpu::util::DrawIndirectArgs {
                            vertex_count: 3,
                            instance_count: count,
                            first_vertex: encoded,
                            first_instance: offset,
                        }
                        .into()
                    },
                )
                .collect();

            queue.write_buffer(
                &self.indirect_draw_buffer.buffer,
                (self.data_entries_written * size_of::<wgpu::util::DrawIndirectArgs>()) as u64,
                bytemuck::cast_slice(&to_write),
            );
            self.data_entries_written = self.data.len();
        }
    }
    fn cull(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.cull_kernel.x = ((self.data_entries_written + 255) / 256) as u32;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cull pass"),
            timestamp_writes: None,
        });
        self.cull_kernel.dispatch(&mut pass);
    }
    fn draw<'this, 'pass>(&'this mut self, pass: &mut wgpu::RenderPass<'pass>, culled: bool)
    where
        'this: 'pass,
    {
        pass.set_vertex_buffer(0, self.b0.buffer.slice(..));
        // for &DataEntry {
        //     offset,
        //     count,
        //     x,
        //     y,
        //     z,
        // } in &self.data
        // {
        //     let encoded = vertex_encode(x, y, z);
        //     pass.draw(encoded..(encoded + 3), offset..(offset + count));
        // }
        pass.multi_draw_indirect(
            &if culled {
                &self.culled_indirect_draw_buffer
            } else {
                &self.indirect_draw_buffer
            }
            .buffer,
            0,
            self.data_entries_written as u32,
        );
    }
}

fn vertex_encode(x: u32, y: u32, z: u32) -> u32 {
    let encoded = (x | (y << 10) | (z << 20)) * 3;
    encoded
}
// Copying GC cycle:
// 1. pause voxel creation
// 2. copy all alive data to second buffer (crop sorted).
// 3. swap buffers.

fn default<T: Default>() -> T {
    Default::default()
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct Vertex {
    pos: [f32; 3],
}

macro_rules! wgsl_uniform {
    ($name:ident $str_const:ident $($(# $tt:tt)* $field_name:ident : $t:tt,)*) => {
        unsafe impl bytemuck::NoUninit for $name {}
        #[derive(Copy, Clone)]
        #[repr(C)]
        struct $name { $( $( # $tt )* $field_name : $t,)* }
        impl $name {
            fn register(v: &mut Vec<(&'static str, &'static str)>) {
                v.push((stringify!($str_const), $str_const));
            }
        }
        const $str_const: &'static str = concat!("struct ", stringify!($name), " {", $(stringify!($field_name),": ",wgsl_uniform!($t),", ",)* " }");
    };
    ([$t:tt; $e:expr]) => {
        concat!("array<", wgsl_uniform!($t), ", ", stringify!($e), ">")
    };
    (u32) => {"u32"};
}

wgsl_uniform!(TriAllocUniform TRI_ALLOC_UNIFORM
    // 3 bit (0..=4)
    case_to_size: [u32; 256],
);
impl TriAllocUniform {
    fn new(case: &CubeMarch) -> Self {
        Self {
            case_to_size: case.case_to_size.map(|size| size as u32),
        }
    }
}
wgsl_uniform!(TriWriteBackUniform TRI_WRITE_BACK_UNIFORM
    /// triangle lives on 3 edges
    /// each edge is on 2 corners
    /// edge0 edge1, edge2
    /// XYZ XYZ XYZ XYZ XYZ XYZ TRIA = 22 significant bits
    offset_to_bitpack: [u32; 820],
    /// 10 bit (0..732)
    case_to_offset: [u32; 257],
    /// Padding
    _unused0: u32,
    /// Padding
    _unused1: u32,
    /// Padding
    _unused2: u32,
);
impl TriWriteBackUniform {
    fn new(case: &CubeMarch) -> Self {
        Self {
            offset_to_bitpack: case.offset_to_triangle.map(|triangle| {
                let [a, b, c] = case.triangle_to_edge[triangle].map(|edge| {
                    let [a, b] = case.edge_to_corner[edge];
                    (a as u32) | ((b as u32) << 3)
                });
                let triangle = triangle as u32;
                // a | (b << 6) | (c << 12) | (triangle << 18)
                triangle | (a << 8) | (b << 14) | (c << 20)
            }),
            case_to_offset: case.case_to_offset.map(|offset| offset as u32),
            _unused0: 0,
            _unused1: 0,
            _unused2: 0,
        }
    }
}

wgsl_uniform!(RenderCaseUniform RENDER_CASE_UNIFORM
    /// XYZ XYZ XYZ XYZ XYZ XYZ = 18 bit
    triangle_to_corners: [u32; 190], // 135
);
impl RenderCaseUniform {
    fn new(case: &CubeMarch) -> Self {
        Self {
            triangle_to_corners: case.triangle_to_edge.map(|edges| {
                let [a, b, c] = edges.map(|edge| {
                    let [a, b] = case.edge_to_corner[edge];
                    (a as u32) | ((b as u32) << 3)
                });
                a | (b << 6) | (c << 12)
            }),
        }
    }
}

struct MetaBuffer {
    buffer: wgpu::Buffer,
    bytes: u64,
}
impl MetaBuffer {
    fn new<T: bytemuck::NoUninit>(
        device: &wgpu::Device,
        name: &str,
        flags: wgpu::BufferUsages,
        contents: &[T],
    ) -> Self {
        Self::new_i(
            device,
            name,
            wgpu::BufferUsages::STORAGE | flags,
            bytemuck::cast_slice(contents),
        )
    }
    fn new_constant<T: bytemuck::NoUninit>(
        device: &wgpu::Device,
        name: &str,
        flags: wgpu::BufferUsages,
        contents: &[T],
    ) -> Self {
        Self::new_i(
            device,
            name,
            wgpu::BufferUsages::STORAGE | flags,
            bytemuck::cast_slice(contents),
        )
    }
    fn new_uniform<T: bytemuck::NoUninit>(
        device: &wgpu::Device,
        flags: wgpu::BufferUsages,
        data: &T,
    ) -> Self {
        Self::new_i(
            device,
            &format!("{} uniform buffer", std::any::type_name::<T>()),
            // wgpu::BufferUsages::COPY_DST
            wgpu::BufferUsages::UNIFORM | flags,
            bytemuck::cast_slice(from_ref(data)),
        )
    }
    fn new_uninit(
        device: &wgpu::Device,
        name: &str,
        flags: wgpu::BufferUsages,
        bytes: u64,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: bytes,
            usage: flags,
            mapped_at_creation: false,
        });

        Self { buffer, bytes }
    }
    fn new_i(
        device: &wgpu::Device,
        name: &str,
        usage: wgpu::BufferUsages,
        contents: &[u8],
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents,
            usage,
        });
        Self {
            buffer,
            bytes: contents.len() as u64,
        }
    }
    fn binding_layout(
        &self,
        visibility: wgpu::ShaderStages,
        ty: wgpu::BufferBindingType,
        binding: u32,
    ) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(self.bytes),
            },
            count: None,
        }
    }
    fn binding(&self, binding: u32) -> wgpu::BindGroupEntry {
        wgpu::BindGroupEntry {
            binding,
            resource: self.buffer.as_entire_binding(),
        }
    }
    fn write<T: bytemuck::NoUninit>(&self, queue: &wgpu::Queue, data: T) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[data]));
    }
}

#[derive(Default, Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
struct ChunkUniform {
    x: u32,
    y: u32,
    z: u32,
    offset: u32,
}

struct Kernel {
    timestamp: TimestampQuerySet,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    x: u32,
}
impl Kernel {
    fn new(
        device: &wgpu::Device,
        (name, source): (&str, &str),
        buffers: &[(wgpu::BufferBindingType, &MetaBuffer)],
        x: u32,
    ) -> Self {
        let timestamp = TimestampQuerySet::new(device);
        let (binding_entry_layouts, binding_entries): (Vec<_>, Vec<_>) = buffers
            .iter()
            .zip(0..)
            .map(|(buffer, i)| {
                (
                    buffer.1.binding_layout(COMPUTE_STAGE, buffer.0, i),
                    buffer.1.binding(i),
                )
            })
            .unzip();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{name} bind group layout")),
            entries: &binding_entry_layouts,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{name} bind group")),
            layout: &bind_group_layout,
            entries: &binding_entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{name} pipeline layout")),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let shader_module = create_shader_module(
            device,
            wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{name} shader module")),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
            },
        );
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{name} compute pipeline")),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            compilation_options: default(),
        });
        Self {
            pipeline,
            bind_group,
            x,
            timestamp,
        }
    }
    fn dispatch<'s: 'c, 'c>(&'s mut self, pass: &mut wgpu::ComputePass<'c>) {
        let timestamp = self.timestamp.push();
        if let Some((start, _)) = timestamp {
            pass.write_timestamp(&self.timestamp.query_set, start);
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.x, 1, 1);
        if let Some((_, end)) = timestamp {
            pass.write_timestamp(&self.timestamp.query_set, end);
        }
    }
}

fn create_shader_module(
    device: &wgpu::Device,
    desc: wgpu::ShaderModuleDescriptor,
) -> wgpu::ShaderModule {
    #[cfg(debug_assertions)]
    {
        device.create_shader_module(desc)
    }
    #[cfg(not(debug_assertions))]
    unsafe {
        device.create_shader_module_unchecked(desc)
    }
}

const COMPUTE_STAGE: wgpu::ShaderStages = wgpu::ShaderStages::COMPUTE;
const VERTEX_STAGE: wgpu::ShaderStages = wgpu::ShaderStages::VERTEX;
const FRAGMENT_STAGE: wgpu::ShaderStages = wgpu::ShaderStages::FRAGMENT;

const NONE: wgpu::BufferUsages = wgpu::BufferUsages::empty();
const MAP_READ: wgpu::BufferUsages = wgpu::BufferUsages::MAP_READ;
const MAP_WRITE: wgpu::BufferUsages = wgpu::BufferUsages::MAP_WRITE;
const COPY_SRC: wgpu::BufferUsages = wgpu::BufferUsages::COPY_SRC;
const COPY_DST: wgpu::BufferUsages = wgpu::BufferUsages::COPY_DST;
const INDEX: wgpu::BufferUsages = wgpu::BufferUsages::INDEX;
const VERTEX: wgpu::BufferUsages = wgpu::BufferUsages::VERTEX;
const UNIFORM: wgpu::BufferUsages = wgpu::BufferUsages::UNIFORM;
const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE;
const INDIRECT: wgpu::BufferUsages = wgpu::BufferUsages::INDIRECT;
const QUERY_RESOLVE: wgpu::BufferUsages = wgpu::BufferUsages::QUERY_RESOLVE;

struct TriGenState {
    /// `vec3<i32>, f32`
    chunk: MetaBuffer,
    /// `[f32; 33^3 + 415]`
    sdf_data: MetaBuffer,
    /// `33^3 + 415`
    sdf: Kernel,
    /// `32^3`
    triangle_allocation: Kernel,
    /// [u32; 32^3 + 1]
    triangle_count_prefix: MetaBuffer,
    /// `256`
    prefix_top: Kernel,
    /// `32^3`
    triangle_writeback: Kernel,
    staging_buffer: MetaBuffer,
    render_case_uniform: MetaBuffer,
    inspect_buffer: MetaBuffer,
}

impl TriGenState {
    #[rustfmt::skip]
    fn new(device: &wgpu::Device, cube_march: &CubeMarch, triangle_storage: &MetaBuffer) -> Self {
        let inspect_buffer = MetaBuffer::new_uninit(device, "inspect_buffer", COPY_DST | MAP_READ, (32*32*32 + 1) * 4);
        let sdf_data = MetaBuffer::new(device, "sdf_data", NONE, &vec![0.0_f32; 33*33*33 + 159]);

        let triangle_count_prefix = MetaBuffer::new(device, "triangle_count_prefix", COPY_SRC, &vec![0_u32; 32*32*32 + 1]);

        let staging_buffer = MetaBuffer::new_uninit(device, "tri count buffer", COPY_DST | MAP_READ, size_of::<u32>() as u64);

        // let query_buffer0 = MetaBuffer::new_uninit(device, "query buffer0", COPY_SRC | QUERY_RESOLVE, (std::mem::size_of::<u64>() * 2) as u64);
        // let query_buffer1 = MetaBuffer::new_uninit(device, "query buffer1", COPY_DST | MAP_READ, (std::mem::size_of::<u64>() * 2) as u64);

        let render_case_uniform = RenderCaseUniform::new(&cube_march);
        let render_case_uniform = MetaBuffer::new(device, "render_case_uniform", NONE, from_ref(&render_case_uniform));


        let chunk_uniform = ChunkUniform::default();
        let chunk_uniform = MetaBuffer::new_uniform(device, COPY_DST | STORAGE, &chunk_uniform);

        let tri_alloc_uniform = TriAllocUniform::new(&cube_march);
        let tri_alloc_uniform = MetaBuffer::new_constant(device, "TriAllocUniform", NONE, from_ref(&tri_alloc_uniform));

        let tri_wb_uniform = TriWriteBackUniform::new(&cube_march);
        let tri_wb_uniform = MetaBuffer::new_constant(device, "TriWriteBackUniform", NONE, from_ref(&tri_wb_uniform));
        let _ = mut_map!((&chunk_uniform), (&sdf_data));

        let sdf = Kernel::new(device, source!("sdf"), &mut_map!(
                (chunk_uniform), (&mut sdf_data)
        ), (33*33*33 + 159)/256);
        let triangle_allocation = Kernel::new(device, source!("triangle_allocation"), &mut_map!(
                (&tri_alloc_uniform), (&sdf_data), (&mut triangle_count_prefix), (&tri_wb_uniform)
        ), (32*32*32)/128);
        let prefix_top = Kernel::new(device, source!("prefix_top"), &mut_map!(
                (&mut triangle_count_prefix)
        ), 1);
        let triangle_writeback = Kernel::new(device, source!("triangle_writeback"), &mut_map!(
                (&sdf_data), (&triangle_count_prefix), (&mut triangle_storage), (&tri_wb_uniform), (chunk_uniform)
        ), 32*32*32/128);

        Self {
            chunk: chunk_uniform,
            sdf_data,
            sdf,
            triangle_allocation,
            triangle_count_prefix,
            prefix_top,
            triangle_writeback,
            staging_buffer,
            render_case_uniform,
            inspect_buffer,
            // query_buffer0,
            // query_buffer1,
        }

    }
    fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder, staging_buffer: &MetaBuffer) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        self.sdf.dispatch(&mut pass);
        self.triangle_allocation.dispatch(&mut pass);
        self.prefix_top.dispatch(&mut pass);
        self.triangle_writeback.dispatch(&mut pass);
        drop(pass);
        encoder.copy_buffer_to_buffer(
            &self.triangle_count_prefix.buffer,
            (32 * 32 * 32) * 4,
            &staging_buffer.buffer,
            0,
            4,
        );
    }
}

// gen u8 from sdf, calc number of verts per cell.
// prefix sum verts per cell
// draw the sum of verts of vertices
//
//
// passes:
// gen verts/cell + start prefix sum

macro_rules! uniform_magic {
    (struct $name:ident { $($field_name:ident : $t0:tt $(< $t1:tt $(,$t2:tt)?>)? ,)* }) => {
        unsafe impl bytemuck::NoUninit for $name {}
        #[derive(Copy, Clone)]
        #[repr(C)]
        struct $name { $( $field_name : uniform_magic!($t0 $(, $t1 $(, $t2)?)?),)* }
    };
    (array, $t:tt, $l:tt) => {[uniform_magic!($t); $l]};
    (mat4x4, $t:tt) => {[[uniform_magic!($t); 4]; 4]};
    (vec3, $t:tt) => {[uniform_magic!($t); 4]};
    (vec4, $t:tt) => {[uniform_magic!($t); 4]};
    (u32) => {u32};
    (i32) => {i32};
    (f32) => {f32};
}

uniform_magic!(
    struct CameraUniform {
        world_view: mat4x4<f32>,  // AKA view
        view_world: mat4x4<f32>,  // AKA view_inv

        world_clipw: mat4x4<f32>, // AKA view_proj
        clipw_world: mat4x4<f32>, // AKA view_proj_inv
        clipw_view: mat4x4<f32>,

        world_light: mat4x4<f32>,
        light_world: mat4x4<f32>,

        time: f32,
        cull_radius: f32,
        fog_inv: f32,
        _unused2: f32,

        world_clipw_cull: mat4x4<f32>,

        fog_color: vec3<f32>,
        diffuse_color: vec3<f32>,
        specular_color: vec3<f32>,
        light_color: vec3<f32>,
        sun_color: vec3<f32>,

        base_offset: vec3<i32>,
    }
);

struct Camera {
    pos: cgmath::Point3<f64>,
    dir: cgmath::Quaternion<f64>,

    vel: cgmath::Vector3<f64>,
    rvel: cgmath::Quaternion<f64>,

    debug_above: bool,

    time: f64,

    //vel: cgmath::Vector3<f64>,
    aspect: f64,
    fovy: f64,
    znear: f64,
    zfar: f64,
}
impl Camera {
    #[rustfmt::skip]
    const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f64> = cgmath::Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.5,
        0.0, 0.0, 0.0, 1.0,
    );
    fn uniform(&self, gui: &GuiState) -> CameraUniform {
        // move world to camera pos/rot
        let view = self.view_cull();

        //Matrix4::look_at_rh((&self).pos, (&self).pos + Vector3 {x: 0.0, y: 0.0, z: 1.0 }, (&self).up);
        // projection matrix
        self.uniform_from_view(view, gui)
    }
    fn view_cull(&self) -> Matrix4<f64> {
        let b = Matrix4::from(Matrix3::from(Basis3::from_quaternion(&self.dir)));
        let view = b * Matrix4::from_translation(
            -(self.pos.to_vec() - self.base_offset().map(|v| 2.0 * v as f64).to_vec()),
        );
        view
    }
    fn base_offset(&self) -> Point3<i32> {
        self.pos.map(|x| (x * 0.5) as i32)
    }
    fn uniform_from_view(&self, view: Matrix4<f64>, gui: &GuiState) -> CameraUniform {
        let view_projection = self.proj_view_wgpu();

        let proj = Self::OPENGL_TO_WGPU_MATRIX * self.proj();

        let lmap = Matrix4::from_angle_x(Deg(34.4523423))
            * Matrix4::from_angle_y(Deg(47.3478))
            * Matrix4::from_angle_z(Deg(23.3894));
        let lmap_inv = lmap.invert().unwrap();

        let cull = Self::OPENGL_TO_WGPU_MATRIX
            * self.proj()
            * Matrix4::from_translation(Vector3 {
                x: 0.0,
                y: 0.0,
                z: -4.0,
            })
            * Matrix4::from(Matrix3::from(Basis3::from_quaternion(&self.dir)))
            * Matrix4::from_translation(
                -(self.pos.to_vec() - self.base_offset().map(|v| 2.0 * v as f64).to_vec()),
            );

        fn expand_v(a: Point3<i32>) -> [i32; 4] {
            [a.x, a.y, a.z, 0]
        }
        fn expand(a: [u8; 3]) -> [f32; 4] {
            let a = a.map(|a| (a as f32) / 255.0);
            [a[0], a[1], a[2], 0.0]
        }
        fn conv_mat(m: Matrix4<f64>) -> [[f32; 4]; 4] {
            let m: [[f64; 4]; 4] = m.into();
            m.map(|m| m.map(|m| m as f32))
        }
        CameraUniform {
            world_view: conv_mat(view),
            view_world: conv_mat(view.invert().unwrap()),
            world_clipw: dbg!(conv_mat(view_projection)),
            clipw_world: conv_mat(view_projection.invert().unwrap()),
            world_light: lmap.into(),
            light_world: lmap_inv.into(),
            time: self.time as f32,
            cull_radius: gui.cull_radius(),
            fog_inv: 1.0 / gui.fog().powi(2),
            _unused2: 0.0,
            world_clipw_cull: conv_mat(cull),
            fog_color: expand(gui.fog_color),
            diffuse_color: expand(gui.diffuse_color),
            specular_color: expand(gui.specular_color),
            light_color: expand(gui.light_color),
            sun_color: expand(gui.sun_color),
            clipw_view: conv_mat(proj.invert().unwrap()),
            base_offset: expand_v(self.base_offset()),
        }
    }
    fn proj(&self) -> Matrix4<f64> {
        let proj = cgmath::perspective(
            cgmath::Deg((&self).fovy),
            (&self).aspect,
            (&self).znear,
            (&self).zfar,
        );
        proj
    }
    fn proj_view_wgpu(&self) -> Matrix4<f64> {
        Self::OPENGL_TO_WGPU_MATRIX * self.proj() * self.view_cull()
    }
    fn new(surface_config: &wgpu::SurfaceConfiguration) -> Self {
        Self {
            pos: Point3::from((2.0, 2.0, 2.0)) * 2.0 * 1.0 * 256.0,
            aspect: surface_config.width as f64 / surface_config.height as f64,
            fovy: 45.0,
            znear: 0.01,
            zfar: 100.0,
            dir: Quaternion::look_at(
                Vector3 {
                    x: -1.0,
                    y: 0.0,
                    z: 0.0,
                },
                Vector3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
            ),
            vel: (0.0, 0.0, 0.0).into(),
            rvel: Quaternion::one(),
            time: 0.0,
            debug_above: false,
        }
    }
    fn update(
        &mut self,
        surface_config: &wgpu::SurfaceConfiguration,
        dt: f64,
        held_keys: &BTreeSet<KeyCode>,
        _pressed_keys: &BTreeSet<KeyCode>,
    ) {
        let dt = dt.max(0.0004).min(0.5);
        self.time += dt;
        self.aspect = surface_config.width as f64 / surface_config.height as f64;
        use KeyCode::{
            KeyA, KeyD, KeyI, KeyJ, KeyK, KeyL, KeyO, KeyS, KeyU, KeyW, ShiftLeft, Space,
        };

        let held = |key: KeyCode| held_keys.contains(&key) as u8 as f64;

        let delta = Vector3 {
            x: held(KeyD) - held(KeyA),
            y: held(Space) - held(ShiftLeft),
            z: held(KeyS) - held(KeyW),
        };

        let ry = held(KeyL) - held(KeyJ);
        let rx = held(KeyK) - held(KeyI);
        let rz = held(KeyO) - held(KeyU);

        let tmp_basis = Basis3::from(self.dir);
        let tmp_basis_i = Matrix3::from(tmp_basis.invert());
        let tmp_basis = Matrix3::from(tmp_basis);

        // self.pos += (tmp_basis.invert().unwrap() * delta) * dt * 10.0;
        self.vel += (tmp_basis.invert().unwrap() * delta) * dt * 4.0;
        if self.vel.magnitude() > 0.0001 {
            self.vel -= self.vel * 0.05 * dt * 60.0;
            self.vel -= self.vel.normalize() * dot(self.vel, self.vel) * 0.005 * dt * 60.0;
        } else {
            self.vel = (0.0, 0.0, 0.0).into();
        }
        self.pos += self.vel * dt;

        let r1 = Quaternion::from_axis_angle(tmp_basis_i.x, Rad(rx * dt * 0.2));
        let r2 = Quaternion::from_axis_angle(tmp_basis_i.y, Rad(ry * dt * 0.2));
        let r3 = Quaternion::from_axis_angle(tmp_basis_i.z, Rad(rz * dt * 0.2));

        self.rvel = self.rvel * &r3;
        self.rvel = self.rvel * &r2;
        self.rvel = self.rvel * &r1;
        self.rvel = self.rvel.slerp(Quaternion::one(), 0.1);

        self.dir = self.dir * self.rvel;
    }
}

struct Texture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}
impl Texture {
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    fn new_depth(device: &wgpu::Device, surface_config: &wgpu::SurfaceConfiguration) -> Self {
        let size = wgpu::Extent3d {
            width: surface_config.width,
            height: surface_config.height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }
}

struct TimestampQuerySet {
    s0: u64,
    s1: f64,
    s2: f64,
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    i: u32,
    mean: f64,
    sd: f64,
}
impl TimestampQuerySet {
    fn new(device: &wgpu::Device) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Timestamp query set"),
            ty: wgpu::QueryType::Timestamp,
            count: wgpu::QUERY_SET_MAX_QUERIES,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestap resolve buffer"),
            size: wgpu::QUERY_SET_MAX_QUERIES as u64 * size_of::<u64>() as u64,
            usage: QUERY_RESOLVE | COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestap staging buffer"),
            size: wgpu::QUERY_SET_MAX_QUERIES as u64 * size_of::<u64>() as u64,
            usage: MAP_READ | COPY_DST,
            mapped_at_creation: false,
        });
        let i = 0;
        Self {
            query_set,
            resolve_buffer,
            staging_buffer,
            i,
            s0: 0,
            s1: 0.0,
            s2: 0.0,
            mean: 0.0,
            sd: 0.0,
        }
    }
    fn push(&mut self) -> Option<(u32, u32)> {
        (self.i < wgpu::QUERY_SET_MAX_QUERIES).then(|| {
            let res = (self.i, self.i + 1);
            self.i += 2;
            res
        })
    }
    fn render_timestamp_writes<'a>(&'a mut self) -> Option<wgpu::RenderPassTimestampWrites<'a>> {
        self.push()
            .map(|(start, end)| wgpu::RenderPassTimestampWrites {
                query_set: &self.query_set,
                beginning_of_pass_write_index: Some(start),
                end_of_pass_write_index: Some(end),
            })
    }
    fn resolve_copy(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.i == 0 {
            return;
        }
        encoder.resolve_query_set(&self.query_set, 0..self.i, &self.resolve_buffer, 0);
        let bytes_to_copy = self.i as u64 * size_of::<u64>() as u64;
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.staging_buffer,
            0,
            bytes_to_copy,
        );
    }
    fn resolve_map(&mut self) {
        if self.i == 0 {
            return;
        }
        let bytes_copied = self.i as u64 * size_of::<u64>() as u64;
        self.staging_buffer
            .slice(0..bytes_copied)
            .map_async(wgpu::MapMode::Read, |_| ());
    }
    fn resolve_unmap(&mut self, seconds_per_tick: f64) {
        if self.i == 0 {
            return;
        }
        #[repr(C)]
        #[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Query {
            start: u64,
            end: u64,
        }
        let bytes_copied = self.i as u64 * size_of::<u64>() as u64;
        if self.i == 0 {
            return;
        }
        let tmp = self
            .staging_buffer
            .slice(0..bytes_copied)
            .get_mapped_range();
        let data: &[Query] = bytemuck::cast_slice(&tmp);
        for &Query { start, end } in data {
            let elapsed = (end.wrapping_sub(start)) as f64 * seconds_per_tick;
            self.s0 += 1;
            self.s1 += elapsed;
            self.s2 += elapsed * elapsed;
        }
        drop(tmp);
        self.staging_buffer.unmap();
        let Self { s0, s1, s2, .. } = *self;
        let s0 = (s0 as f64).max(4.0);
        let mean = s1 / s0;
        let sd = ((s2 * s0 - s1 * s1) / (s0 * (s0 - 1.0))).sqrt();

        self.i = 0;
        self.mean = mean;
        self.sd = sd;
    }
    fn reset(&mut self) {
        self.s0 = 0;
        self.s1 = 0.0;
        self.s2 = 0.0;
        self.mean = 0.0;
        self.sd = 0.0;
    }
}

struct UpdateRenderPipeline {
    label: String,
    path: String,
    filename: String,
    last_update: Duration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer_layout: Vec<wgpu::VertexBufferLayout<'static>>,
    surface_format: wgpu::TextureFormat,
    pipeline_layout: Rc<wgpu::PipelineLayout>,
    vs_main: &'static str,
    fs_main: Option<&'static str>,
    polygon_mode: wgpu::PolygonMode,
    timestamp: TimestampQuerySet,
    write_depth: bool,
    validation_error: Option<String>,
}
impl UpdateRenderPipeline {
    fn new(
        label: &'static str,
        device: &wgpu::Device,
        pipeline_layout: Rc<wgpu::PipelineLayout>,
        vertex_buffer_layout: Vec<wgpu::VertexBufferLayout<'static>>,
        surface_format: wgpu::TextureFormat,
        (filename, source): (&'static str, &str),
        vs_main: &'static str,
        fs_main: Option<&'static str>,
        polygon_mode: wgpu::PolygonMode,
        write_depth: bool,
    ) -> Self {
        let vs_main = vs_main;

        let label = format!("{filename} ({label})");
        let pipeline = Self::reconstruct(
            &label,
            device,
            filename,
            source,
            &pipeline_layout,
            vertex_buffer_layout.clone(),
            surface_format,
            vs_main,
            fs_main,
            polygon_mode,
            write_depth,
        );

        let timestamp = TimestampQuerySet::new(device);

        Self {
            label,
            path: format!("src/{filename}.wgsl"),
            last_update: COMPILE_TIME,
            pipeline,
            filename: filename.to_owned(),
            vertex_buffer_layout,
            surface_format,
            pipeline_layout,
            vs_main,
            fs_main,
            polygon_mode,
            timestamp,
            write_depth,
            validation_error: None,
        }
    }
    fn get<'a>(
        &'a mut self,
        device: &wgpu::Device,
    ) -> (
        Option<wgpu::RenderPassTimestampWrites<'a>>,
        &wgpu::RenderPipeline,
    ) {
        #[cfg(not(debug_assertions))]
        {
            return (self.timestamp.render_timestamp_writes(), &self.pipeline);
        }
        let time = std::fs::metadata(&self.path)
            .unwrap()
            .modified()
            .unwrap()
            .duration_since(UNIX_EPOCH)
            .unwrap();

        if time > self.last_update {
            self.last_update = time;
            let source = std::fs::read_to_string(&self.path).unwrap();
            device.push_error_scope(wgpu::ErrorFilter::Validation);
            let pipeline = Self::reconstruct(
                &self.label,
                device,
                &self.filename,
                &source,
                &self.pipeline_layout,
                self.vertex_buffer_layout.clone(),
                self.surface_format,
                self.vs_main,
                self.fs_main,
                self.polygon_mode,
                self.write_depth,
            );
            if let Some(err) = device.pop_error_scope().block_on() {
                if let wgpu::Error::Validation {
                    source,
                    description,
                } = err
                {
                    println!("VALIDATION ERROR: {}", &source);
                    if let Some(wgpu_core::error::ContextError {
                        string: _,
                        cause,
                        label_key: _,
                        label: _,
                    }) = source.downcast_ref::<wgpu_core::error::ContextError>()
                    {
                        self.validation_error = Some(cause.to_string());
                    } else {
                        self.validation_error = Some(description);
                    }
                } else {
                    panic!("{err}");
                }
            } else {
                self.pipeline = pipeline;
                self.validation_error = None;
            }
        }

        (self.timestamp.render_timestamp_writes(), &self.pipeline)
    }
    fn reconstruct(
        label: &str,
        device: &wgpu::Device,
        filename: &str,
        source: &str,
        pipeline_layout: &wgpu::PipelineLayout,
        vertex_buffer_layout: Vec<wgpu::VertexBufferLayout<'static>>,
        surface_format: wgpu::TextureFormat,
        vs_main: &'static str,
        fs_main: Option<&'static str>,
        polygon_mode: wgpu::PolygonMode,
        write_depth: bool,
    ) -> wgpu::RenderPipeline {
        let start = Instant::now();
        let shader_module = create_shader_module(
            &device,
            wgpu::ShaderModuleDescriptor {
                label: Some(filename),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
            },
        );

        let targets = [Some(wgpu::ColorTargetState {
            format: surface_format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: vs_main,
                buffers: &vertex_buffer_layout,
                compilation_options: default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, //Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode,
                conservative: false,
            },
            depth_stencil: write_depth.then(|| wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: fs_main.map(|fs_main| wgpu::FragmentState {
                module: &shader_module,
                entry_point: fs_main,
                targets: &targets,
                compilation_options: default(),
            }),
            multiview: None,
        });
        println!("Updated {filename} pipeline in {:?}", start.elapsed());
        render_pipeline
    }
}

pub(crate) use egui_render::EguiRenderer;
mod egui_render {
    pub(crate) struct EguiRenderer {
        ctx: egui::Context,
        state: egui_winit::State,
        renderer: egui_wgpu::Renderer,
    }
    impl EguiRenderer {
        pub(crate) fn new(
            device: &wgpu::Device,
            color_format: wgpu::TextureFormat,
            //depth_format: wgpu::TextureFormat,
            window: &winit::window::Window,
        ) -> Self {
            let ctx = egui::Context::default();
            let id = ctx.viewport_id();

            let visuals = egui::Visuals::default();

            ctx.set_visuals(visuals);

            let state = egui_winit::State::new(ctx.clone(), id, &window, None, None);

            let renderer = egui_wgpu::Renderer::new(device, color_format, None, 1);

            Self {
                ctx,
                state,
                renderer,
            }
        }
        pub(crate) fn input(
            &mut self,
            window: &winit::window::Window,
            event: &winit::event::WindowEvent,
        ) -> egui_winit::EventResponse {
            self.state.on_window_event(window, event)
        }
        pub(crate) fn draw(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            encoder: &mut wgpu::CommandEncoder,
            window: &winit::window::Window,
            window_surface_view: &wgpu::TextureView,
            screen_descriptor: egui_wgpu::ScreenDescriptor,
            run_ui: impl FnOnce(&egui::Context),
        ) {
            let input = self.state.take_egui_input(&window);
            let output = self.ctx.run(input, |ui| run_ui(ui));

            self.state
                .handle_platform_output(&window, output.platform_output);

            let paint_jobs = self.ctx.tessellate(output.shapes, output.pixels_per_point);

            for (id, image_delta) in &output.textures_delta.set {
                self.renderer
                    .update_texture(device, queue, *id, image_delta)
            }

            let _: Vec<wgpu::CommandBuffer> = self.renderer.update_buffers(
                device,
                queue,
                encoder,
                &paint_jobs,
                &screen_descriptor,
            );

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &window_surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.renderer
                .render(&mut rpass, &paint_jobs, &screen_descriptor);
            drop(rpass);
            for texture in &output.textures_delta.free {
                self.renderer.free_texture(texture);
            }
        }
    }
}

use limits::LimitsCmp;
mod limits;

#[derive(Default)]
struct TimeStamps {
    forward: (f64, f64),
    wireframe: (f64, f64),
    depth: (f64, f64),
    deferred: (f64, f64),
    sdf: (f64, f64),
    triangle_allocation: (f64, f64),
    prefix_top: (f64, f64),
    triangle_writeback: (f64, f64),
    cull: (f64, f64),
    fps: f64,
}

struct GuiState {
    culling: bool,
    distance_culling: bool,
    cull_radius: f32,
    wireframe: bool,
    forward: bool,
    fog_radius: f32,
    fog: bool,
    show_render: bool,
    deferred: bool,
    depth_pre_pass: bool,
    clear_render_times: bool,
    compile_fail: Option<String>,

    timestamps: TimeStamps,

    pos: Point3<u32>,

    fog_color: [u8; 3],
    diffuse_color: [u8; 3],
    specular_color: [u8; 3],
    light_color: [u8; 3],
    sun_color: [u8; 3],
}
fn srgb(c: f32) -> u8 {
    (c * 255.0) as u8
}
impl GuiState {
    fn new() -> Self {
        Self {
            culling: true,
            cull_radius: 6.0,
            distance_culling: true,
            wireframe: false,
            forward: true,
            deferred: false,
            fog_radius: 0.9,
            fog: true,
            timestamps: TimeStamps::default(),
            show_render: true,
            clear_render_times: true,
            depth_pre_pass: false,
            compile_fail: None,
            fog_color: [0.529, 0.808, 0.922].map(|c| c * 0.7).map(srgb),

            light_color: [0.780, 0.141, 0.694].map(srgb),
            diffuse_color: [0.5, 0.5, 0.5].map(srgb),

            specular_color: [0.5, 0.5, 0.5].map(srgb),
            sun_color: [0.98, 0.77, 0.40].map(srgb),
            pos: (0, 0, 0).into(),
        }
    }

    fn run(&mut self, ui: &egui::Context) {
        egui::Window::new("Options")
            .default_open(true)
            .min_width(200.0)
            .resizable(true)
            .anchor(egui::Align2::LEFT_TOP, [0.0, 0.0])
            .show(&ui, |ui| {
                ui.checkbox(&mut self.forward, "Forward shading");
                ui.checkbox(&mut self.deferred, "Deferred shading");
                ui.checkbox(&mut self.wireframe, "Wireframe");
                ui.checkbox(&mut self.depth_pre_pass, "Depth pre pass");

                ui.separator();

                ui.checkbox(&mut self.culling, "Frustrum culling");
                if self.culling {
                    ui.checkbox(&mut self.distance_culling, "Distance culling");
                    if self.distance_culling {
                        ui.add(egui::Slider::new(&mut self.cull_radius, 1.0..=16.0).text("radius"));
                    }
                }

                ui.separator();

                if self.culling && self.distance_culling {
                    ui.checkbox(&mut self.fog, "Distance fog");
                    if self.fog {
                        ui.add(
                            egui::Slider::new(&mut self.fog_radius, 0.1..=1.0)
                                .text("of cull radius"),
                        );
                    }

                    ui.separator();
                }

                ui.label(format!("FPS (vsync): {:.1}", self.timestamps.fps));
                ui.label(format!(
                    "time per frame: {:?}",
                    Duration::from_secs_f64(
                        if self.forward {
                            self.timestamps.forward.0
                        } else {
                            0.0
                        } + if self.wireframe {
                            self.timestamps.wireframe.0
                        } else {
                            0.0
                        } + if self.deferred {
                            self.timestamps.deferred.0
                        } else {
                            0.0
                        } + if self.deferred || self.depth_pre_pass {
                            self.timestamps.depth.0
                        } else {
                            0.0
                        }
                    )
                ));
                ui.label(format!(
                    "pos: {}, {}, {}",
                    self.pos.x, self.pos.y, self.pos.z
                ));
                ui.separator();

                if let Some(err) = self.compile_fail.as_ref() {
                    ui.label("COMPILE FAIL");
                    let txt = egui::widget_text::RichText::new(err).monospace();
                    ui.label(txt);
                    ui.separator();
                } else {
                    egui::Grid::new("my_grid").num_columns(2).show(ui, |ui| {
                        let mut color = |label, write| {
                            ui.label(label);
                            egui::widgets::color_picker::color_edit_button_srgb(ui, write);
                            ui.end_row();
                        };
                        color("specular", &mut self.specular_color);
                        color("fog", &mut self.fog_color);
                        color("light", &mut self.light_color);
                        color("sun", &mut self.sun_color);
                        color("diffuse", &mut self.diffuse_color);
                    });

                    ui.checkbox(&mut self.show_render, "Show render times in graph");
                    ui.checkbox(
                        &mut self.clear_render_times,
                        "Clear render times on each update",
                    );
                    egui_plot::Plot::new("")
                        .legend(egui_plot::Legend::default())
                        .allow_drag(false)
                        .allow_zoom(false)
                        .allow_scroll(false)
                        .allow_boxed_zoom(false)
                        .allow_double_click_reset(false)
                        .y_axis_label("time [ms]")
                        .show(ui, |plot_ui| {
                            let mut v = Vec::new();
                            if self.show_render {
                                v.push((self.timestamps.forward, "render"));
                                v.push((self.timestamps.wireframe, "wireframe"));
                                v.push((self.timestamps.depth, "depth"));
                                v.push((self.timestamps.deferred, "deferred"));
                            }
                            v.extend([
                                (self.timestamps.sdf, "sdf"),
                                (self.timestamps.triangle_allocation, "triangle_allocation"),
                                (self.timestamps.prefix_top, "prefix_top"),
                                (self.timestamps.triangle_writeback, "triangle_writeback"),
                                (self.timestamps.cull, "cull"),
                            ]);
                            for (i, ((mean, _), label)) in v.iter().enumerate() {
                                let bar = egui_plot::BarChart::new(vec![egui_plot::Bar::new(
                                    i as f64,
                                    (*mean) * 1000.0,
                                )])
                                .name(label);
                                plot_ui.bar_chart(bar);
                            }
                        });
                }
            });
    }
    fn cull_radius(&self) -> f32 {
        if self.distance_culling {
            self.cull_radius
        } else {
            f32::INFINITY
        }
    }
    fn fog(&self) -> f32 {
        if self.culling && self.fog && self.distance_culling {
            self.fog_radius * self.cull_radius
        } else {
            f32::INFINITY
        }
    }
}

struct State<'a> {
    device: wgpu::Device,
    queue: wgpu::Queue,
    instance: wgpu::Instance,
    surface: wgpu::Surface<'a>,
    surface_config: wgpu::SurfaceConfiguration,
    bitpack_render_pipeline: UpdateRenderPipeline,
    wireframe_render_pipeline: UpdateRenderPipeline,
    depth_texture: Texture,
    camera: Camera,

    bitpack_bind_group: wgpu::BindGroup,
    t0: Instant,
    camera_buffer: MetaBuffer,
    last_time: Instant,
    last_count: u32,
    storage: Storage,

    egui: EguiRenderer,
    gui: GuiState,

    deferred_bind_group_depth: wgpu::BindGroup,
    deferred_bind_group_camera: wgpu::BindGroup,
    deferred_render_pipeline: UpdateRenderPipeline,
    depth_render_pipeline: UpdateRenderPipeline,
}
impl<'a> State<'a> {
    fn new(window: &'a winit::window::Window, cube_march: &CubeMarch) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: default(), //wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
            // | wgpu::InstanceFlags::DEBUG
            // | wgpu::InstanceFlags::VALIDATION
            dx12_shader_compiler: default(),
            gles_minor_version: default(),
        });
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        eprintln!("--adapters:");
        for adapter in &adapters {
            dbg!(adapter.get_info());
        }
        eprintln!("--adapters");

        let surface: wgpu::Surface<'_> = instance.create_surface(window).unwrap();
        let adapter = dbg!(adapters)
            .into_iter()
            .find(|a| dbg!(a).is_surface_supported(&surface))
            .unwrap();
        let adapter_limits = adapter.limits();
        let surface_caps = dbg!(surface.get_capabilities(&adapter));
        let surface_format = dbg!(surface
            .get_capabilities(&adapter)
            .formats
            .into_iter()
            .find(|format| format.is_srgb())
            .unwrap());
        // egui_wgpu::preferred_framebuffer_format(&surface.get_capabilities(&adapter).formats)
        //     .unwrap();

        dbg!(adapter.features());
        dbg!(!adapter.features());
        dbg!(adapter.get_info());

        let mut required_limits = wgpu::Limits::downlevel_defaults();
        required_limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;
        required_limits.max_buffer_size = adapter_limits.max_buffer_size;
        println!("{}", LimitsCmp::new(&adapter.limits(), &required_limits));
        let (device, queue) = match adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::MULTI_DRAW_INDIRECT
                        | wgpu::Features::INDIRECT_FIRST_INSTANCE
                        | wgpu::Features::TIMESTAMP_QUERY
                        // | wgpu::Features::VERTEX_WRITABLE_STORAGE,
                        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
                        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                        | wgpu::Features::SUBGROUP
                        | wgpu::Features::POLYGON_MODE_LINE
                        | wgpu::Features::SHADER_F64,
                    required_limits,
                },
                None,
            )
            .block_on()
        {
            Ok(o) => o,
            Err(e) => panic!("{}", e),
        };

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: 200,
            height: 200,
            present_mode: wgpu::PresentMode::AutoVsync, // wgpu::PresentMode::AutoNoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: dbg!(&surface_caps.alpha_modes)[0],
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let camera = Camera::new(&surface_config);

        let gui = GuiState::new();

        let camera_buffer =
            MetaBuffer::new_uniform(&device, UNIFORM | COPY_DST, &camera.uniform(&gui));

        let depth_texture = Texture::new_depth(&device, &surface_config);

        let storage = Storage::new(&device, &cube_march, &camera_buffer);

        let bitpack_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bitpack_layout"),
            entries: &[
                camera_buffer.binding_layout(
                    VERTEX_STAGE | FRAGMENT_STAGE,
                    wgpu::BufferBindingType::Uniform,
                    0,
                ),
                storage.trigen.render_case_uniform.binding_layout(
                    VERTEX_STAGE,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    1,
                ),
            ],
        });

        let bitpack_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bitpack_bind_group"),
            layout: &bitpack_layout,
            entries: &[
                camera_buffer.binding(0),
                storage.trigen.render_case_uniform.binding(1),
            ],
        });

        let bitpack_pipeline_layout = Rc::new(device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("bitpack_pipeline_layout"),
                bind_group_layouts: &[&bitpack_layout],
                push_constant_ranges: &[],
            },
        ));

        let bitpack_vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<u32>() as _,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Uint32],
        };

        let bitpack_render_pipeline = UpdateRenderPipeline::new(
            "forward shading",
            &device,
            bitpack_pipeline_layout.clone(),
            vec![bitpack_vertex_buffer_layout.clone()],
            surface_config.format,
            source!("chunk_draw"),
            "vs_main",
            Some("fs_main"),
            wgpu::PolygonMode::Fill,
            true,
        );

        let deferred_bind_group_camera_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("deferred bind group camera layout"),
                entries: &[camera_buffer.binding_layout(
                    VERTEX_STAGE | FRAGMENT_STAGE,
                    wgpu::BufferBindingType::Uniform,
                    0,
                )],
            });

        let deferred_bind_group_camera = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deferred bind group camera"),
            layout: &deferred_bind_group_camera_layout,
            entries: &[camera_buffer.binding(0)],
        });

        let deferred_bind_group_depth_layout = create_deferred_bind_group_depth_layout(&device);

        let deferred_bind_group_depth = create_deferred_bind_group_depth(
            &device,
            &deferred_bind_group_depth_layout,
            &depth_texture,
        );

        let deferred_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("deferred pipeline layout"),
                bind_group_layouts: &[
                    &deferred_bind_group_camera_layout,
                    &deferred_bind_group_depth_layout,
                ],
                push_constant_ranges: &[],
            });

        let deferred_render_pipeline = UpdateRenderPipeline::new(
            "deferred shading",
            &device,
            Rc::new(deferred_pipeline_layout),
            vec![],
            surface_config.format,
            source!("chunk_draw"),
            "vs_deferred",
            Some("fs_deferred"),
            wgpu::PolygonMode::Fill,
            false,
        );

        let depth_render_pipeline = UpdateRenderPipeline::new(
            "depth pass",
            &device,
            bitpack_pipeline_layout.clone(),
            vec![bitpack_vertex_buffer_layout.clone()],
            surface_config.format,
            source!("chunk_draw"),
            "vs_main",
            None,
            wgpu::PolygonMode::Fill,
            true,
        );

        let wireframe_render_pipeline = UpdateRenderPipeline::new(
            "wireframe",
            &device,
            bitpack_pipeline_layout,
            vec![bitpack_vertex_buffer_layout],
            surface_config.format,
            source!("chunk_draw"),
            "vs_main",
            Some("fs_wire"),
            wgpu::PolygonMode::Line,
            true,
        );

        let last_time = Instant::now();
        let last_count = 0;
        let t0 = Instant::now();

        let egui = EguiRenderer::new(&device, surface_format, window);

        Self {
            device,
            queue,
            instance,
            surface,
            depth_texture,
            camera,
            surface_config,
            t0,
            camera_buffer,
            last_time,
            last_count,
            bitpack_render_pipeline,
            bitpack_bind_group,
            storage,
            wireframe_render_pipeline,
            egui,
            gui,
            deferred_bind_group_depth,
            deferred_bind_group_camera,
            deferred_render_pipeline,
            depth_render_pipeline,
        }
    }
    fn draw(
        &mut self,
        window: &'a winit::window::Window,
        held_keys: &BTreeSet<KeyCode>,
        pressed_keys: &BTreeSet<KeyCode>,
    ) {
        let dt = replace(&mut self.t0, Instant::now())
            .elapsed()
            .as_secs_f64();
        let output = self.surface.get_current_texture().unwrap();
        let view = &output.texture.create_view(&default());
        let mut encoder = self.device.create_command_encoder(&default());
        let pos = pos_to_target(self.camera.pos);
        self.gui.pos = pos;
        self.storage.dispatch(&self.device, &self.queue, pos);
        let mut first_color = true;
        let mut first_depth = true;

        if self.gui.culling {
            self.storage.cull(&mut encoder);
        }

        let clear_color = wgpu::Color {
            r: (self.gui.fog_color[0] as f64) / 255.0,
            g: (self.gui.fog_color[1] as f64) / 255.0,
            b: (self.gui.fog_color[2] as f64) / 255.0,
            a: 1.0,
        };

        if self.gui.depth_pre_pass || self.gui.deferred {
            let (timestamp_writes, pipeline) = self.depth_render_pipeline.get(&self.device);
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: if first_depth {
                            wgpu::LoadOp::Clear(1.0)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, &self.bitpack_bind_group, &[]);
            self.storage.draw(&mut render_pass, self.gui.culling);
            first_depth = false;
        }

        if self.gui.deferred {
            let (timestamp_writes, pipeline) = self.deferred_render_pipeline.get(&self.device);
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes,
                occlusion_query_set: None,
            });
            render_pass.set_bind_group(0, &self.deferred_bind_group_camera, &[]);
            render_pass.set_bind_group(1, &self.deferred_bind_group_depth, &[]);
            render_pass.set_pipeline(pipeline);
            render_pass.draw(0..6, 0..1);

            first_color = false;
            first_depth = false;
        }

        if self.gui.forward {
            let (timestamp_writes, pipeline) = self.bitpack_render_pipeline.get(&self.device);
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if first_color {
                            wgpu::LoadOp::Clear(clear_color)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: if first_depth {
                            wgpu::LoadOp::Clear(1.0)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, &self.bitpack_bind_group, &[]);
            self.storage.draw(&mut render_pass, self.gui.culling);
            first_color = false;
            first_depth = false;
        }

        if self.gui.wireframe {
            let (timestamp_writes, pipeline) = self.wireframe_render_pipeline.get(&self.device);
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if first_color {
                            wgpu::LoadOp::Clear(clear_color)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: if first_depth {
                            wgpu::LoadOp::Clear(1.0)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, &self.bitpack_bind_group, &[]);
            self.storage.draw(&mut render_pass, self.gui.culling);
            first_color = false;
            first_depth = false;
        }

        {
            let screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.surface_config.width, self.surface_config.height],
                pixels_per_point: window.scale_factor() as f32,
            };

            self.egui.draw(
                &self.device,
                &self.queue,
                &mut encoder,
                window,
                &view,
                screen_descriptor,
                |ui| self.gui.run(ui),
            );
        }

        self.camera
            .update(&self.surface_config, dt, held_keys, pressed_keys);
        self.camera_buffer
            .write(&self.queue, self.camera.uniform(&self.gui));
        let elapsed = self.last_time.elapsed().as_secs_f64();
        self.last_count += 1;

        if elapsed > 1.0 {
            let fps = self.last_count as f64 / elapsed;
            println!(
                "FPS: {fps}, tri count: {} million, draw calls: {}",
                self.storage.offset as f64 / 1_000_000.0,
                self.storage.data.len(),
            );

            self.last_count = 0;
            self.last_time = Instant::now();

            let fail_ref = [
                &self.deferred_render_pipeline,
                &self.bitpack_render_pipeline,
                &self.wireframe_render_pipeline,
            ]
            .iter()
            .find_map(|p| p.validation_error.as_ref());
            if fail_ref.is_some() ^ self.gui.compile_fail.is_some() {
                self.gui.compile_fail = fail_ref.map(String::clone);
            }
            {
                if self.gui.clear_render_times {
                    self.bitpack_render_pipeline.timestamp.reset();
                    self.wireframe_render_pipeline.timestamp.reset();
                    self.depth_render_pipeline.timestamp.reset();
                    self.deferred_render_pipeline.timestamp.reset();
                }

                let mut gui_timestamps = TimeStamps::default();
                gui_timestamps.fps = fps;

                let mut timestamps = [
                    (
                        &mut self.depth_render_pipeline.timestamp,
                        &mut gui_timestamps.depth,
                    ),
                    (
                        &mut self.deferred_render_pipeline.timestamp,
                        &mut gui_timestamps.deferred,
                    ),
                    (
                        &mut self.bitpack_render_pipeline.timestamp,
                        &mut gui_timestamps.forward,
                    ),
                    (
                        &mut self.wireframe_render_pipeline.timestamp,
                        &mut gui_timestamps.wireframe,
                    ),
                    (
                        &mut self.storage.trigen.sdf.timestamp,
                        &mut gui_timestamps.sdf,
                    ),
                    (
                        &mut self.storage.trigen.triangle_allocation.timestamp,
                        &mut gui_timestamps.triangle_allocation,
                    ),
                    (
                        &mut self.storage.trigen.prefix_top.timestamp,
                        &mut gui_timestamps.prefix_top,
                    ),
                    (
                        &mut self.storage.trigen.triangle_writeback.timestamp,
                        &mut gui_timestamps.triangle_writeback,
                    ),
                    (
                        &mut self.storage.cull_kernel.timestamp,
                        &mut gui_timestamps.cull,
                    ),
                ];

                for (timestamp, _) in timestamps.iter_mut() {
                    timestamp.resolve_copy(&mut encoder);
                }

                self.queue.submit([encoder.finish()]);

                for (timestamp, _) in timestamps.iter_mut() {
                    timestamp.resolve_map();
                }

                self.device.poll(wgpu::MaintainBase::Wait);

                let seconds_per_tick = self.queue.get_timestamp_period() as f64 / 1_000_000_000.0;

                for (timestamp, gtime) in timestamps.iter_mut() {
                    timestamp.resolve_unmap(seconds_per_tick);
                    **gtime = (timestamp.mean, timestamp.sd);
                }
                self.gui.timestamps = gui_timestamps;
            }
        } else {
            self.queue.submit([encoder.finish()]);
        }
        output.present();
    }
    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.surface_config.width = size.width;
            self.surface_config.height = size.height;
            self.surface.configure(&self.device, &self.surface_config);
            self.depth_texture = Texture::new_depth(&self.device, &self.surface_config);
            let deferred_bind_group_depth_layout =
                create_deferred_bind_group_depth_layout(&self.device);
            self.deferred_bind_group_depth = create_deferred_bind_group_depth(
                &self.device,
                &deferred_bind_group_depth_layout,
                &self.depth_texture,
            );
        }
    }
}

fn create_deferred_bind_group_depth(
    device: &wgpu::Device,
    deferred_bind_group_depth_layout: &wgpu::BindGroupLayout,
    depth_texture: &Texture,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("deferred bind group depth"),
        layout: deferred_bind_group_depth_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&depth_texture.view),
        }],
    })
}

fn create_deferred_bind_group_depth_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("deferred bind group depth layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    })
}

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let cube_march = CubeMarch::new();
    env_logger::init();
    println!("Do not forget `nix-shell`");
    let event_loop = EventLoop::new()
        .map_err(|e| (e, "you probably forgot nix-shell"))
        .unwrap();
    let window: winit::window::Window = WindowBuilder::new()
        .with_title("Marching Cubes")
        .build(&event_loop)
        .unwrap();
    let window = &window;

    println!("construct state");
    let mut state = State::new(window, &cube_march);
    println!("start event loop");

    let mut held_keys = BTreeSet::new();
    let mut pressed_keys = BTreeSet::new();

    event_loop
        .run(|event, window_target| match event {
            Event::NewEvents(_) => (),
            Event::WindowEvent { window_id, event } => {
                if window_id == window.id() {
                    let egui_winit::EventResponse { consumed, .. }= state.egui.input(window, &event);
                    if consumed { return }
                    match event {
                        WindowEvent::CloseRequested | WindowEvent::KeyboardInput {
                            event: KeyEvent { logical_key: winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape), .. }, ..
                        } => window_target.exit(),
                        WindowEvent::RedrawRequested => {
                            state.draw(window, &held_keys, &pressed_keys);
                            pressed_keys.clear();
                            window.request_redraw();
                        }
                        WindowEvent::Resized(size) => {
                            state.resize(size);
                        }
                        WindowEvent::KeyboardInput { event: winit::event::KeyEvent { physical_key: winit::keyboard::PhysicalKey::Code(key_code), state, ..}, ..} => {
                            if state.is_pressed() {
                                let _: bool = held_keys.insert(key_code);
                                let _: bool = pressed_keys.insert(key_code);
                            } else {
                                let _: bool = held_keys.remove(&key_code);
                                let _: bool = pressed_keys.remove(&key_code);
                            }
                        }
                        _ => (),
                    }
                }
            }
            _ => (),
        })
        .unwrap();
}
