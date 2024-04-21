use cgmath::prelude::*;
use std::{
    array,
    cmp::Eq,
    collections::{BTreeSet, HashMap, VecDeque},
    fmt::Display,
    marker::Copy,
    mem::size_of,
    slice::from_ref,
    sync::Arc,
    time::{Duration, Instant, UNIX_EPOCH},
};

include!(concat!(env!("OUT_DIR"), "/compile_time.rs"));

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
}
impl Storage {
    fn new(device: &wgpu::Device, cube_march: &CubeMarch) -> Self {
        let (send, recv) = std::sync::mpsc::channel();
        // 128 MiB = 134 million triangles
        let capacity: u32 = 32 * 1024 * 1024;
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
        let indirect_draw_buffer = MetaBuffer::new_uninit(
            device,
            "indirect_draw_buffer",
            INDIRECT | COPY_DST,
            32 * 1024 * 1024,
        ); // TODO: pick size
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
        }
    }
    fn dispatch(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let target: (u32, u32, u32) = (0, 0, 0);
        let radius = 32;

        if self.capacity < self.offset + MAX_TRIS_PER_CHUNK {
            return;
        }

        if self.compute_queue.len() == 0 {
            for x in target.0.saturating_sub(radius)..(target.0 + radius).min(1023) {
                for y in target.1.saturating_sub(radius)..(target.1 + radius).min(1023) {
                    for z in target.2.saturating_sub(radius)..(target.2 + radius).min(1023) {
                        if self.in_queue.insert((x, y, z)) {
                            self.compute_queue.push_back((x, y, z));
                        }
                    }
                }
            }
        }

        for _ in 0..1 {
            if self.capacity < self.offset + MAX_TRIS_PER_CHUNK {
                return;
            }
            let Some((x, y, z)) = self.compute_queue.pop_front() else {
                return;
            };

            if let Some((x, y, z)) = self.awaiting.take() {
                device.poll(wgpu::Maintain::Wait); // :(
                let _: () = self.recv.recv().unwrap().unwrap(); // TODO: read and unmap in callback.
                let count = bytemuck::cast_slice::<u8, u32>(
                    &self
                        .count_staging_buffer
                        .buffer
                        .slice(..)
                        .get_mapped_range(),
                )[0];
                self.count_staging_buffer.buffer.unmap();
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
    fn draw<'this, 'pass>(&'this mut self, pass: &mut wgpu::RenderPass<'pass>)
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
            &self.indirect_draw_buffer.buffer,
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

use cgmath::SquareMatrix;
use pollster::FutureExt;
use wgpu::{hal::Instance, util::DeviceExt};
use winit::{event::*, event_loop::EventLoop, window::WindowBuilder};

fn default<T: Default>() -> T {
    Default::default()
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct Vertex {
    pos: [f32; 3],
}

// ... -(vertex shader)> triangle + index.
mod marching_cubes;
use marching_cubes::{cube_march_cpu, CubeMarch};

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct TriAllocUniform {
    // 3 bit (0..=4)
    case_to_size: [u32; 256],
}
impl TriAllocUniform {
    fn new(case: &CubeMarch) -> Self {
        Self {
            case_to_size: case.case_to_size.map(|size| size as u32),
        }
    }
}
unsafe impl bytemuck::NoUninit for TriWriteBackUniform {}
#[derive(Copy, Clone)]
#[repr(C)]
struct TriWriteBackUniform {
    /// triangle lives on 3 edges
    /// each edge is on 2 corners
    /// edge0 edge1, edge2
    /// XYZ XYZ XYZ XYZ XYZ XYZ TRIA = 22 significant bits
    offset_to_bitpack: [u32; 732],
    /// 10 bit (0..732)
    case_to_offset: [u32; 257],
    /// Padding
    _unused0: u32,
    /// Padding
    _unused1: u32,
    /// Padding
    _unused2: u32,
}
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
unsafe impl bytemuck::NoUninit for RenderCaseUniform {}
#[derive(Copy, Clone)]
#[repr(C)]
struct RenderCaseUniform {
    /// XYZ XYZ XYZ XYZ XYZ XYZ = 18 bit
    triangle_to_corners: [u32; 135],
    /// Padding
    _unused0: u32,
}
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
            _unused0: 0,
        }
    }
}

struct MetaBuffer {
    buffer: wgpu::Buffer,
    bytes: u64,
    ty: wgpu::BufferBindingType,
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
            wgpu::BufferBindingType::Storage { read_only: false },
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
            wgpu::BufferBindingType::Storage { read_only: true },
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
            wgpu::BufferBindingType::Uniform,
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

        Self {
            buffer,
            bytes,
            ty: wgpu::BufferBindingType::Storage { read_only: false },
        }
    }
    fn new_i(
        device: &wgpu::Device,
        name: &str,
        ty: wgpu::BufferBindingType,
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
            ty,
        }
    }
    fn binding_layout(&self, binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: self.ty,
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

#[rustfmt::skip]
macro_rules! source {
    ($filename:expr) => {{
        #[cfg(debug_assertions)]
        { ($filename, &std::fs::read_to_string(concat!("src/",$filename,".wgsl")).unwrap()) }
        #[cfg(not(debug_assertions))]
        { ($filename, include_str!(concat!($filename, ".wgsl"))) }
    }};
}

struct Kernel {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    x: u32,
}
impl Kernel {
    fn new(
        device: &wgpu::Device,
        (name, source): (&str, &str),
        buffers: &[&MetaBuffer],
        x: u32,
    ) -> Self {
        let (binding_entry_layouts, binding_entries): (Vec<_>, Vec<_>) = buffers
            .iter()
            .zip(0..)
            .map(|(buffer, i)| (buffer.binding_layout(i), buffer.binding(i)))
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
        });
        Self {
            pipeline,
            bind_group,
            x,
        }
    }
    fn dispatch<'s: 'c, 'c>(&'s self, pass: &mut wgpu::ComputePass<'c>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.x, 1, 1);
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
    /// [u32; 257]
    //case_triangle_count: MetaBuffer,
    /// `32^3`
    triangle_allocation: Kernel,
    /// [u32; 32^3 + 1]
    triangle_count_prefix: MetaBuffer,
    /// `256`
    prefix_top: Kernel,
    /// `[u32]`
    //case_triangle_offset: MetaBuffer,
    /// `[u32]`
    //case_triangle_number: MetaBuffer,
    /// `32^3`
    triangle_writeback: Kernel,
    staging_buffer: MetaBuffer,
    render_case_uniform: MetaBuffer,
    // query_buffer0: MetaBuffer,
    // query_buffer1: MetaBuffer,
}


struct QuerySetState<T: Seq> {
    set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    i: u32,
    _phantom: std::marker::PhantomData<T>,
}
impl<T: Seq> QuerySetState<T> {
    fn next(mut self) -> QuerySetState<T::Next> {
        self.i += 1;
        let Self {
            set,
            resolve_buffer,
            staging_buffer,
            i,
            _phantom,
        } = self;
        QuerySetState {
            set,
            resolve_buffer,
            staging_buffer,
            i,
            _phantom: std::marker::PhantomData,
        }
    }
    fn new(device: &wgpu::Device) -> QuerySetState<MarkerStart> {
        let set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("query set"),
            ty: wgpu::QueryType::Timestamp,
            count: wgpu::QUERY_SET_MAX_QUERIES,
        });
        let i = 0;
        let size = wgpu::QUERY_SET_MAX_QUERIES as u64 * size_of::<u64>() as u64;

        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resolve buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resolve buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        QuerySetState::<MarkerPreCompute> {
            set,
            i,
            resolve_buffer,
            staging_buffer,
            _phantom: std::marker::PhantomData,
        }
    }
    fn pass(self, pass: &mut wgpu::ComputePass) -> QuerySetState<T::Next> {
        pass.write_timestamp(&self.set, self.i);
        self.next()
    }
    fn encoder(self, encoder: &mut wgpu::CommandEncoder) -> QuerySetState<T::Next> {
        encoder.write_timestamp(&self.set, self.i);
        self.next()
    }
}
impl QuerySetState<MarkerStart> {
    fn need_resolve(&self) -> bool {
        self.i < wgpu::QUERY_SET_MAX_QUERIES / 2
    }
    fn resolve(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&default());
        encoder.resolve_query_set(&self.set, 0..self.i, &self.resolve_buffer, 0);

        let bytes = self.i as u64 * size_of::<u64>() as u64;

        encoder.copy_buffer_to_buffer(&self.resolve_buffer, 0, &self.staging_buffer, 0, bytes);

        self.staging_buffer
            .slice(0..bytes)
            .map_async(wgpu::MapMode::Read, |_| ());

        queue.submit([encoder.finish()]);

        device.poll(wgpu::MaintainBase::Wait);

        let tmp = self.staging_buffer.slice(0..bytes).get_mapped_range();
        let data: &[QueryRaw] = bytemuck::cast_slice(&tmp);

        dbg!(data);

        let seconds_per_tick = (queue.get_timestamp_period() as f64) / 1_000_000_000.0;

        for d in data {
            dbg!(d.query(seconds_per_tick));
        }
    }
}

trait Seq {
    type Next: Seq;
}
macro_rules! seq {
    ($query:ident $raw:ident $start:ident $($a:ident $c:ident)*) => {
        $(struct $a;)*
        #[repr(C)]
        #[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct $raw { $($c: u64,)* }
        #[derive(Debug)]
        struct $query { $($c: Duration,)* }
        seq!($start [$($a)*]);
        seq!($query $raw ($($c)*));
    };
    ($query:ident $raw:ident ($a:ident $($as:ident)*)) => { seq!($query $raw ($a $($as)*), ($($as)* $a)); };
    ($query:ident $raw:ident ($($as:ident)*), ($($bs:ident)*)) => {
        impl $raw {
            fn query(self, seconds_per_tick: f64) -> $query {
                $( let $as = (self.$as as f64) * seconds_per_tick;)*
                $query {$(
                    $bs: Duration::from_secs_f64(($bs - $as).abs()),
                )*}
            }
        }
    };
    ($start:ident [$a:ident $($as:ident)*]) => { 
        type $start = $a;
        seq!([$a $($as)*], [$($as)* $a]); 
    };
    ([$($as:ident)*], [$($bs:ident)*]) => { $(impl Seq for $as { type Next = $bs; })* }
}

seq!(Query2 QueryRaw MarkerStart
    MarkerPreCompute pre_compute
    MarkerPrePass pre_pass
    MarkerSdf sdf
    MarkerTriangleAllocation triangle_allocation
    MarkerPrefixTop prefix_top
    MarkerTriangleWriteback triangle_writeback
    MarkerPostCompute post_compute
);


impl TriGenState {
    #[rustfmt::skip]
    fn new(device: &wgpu::Device, cube_march: &CubeMarch, triangle_storage: &MetaBuffer) -> Self {
        let sdf_data = MetaBuffer::new(device, "sdf_data", NONE, &vec![0.0_f32; 33*33*33 + 159]);

        let triangle_count_prefix = MetaBuffer::new(device, "triangle_count_prefix", COPY_SRC, &vec![0_u32; 32*32*32 + 1]);

        let staging_buffer = MetaBuffer::new_uninit(device, "tri count buffer", COPY_DST | MAP_READ, size_of::<u32>() as u64);

        // let query_buffer0 = MetaBuffer::new_uninit(device, "query buffer0", COPY_SRC | QUERY_RESOLVE, (std::mem::size_of::<u64>() * 2) as u64);
        // let query_buffer1 = MetaBuffer::new_uninit(device, "query buffer1", COPY_DST | MAP_READ, (std::mem::size_of::<u64>() * 2) as u64);

        let render_case_uniform = RenderCaseUniform::new(&cube_march);
        let render_case_uniform = MetaBuffer::new(device, "render_case_uniform", NONE, from_ref(&render_case_uniform));


        let chunk_uniform = ChunkUniform::default();
        let chunk_uniform = MetaBuffer::new_uniform(device, COPY_DST, &chunk_uniform);

        let tri_alloc_uniform = TriAllocUniform::new(&cube_march);
        let tri_alloc_uniform = MetaBuffer::new_constant(device, "TriAllocUniform", NONE, from_ref(&tri_alloc_uniform));

        let tri_wb_uniform = TriWriteBackUniform::new(&cube_march);
        let tri_wb_uniform = MetaBuffer::new_constant(device, "TriWriteBackUniform", NONE, from_ref(&tri_wb_uniform));

        let sdf = Kernel::new(device, source!("sdf"), &[&chunk_uniform, &sdf_data], (33*33*33 + 159)/256);
        let triangle_allocation = Kernel::new(device, source!("triangle_allocation"), &[&tri_alloc_uniform, &sdf_data, &triangle_count_prefix], (32*32*32)/128);
        let prefix_top = Kernel::new(device, source!("prefix_top"), &[&triangle_count_prefix], 1);
        let triangle_writeback = Kernel::new(device, source!("triangle_writeback"), &[&sdf_data, &triangle_count_prefix, &triangle_storage, &tri_wb_uniform, &chunk_uniform], 32*32*32/128);

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
            // query_buffer0,
            // query_buffer1,
        }

    }
    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        staging_buffer: &MetaBuffer,
        //mut qset: Option<&mut QuerySetState>,
    ) {
        // let mut encoder = device.create_command_encoder(&default());
        //encoder.write_timestamp(qset, 0);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        // if let Some(q) = &mut qset {
        //     q.pass(&mut pass);
        // }
        self.sdf.dispatch(&mut pass);
        // if let Some(q) = &mut qset {
        //     q.pass(&mut pass);
        // }
        self.triangle_allocation.dispatch(&mut pass);
        //if let Some(q) = &mut qset {
        //    q.pass(&mut pass);
        //}
        self.prefix_top.dispatch(&mut pass);
        //if let Some(q) = &mut qset {
        //    q.pass(&mut pass);
        //}
        self.triangle_writeback.dispatch(&mut pass);
        //if let Some(q) = &mut qset {
        //    q.pass(&mut pass);
        //}
        drop(pass);

        //encoder.write_timestamp(qset, 1);

        encoder.copy_buffer_to_buffer(
            &self.triangle_count_prefix.buffer,
            (32 * 32 * 32) * 4,
            &staging_buffer.buffer,
            0,
            4,
        );
        //encoder.resolve_query_set(&qset, 0..2, &self.query_buffer0.buffer, 0);
        //encoder.copy_buffer_to_buffer(&self.query_buffer0.buffer, 0, &self.query_buffer1.buffer, 0, 2 * std::mem::size_of::<u64>() as u64);
        // queue.submit([encoder.finish()]);
        //self.query_buffer1.buffer.slice(..).map_async(wgpu::MapMode::Read, |x| x.unwrap());
        // self.staging_buffer
        //     .buffer
        //     .slice(..)
        //     .map_async(wgpu::MapMode::Read, |x| x.unwrap());
        // device.poll(wgpu::Maintain::Wait);
        // let tmp = self
        //     .staging_buffer
        //     .buffer
        //     .slice(..)
        //     .get_mapped_range();
        // let data: &[u32] = bytemuck::cast_slice(&tmp);
        // //println!("Expected triangles written: {}.", data[0]);
        // let num_tris = data[0];
        // drop(tmp);
        // self.staging_buffer.buffer.unmap();

        //let tmp2 = self.query_buffer1.buffer.slice(..).get_mapped_range();
        //let data2: &[u64] = bytemuck::cast_slice(&tmp2);

        // let start = data2[0];
        // let end = data2[1];

        // let dur = std::time::Duration::from_secs_f64(((end - start) as f64 * queue.get_timestamp_period() as f64)*(1.0/1_000_000_000.0));

        // panic!("{dur:?}");

        // num_tris
    }
    // fn inspect(&self, triangle_storage: &MetaBuffer, device: &wgpu::Device, queue: &wgpu::Queue) {
    //     let read_buf = triangle_storage;

    //     let bytes = 3000 * 4;

    //     let tmp_buf = MetaBuffer::new_uninit(device, "tmp dst", MAP_READ | COPY_DST, bytes);

    //     let mut encoder = device.create_command_encoder(&default());

    //     encoder.copy_buffer_to_buffer(&read_buf.buffer, 0, &tmp_buf.buffer, 0, bytes);
    //     queue.submit([encoder.finish()]);

    //     tmp_buf
    //         .buffer
    //         .slice(..)
    //         .map_async(wgpu::MapMode::Read, |_| ());
    //     device.poll(wgpu::Maintain::Wait);

    //     let slice: &[u8] = &tmp_buf.buffer.slice(..).get_mapped_range();

    //     let slice_u32: &[u32] = bytemuck::cast_slice(slice);
    //     println!("{slice_u32:?}");

    //     // let diff = slice_u32.windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>();
    //     // println!("{diff:?}");
    //     // for c in slice_u32[1..].chunks(128) {
    //     //     println!("{:?}", c.last());
    //     // }
    // }
}

fn cmap<U: Copy, V: Copy, const N: usize>(a: [U; N], f: fn(U) -> V, v_default: V) -> [V; N] {
    let mut output: [V; N] = [v_default; N];
    let mut i = 0;
    while i < N {
        let input = a[i];
        output[i] = f(input);

        i += 1;
    }
    output
}
// gen u8 from sdf, calc number of verts per cell.
// prefix sum verts per cell
// draw the sum of verts of vertices
//
//
// passes:
// gen verts/cell + start prefix sum

fn indexify(v: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut idxs = Vec::new();
    let mut data = Vec::new();

    // slow but fine for cpu testing.
    let mut map: HashMap<[u32; 3], usize> = HashMap::new();

    for &v in v {
        let id = {
            let vb = v.map(|v| v.to_bits());
            if let Some(id) = map.get(&vb) {
                *id
            } else {
                let l = map.len();
                map.insert(vb, l);
                data.push(v);
                l
            }
        };
        idxs.push(id as _);
    }

    (data, idxs)
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    lmap: [[f32; 4]; 4],
    lmap_inv: [[f32; 4]; 4],
}
struct Camera {
    // camera pos
    eye: cgmath::Point3<f32>,
    // point camera is looking at
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}
impl Camera {
    #[rustfmt::skip]
    const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.5,
        0.0, 0.0, 0.0, 1.0,
    );
    fn view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        // move world to camera pos/rot
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        // projection matrix
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        // "normalized device coordinates" is different from OpenGL.
        Self::OPENGL_TO_WGPU_MATRIX * proj * view
    }
    fn uniform(&self) -> CameraUniform {
        let this = &self;
        // move world to camera pos/rot
        let view = cgmath::Matrix4::look_at_rh(this.eye, this.target, this.up);
        // projection matrix
        let proj = cgmath::perspective(cgmath::Deg(this.fovy), this.aspect, this.znear, this.zfar);

        let view_projection_matrix = Self::OPENGL_TO_WGPU_MATRIX * proj * view;

        let lmap = cgmath::Matrix4::from_angle_x(cgmath::Deg(34.4523423))
            * cgmath::Matrix4::from_angle_y(cgmath::Deg(47.3478))
            * cgmath::Matrix4::from_angle_z(cgmath::Deg(23.3894));
        let lmap_inv = lmap.invert().unwrap();

        CameraUniform {
            view: view.into(),
            view_proj: view_projection_matrix.into(),
            lmap: lmap.into(),
            lmap_inv: lmap_inv.into(),
        }
    }
    fn new(surface_config: &wgpu::SurfaceConfiguration) -> Self {
        Self {
            eye: (0.0, 1.0, 2.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: surface_config.width as f32 / surface_config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        }
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
struct UpdateRenderPipeline {
    path: String,
    filename: String,
    last_update: Duration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer_layout: wgpu::VertexBufferLayout<'static>,
    surface_format: wgpu::TextureFormat,
    pipeline_layout: wgpu::PipelineLayout,
}
impl UpdateRenderPipeline {
    fn new(
        device: &wgpu::Device,
        pipeline_layout: wgpu::PipelineLayout,
        vertex_buffer_layout: wgpu::VertexBufferLayout<'static>,
        surface_format: wgpu::TextureFormat,
        (filename, source): (&'static str, &str),
    ) -> Self {
        let pipeline = Self::reconstruct(
            device,
            filename,
            source,
            &pipeline_layout,
            vertex_buffer_layout.clone(),
            surface_format,
        );

        Self {
            path: format!("src/{filename}.wgsl"),
            last_update: COMPILE_TIME,
            pipeline,
            filename: filename.to_owned(),
            vertex_buffer_layout,
            surface_format,
            pipeline_layout,
        }
    }
    fn get(&mut self, device: &wgpu::Device) -> &wgpu::RenderPipeline {
        #[cfg(not(debug_assertions))]
        {
            return &self.pipeline;
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
                device,
                &self.filename,
                &source,
                &self.pipeline_layout,
                self.vertex_buffer_layout.clone(),
                self.surface_format,
            );
            if let Some(err) = device.pop_error_scope().block_on() {
                println!("VALIDATION ERROR: {err}");
            } else {
                self.pipeline = pipeline;
            }
        }

        &self.pipeline
    }
    fn reconstruct(
        device: &wgpu::Device,
        filename: &str,
        source: &str,
        pipeline_layout: &wgpu::PipelineLayout,
        vertex_buffer_layout: wgpu::VertexBufferLayout<'static>,
        surface_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let start = Instant::now();
        let shader_module = create_shader_module(
            &device,
            wgpu::ShaderModuleDescriptor {
                label: Some(filename),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
            },
        );

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(filename),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                buffers: &[vertex_buffer_layout],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
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
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });
        println!("Updated {filename} pipeline in {:?}", start.elapsed());
        render_pipeline
    }
}

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
struct LimitsCmp {
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
    fn new(a: &wgpu::Limits, b: &wgpu::Limits) -> Self {
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

struct State<'a> {
    device: wgpu::Device,
    queue: wgpu::Queue,
    instance: wgpu::Instance,
    surface: wgpu::Surface<'a>,
    surface_config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    bitpack_render_pipeline: UpdateRenderPipeline,
    depth_texture: Texture,
    camera_bind_group: wgpu::BindGroup,
    bitpack_bind_group: wgpu::BindGroup,
    // triangle_storage: wgpu::Buffer,
    camera: Camera,
    vertex_buffer: wgpu::Buffer,
    t0: Instant,
    num_vert: u32,
    camera_buffer: wgpu::Buffer,
    last_time: Instant,
    last_count: u32,
    storage: Storage,
}
impl<'a> State<'a> {
    fn new(window: &'a winit::window::Window) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
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
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        dbg!(adapter.features());
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
                        | wgpu::Features::VERTEX_WRITABLE_STORAGE, // TODO: omit
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

        let shader_source = include_str!("shader.wgsl");

        let shader_module = create_shader_module(
            &device,
            wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            },
        );

        // let pre = Instant::now();
        let vert: Vec<_> = cube_march_cpu().into_iter().flatten().collect(); // VERT;
                                                                             // panic!("{:?}", pre.elapsed());
        let num_vert = vert.len() as u32;

        let _ = indexify(&vert);

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<Vertex>() as _,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3],
        };

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&vert),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let t0 = Instant::now();
        let last_time = Instant::now();
        let last_count = 0;
        let camera = Camera::new(&surface_config);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[camera.uniform()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let depth_texture = Texture::new_depth(&device, &surface_config);

        // let triangle_storage = MetaBuffer::new_uninit(
        //     &device,
        //     "tri_buffer",
        //     STORAGE | COPY_SRC | COPY_DST | VERTEX,
        //     1048576,
        // );
        let cubemarch = CubeMarch::new();
        let storage = Storage::new(&device, &cubemarch);
        let triangle_storage = &storage.b0;
        // let trigen_state = TriGenState::new(&device, &cubemarch, &triangle_storage);

        let bitpack_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bitpack_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage.trigen.render_case_uniform.binding_layout(1),
            ],
        });

        let bitpack_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bitpack_bind_group"),
            layout: &bitpack_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                storage.trigen.render_case_uniform.binding(1),
            ],
        });
        let bitpack_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bitpack_pipeline_layout"),
                bind_group_layouts: &[&bitpack_layout],
                push_constant_ranges: &[],
            });

        let bitpack_vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<u32>() as _,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Uint32],
        };

        let bitpack_shader_module = create_shader_module(
            &device,
            wgpu::ShaderModuleDescriptor {
                label: Some(source!("chunk_draw").0),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    source!("chunk_draw").1,
                )),
            },
        );

        let bitpack_render_pipeline = UpdateRenderPipeline::new(
            &device,
            bitpack_pipeline_layout,
            bitpack_vertex_buffer_layout,
            surface_config.format,
            source!("chunk_draw"),
        );
        // let bitpack_render_pipeline =
        //     device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        //         label: Some("bitpack_render_pipeline"),
        //         layout: Some(&bitpack_pipeline_layout),
        //         vertex: wgpu::VertexState {
        //             module: &bitpack_shader_module,
        //             entry_point: "vs_main",
        //             buffers: &[bitpack_vertex_buffer_layout],
        //         },
        //         primitive: wgpu::PrimitiveState {
        //             topology: wgpu::PrimitiveTopology::TriangleList,
        //             strip_index_format: None,
        //             front_face: wgpu::FrontFace::Ccw,
        //             cull_mode: None,
        //             unclipped_depth: false,
        //             polygon_mode: wgpu::PolygonMode::Fill,
        //             conservative: false,
        //         },
        //         depth_stencil: Some(wgpu::DepthStencilState {
        //             format: Texture::DEPTH_FORMAT,
        //             depth_write_enabled: true,
        //             depth_compare: wgpu::CompareFunction::Less,
        //             stencil: wgpu::StencilState::default(),
        //             bias: wgpu::DepthBiasState::default(),
        //         }),
        //         multisample: wgpu::MultisampleState {
        //             count: 1,
        //             mask: !0,
        //             alpha_to_coverage_enabled: false,
        //         },
        //         fragment: Some(wgpu::FragmentState {
        //             module: &bitpack_shader_module,
        //             entry_point: "fs_main",
        //             targets: &[Some(wgpu::ColorTargetState {
        //                 format: surface_config.format,
        //                 blend: Some(wgpu::BlendState::REPLACE),
        //                 write_mask: wgpu::ColorWrites::ALL,
        //             })],
        //         }),
        //         multiview: None,
        //     });

        // let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        //     label: Some("query_set"),
        //     ty: wgpu::QueryType::Timestamp,
        //     count: 2,
        // });

        // let num_bitpack_tris = trigen_state.dispatch(&device, &queue);

        // trigen_state.inspect(&triangle_storage, &device, &queue);

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                buffers: &[vertex_buffer_layout],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
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
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        Self {
            device,
            queue,
            instance,
            surface,
            render_pipeline,
            depth_texture,
            camera_bind_group,
            camera,
            surface_config,
            vertex_buffer,
            t0,
            num_vert,
            camera_buffer,
            last_time,
            last_count,
            bitpack_render_pipeline,
            bitpack_bind_group,
            storage,
        }
    }
    fn render(&mut self) {
        let output = self.surface.get_current_texture().unwrap();
        let view = &output.texture.create_view(&default());
        let mut encoder = self.device.create_command_encoder(&default());
        self.storage.dispatch(&self.device, &self.queue);
        if false {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            render_pass.draw(0..self.num_vert, 0..1);
        } else {
            let pipeline = self.bitpack_render_pipeline.get(&self.device);
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // what to do with data from previous frame
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        // if color result should be stored.
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, &self.bitpack_bind_group, &[]);
            // render_pass.multi_draw_indirect(&self.vertex_buffer, 0, 1);
            self.storage.draw(&mut render_pass);
            // render_pass.draw(0..3, 0..self.num_bitpack_tris);
        }
        self.camera = Camera::new(&self.surface_config);
        let t = self.t0.elapsed().as_secs_f32();
        self.camera.eye = (
            2.0 * t.sin(),
            (t * 2.0_f32.sqrt() * 0.5).sin() * 0.1,
            2.0 * t.cos(),
        )
            .into();
        self.camera.eye *= 32.0;
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera.uniform()]),
        );
        self.queue.submit([encoder.finish()]);
        output.present();
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
        }
    }
    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.surface_config.width = size.width;
            self.surface_config.height = size.height;
            self.surface.configure(&self.device, &self.surface_config);
            self.depth_texture = Texture::new_depth(&self.device, &self.surface_config);
        }
    }
}

fn main() {
    env_logger::init();
    std::env::set_var("RUST_BACKTRACE", "1");
    //cube_march_cpu();
    println!("Do not forget `nix-shell`");
    let event_loop = EventLoop::new().unwrap();
    let window: winit::window::Window = WindowBuilder::new()
        .with_title("Marching Cubes")
        .build(&event_loop)
        .unwrap();
    let window = &window;

    println!("construct state");
    let mut state = State::new(window);
    println!("start event loop");

    event_loop
        .run(|event, window_target| match event {
            Event::NewEvents(_) => (),
            Event::WindowEvent { window_id, event } => {
                if window_id == window.id() {
                    match event {
                        WindowEvent::CloseRequested | WindowEvent::KeyboardInput {
                            event: KeyEvent { logical_key: winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape), .. }, ..
                        } => window_target.exit(),
                        WindowEvent::RedrawRequested => {
                            state.render();
                            window.request_redraw();
                        }
                        WindowEvent::Resized(size) => {
                            state.resize(size);
                        }
                        _ => (),
                    }
                }
            }
            _ => (),
        })
        .unwrap();
}
