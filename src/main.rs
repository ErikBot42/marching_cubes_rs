use std::{
    array,
    cmp::Eq,
    collections::{HashMap, HashSet, VecDeque},
    hash::Hash,
    marker::Copy,
    mem::{transmute, MaybeUninit},
    ops::Index,
    time::Instant,
};

use pollster::FutureExt;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn default<T: Default>() -> T {
    Default::default()
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct Vertex {
    pos: [f32; 3],
}

// ... -(vertex shader)> triangle + index.

fn gen_cases() -> Box<CubeMarch> {
    const VERTS: [[i32; 3]; 8] = [
        [-1, -1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1],
    ];

    const EDGES: [[usize; 2]; 12] = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [0, 2],
        [1, 3],
        [4, 6],
        [5, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ];
    //
    //     .4------5
    //   .' |    .'|
    //  6---+--7'  |
    //  |   |  |   |
    //  |  .0--+---1
    //  |.'    | .'
    //  2------3'
    //
    //
    //      z
    //      |
    //      |
    //      |
    //     .0------x
    //   .'
    //  y

    fn make_transform(verts: &[[i32; 3]; 8], mtx: [[i32; 3]; 3]) -> [usize; 8] {
        let v = VERTS.map(|v| {
            let transformed = [
                mtx[0][0] * v[0] + mtx[0][1] * v[1] + mtx[0][2] * v[2],
                mtx[1][0] * v[0] + mtx[1][1] * v[1] + mtx[1][2] * v[2],
                mtx[2][0] * v[0] + mtx[2][1] * v[1] + mtx[2][2] * v[2],
            ];
            verts.indexof(transformed)
        });
        v
    }

    let transforms = [
        [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    ]
    .map(|m| make_transform(&VERTS, m));

    let s01: usize = EDGES.indexof([0, 1]);
    let s23: usize = EDGES.indexof([2, 3]);
    let s45: usize = EDGES.indexof([4, 5]);
    let s67: usize = EDGES.indexof([6, 7]);
    let s02: usize = EDGES.indexof([0, 2]);
    let s13: usize = EDGES.indexof([1, 3]);
    let s46: usize = EDGES.indexof([4, 6]);
    let s57: usize = EDGES.indexof([5, 7]);
    let s04: usize = EDGES.indexof([0, 4]);
    let s15: usize = EDGES.indexof([1, 5]);
    let s26: usize = EDGES.indexof([2, 6]);
    let s37: usize = EDGES.indexof([3, 7]);

    let t0: usize = 1 << 0;
    let t1: usize = 1 << 1;
    let t2: usize = 1 << 2;
    let t3: usize = 1 << 3;
    let t4: usize = 1 << 4;
    let t5: usize = 1 << 5;
    let t6: usize = 1 << 6;
    let t7: usize = 1 << 7;
    // https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/MarchingCubesEdit.svg/525px-MarchingCubesEdit.svg.png
    //     .4------5
    //   .' |    .'|
    //  6---+--7'  |
    //  |   |  |   |
    //  |  .0--+---1
    //  |.'    | .'
    //  2------3'
    //

    // created manually from the cases presented on wikipedia
    #[rustfmt::skip]
    let cases = [
        (0, vec![]),
        (t2, vec![[s02, s23, s26]]),
        (t2|t3, vec![[s02, s13, s26], [s26, s37, s13]]),
        (t2|t7, vec![[s02, s23, s26], [s67, s57, s37]]),
        (t0|t1|t3, vec![[s02, s23, s37], [s02, s04, s37], [s04, s15, s37]]),
        // ----
        (t0|t1|t2|t3, vec![[s04, s15, s26], [s15, s37, s26]]),
        (t6|t0|t1|t3, vec![[s46, s67, s26], [s04, s15, s37], [s37, s04, s02], [s23, s37, s02]]),
        (t4|t7|t2|t1, vec![[s02, s23, s26], [s46, s45, s04], [s67, s57, s37], [s01, s13, s15]]),
        (t4|t1|t0|t2, vec![[s45, s46, s26], [s26, s45, s23], [s45, s15, s23], [s15, s13, s23]]),
        (t4|t0|t1|t3, vec![[s45, s15, s46], [s46, s15, s23], [s46, s23, s02], [s23, s15, s37]]),
        // ----
        (t2|t5, vec![[s45, s15, s57], [s02, s23, s26]]),
        (t2|t3|t5, vec![[s45, s15, s57], [s02, s13, s26], [s13, s26, s37]]),
        (t6|t5|t3, vec![[s46, s67, s26], [s15, s57, s45], [s37, s13, s23]]),
        (t2|t6|t1|t5, vec![[s46, s02, s23], [s46, s67, s23], [s45, s57, s13], [s01, s13, s45]]),
        (t1|t5|t2|t0, vec![[s45, s57, s04], [s26, s23, s04], [s04, s57, s23], [s13, s23, s57]]),
    ];
    let mut front: VecDeque<_> = cases.into_iter().collect();

    let mut found: [_; 256] = std::array::from_fn(|_| None);

    while let Some((i, c)) = front.pop_front() {
        for transform in transforms {
            let mut j = 0;
            for bit in 0..8 {
                let mask = 1 << bit;
                if i & mask != 0 {
                    j |= 1 << transform[bit];
                }
            }
            for j in [j, j ^ 255] {
                if found[j].is_some() || i == j {
                    continue;
                }
                let c: Vec<_> = c
                    .iter()
                    .map(|tri| {
                        tri.map(|edge_vertex| {
                            let mut transformed =
                                EDGES[edge_vertex].map(|vertex| transform[vertex]);
                            transformed.sort();
                            EDGES.indexof(transformed)
                        })
                    })
                    .collect();
                front.push_back((j, c));
            }
        }
        found[i] = Some(c);
    }
    // let found: [Vec<_>; 256] = found.map(|f| {
    //     f.unwrap()
    //         .into_iter()
    //         .map(|v| {
    //             v.map(|i| {
    //                 let [[ax, ay, az], [bx, by, bz]] = EDGES[i].map(|s| VERTS[s].map(|v| v as f32));
    //                 [ax + bx, ay + by, az + bz].map(|e| e * 0.5)
    //             })
    //         })
    //         .collect()
    // });
    let found = found.map(|f| {
        let mut f = f.unwrap();
        f.iter_mut().for_each(|f| f.sort());
        f
    });
    println!("{:?}", &found);
    println!("{:?}", found.clone().map(|f| f.len()));
    let flattened: Vec<_> = found.iter().cloned().flatten().collect();
    println!("{:?}", &flattened);
    println!("{:?}", flattened.len());
    let triangle_to_edge = {
        let mut f: Vec<_> = found.iter().flatten().copied().collect();
        f.sort();
        f.dedup();
        collect_arr(f.into_iter())
    };
    let found = found.map(|f| {
        f.into_iter()
            .map(|f| triangle_to_edge.indexof(f))
            .collect::<Vec<_>>()
    });

    let mut prefix = 0;

    let case_to_offset = array::from_fn(|i| {
        prefix += found.get(i.wrapping_sub(1)).map(|i| i.len()).unwrap_or(0);
        prefix
    });
    let case_to_size = array::from_fn(|i| found[i].len());
    // 732 135
    let offset_to_triangle = collect_arr(found.iter().flat_map(|v| v.iter()).copied());
    // 732 135
    //panic!("{} {}", offset_to_triangle.len(), triangle_to_edge.len());
    Box::new(CubeMarch {
        case_to_offset,
        offset_to_triangle,
        triangle_to_edge,
        edge_to_corner: EDGES,
        corner_to_pos: VERTS,
        case_to_size,
    })
}

fn collect_arr<const N: usize, T>(mut i: impl Iterator<Item = T>) -> [T; N] {
    let arr = std::array::from_fn(|_| i.next().unwrap());
    assert!(i.next().is_none());
    arr
}

/// All the LUTs for cube maching
/// The uniforms denormalize this.
struct CubeMarch {
    // case -> size
    case_to_size: [usize; 256],
    // case -> offset
    case_to_offset: [usize; 257],
    // offsets -> triangle
    offset_to_triangle: [usize; 732],
    // triangle -> 3*edge.
    triangle_to_edge: [[usize; 3]; 135],
    // edge -> 2*vertex.
    edge_to_corner: [[usize; 2]; 12],
    // vertex -> normalized position.
    corner_to_pos: [[i32; 3]; 8],
}
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
struct TriWriteBackUniform {
    /// triangle lives on 3 edges
    /// each edge is on 2 corners
    /// edge0 edge1, edge2
    /// XYZ XYZ XYZ XYZ XYZ XYZ TRIA = 22 significant bits
    offset_to_corners_and_triangle: [u32; 732],
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
            offset_to_corners_and_triangle: case.offset_to_triangle.map(|triangle| {
                let [a, b, c] = case.triangle_to_edge[triangle].map(|edge| {
                    let [a, b] = case.edge_to_corner[edge];
                    (a as u32) | ((b as u32) << 3)
                });
                let triangle = triangle as u32;
                a | (b << 6) | (c << 12) | (triangle << 18)
            }),
            case_to_offset: case.case_to_offset.map(|offset| offset as u32),
            _unused0: 0,
            _unused1: 0,
            _unused2: 0,
        }
    }
}
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

struct CubeMarchUniform {}

struct MetaBuffer {
    buffer: wgpu::Buffer,
    bytes: u64,
    ty: wgpu::BufferBindingType,
}
impl MetaBuffer {
    fn new<T: bytemuck::NoUninit>(device: &wgpu::Device, name: &str, contents: &[T]) -> Self {
        Self::new_i(
            device,
            name,
            wgpu::BufferBindingType::Storage { read_only: false },
            wgpu::BufferUsages::STORAGE,
            bytemuck::cast_slice(contents),
        )
    }
    fn new_constant<T: bytemuck::NoUninit>(
        device: &wgpu::Device,
        name: &str,
        contents: &[T],
    ) -> Self {
        Self::new_i(
            device,
            name,
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::BufferUsages::STORAGE,
            bytemuck::cast_slice(contents),
        )
    }
    fn new_uniform<T: Default + bytemuck::NoUninit>(device: &wgpu::Device) -> Self {
        let data = T::default();
        Self::new_i(
            device,
            &format!("{} uniform buffer", std::any::type_name::<T>()),
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            bytemuck::cast_slice(&[data]),
        )
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
            visibility: wgpu::ShaderStages::COMPUTE,
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
    fn write<T: bytemuck::NoUninit>(&self, queue: wgpu::Queue, data: T) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[data]));
    }
}

#[derive(Default, Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
struct ChunkUniform {
    x: i32,
    y: i32,
    z: i32,
    write_offset: u32,
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
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{name} shader module")),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
        });
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
        pass.dispatch_workgroups(self.x, 0, 0);
    }
}

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
}
impl TriGenState {
    #[rustfmt::skip]
    fn new(device: &wgpu::Device, cube_march: &CubeMarch, triangle_storage: &MetaBuffer) -> Self {
        let chunk = MetaBuffer::new_uniform::<ChunkUniform>(device);
        let sdf_data = MetaBuffer::new(device, "sdf_data", &vec![0.0_f32; 33*33*33 + 415]);
        let triangle_count_prefix = MetaBuffer::new(device, "triangle_count_prefix", &vec![0_u32; 32*32*32 + 1]);


        let sdf = Kernel::new(device, source!("sdf"), &[&chunk, &sdf_data], (33*33*33 + 415)/512);
        let triangle_allocation = Kernel::new(device, source!("triangle_allocation"), &[&sdf_data, &triangle_count_prefix], (32*32*32)/128);
        let prefix_top = Kernel::new(device, source!("prefix_top"), &[&triangle_count_prefix], 1);
        let triangle_writeback = Kernel::new(device, source!("triangle_writeback"), &[&sdf_data, &triangle_count_prefix, &triangle_storage], 32*32*32/128);

        Self { 
            chunk, 
            sdf_data, 
            sdf, 
            triangle_allocation, 
            triangle_count_prefix, 
            prefix_top,
            triangle_writeback,
        }

    }
    fn dispatch<'s: 'c, 'c>(&'s self, pass: &mut wgpu::ComputePass<'c>) {
        self.sdf.dispatch(pass);
        self.triangle_allocation.dispatch(pass);
        self.prefix_top.dispatch(pass);
        self.triangle_writeback.dispatch(pass);
    }
}

trait SearchExt<T: Copy + Eq> {
    fn indexof(self, t: T) -> usize;
}
impl<T: Copy + Eq> SearchExt<T> for &[T] {
    fn indexof(self, t: T) -> usize {
        for (i, v) in self.iter().copied().enumerate() {
            if t == v {
                return i;
            }
        }
        panic!()
    }
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
// plan:
// gen u8 from sdf, calc number of verts per cell.
// prefix sum verts per cell
// draw the sum of verts of vertices?
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

fn cube_march_cpu() -> Vec<[[f32; 3]; 3]> {
    fn sdf(x: usize, y: usize, z: usize) -> f32 {
        let x = (x as f32 / 2.0) % 30.0 - 15.0;
        let y = (y as f32 / 2.0) % 30.0 - 15.0;
        let z = (z as f32 / 2.0) % 30.0 - 15.0;

        (x * x + y * y + z * z).sqrt() - 10.0
    }

    let cases = gen_cases();

    let mut tris: Vec<[[f32; 3]; 3]> = Vec::new();

    let size = 32;

    for x in 0..size {
        for y in 0..size {
            for z in 0..size {
                let mut idx = 0;
                //     .4------5
                //   .' |    .'|
                //  6---+--7'  |
                //  |   |  |   |
                //  |  .0--+---1
                //  |.'    | .'
                //  2------3'
                //
                //      z
                //      |
                //      |
                //      |
                //     .0------x
                //   .'
                //  y
                let scale = size as f32;
                let s = [
                    sdf(x + 0, y + 0, z + 0).clamp(-0.5, 0.5),
                    sdf(x + 1, y + 0, z + 0).clamp(-0.5, 0.5),
                    sdf(x + 0, y + 1, z + 0).clamp(-0.5, 0.5),
                    sdf(x + 1, y + 1, z + 0).clamp(-0.5, 0.5),
                    sdf(x + 0, y + 0, z + 1).clamp(-0.5, 0.5),
                    sdf(x + 1, y + 0, z + 1).clamp(-0.5, 0.5),
                    sdf(x + 0, y + 1, z + 1).clamp(-0.5, 0.5),
                    sdf(x + 1, y + 1, z + 1).clamp(-0.5, 0.5),
                ];
                for (i, &s) in s.iter().enumerate() {
                    idx |= ((s > 0.0) as u8) << i;
                }

                let x = (x as f32) * 2.0 - (size - 1) as f32;
                let y = (y as f32) * 2.0 - (size - 1) as f32;
                let z = (z as f32) * 2.0 - (size - 1) as f32;
                // tris.push(
                //     [[0.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0]].map(|[px, py, pz]| {
                //         [px + x, py + y, pz + z].map(|e| e * (1.0 / size as f32))
                //     }),
                // );
                let count =
                    cases.case_to_offset[idx as usize + 1] - cases.case_to_offset[idx as usize];
                let offset = cases.case_to_offset[idx as usize];
                tris.extend(
                    (0..count)
                        .map(|data_idx| {
                            cases.triangle_to_edge[cases.offset_to_triangle[offset + data_idx]]
                        })
                        .map(|tri| {
                            tri.map(|i| {
                                let [(sa, [ax, ay, az]), (sb, [bx, by, bz])] =
                                    cases.edge_to_corner[i].map(|vertex| {
                                        (s[vertex], cases.corner_to_pos[vertex].map(|v| v as f32))
                                    });
                                let sa = sa.abs();
                                let sb = sb.abs();
                                let sa = sa / (sa + sb);
                                let sb = 1.0 - sa;
                                let (sa, sb) = (2.0 * sb, 2.0 * sa);
                                [sa * ax + sb * bx, sa * ay + sb * by, sa * az + sb * bz]
                                    .map(|e| e * 0.5)
                            })
                            .map(|[px, py, pz]| [x + px, y + py, z + pz].map(|e| e / size as f32))
                        }),
                )
            }
        }
    }
    dbg!(tris.len());
    tris
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}
impl CameraUniform {
    fn from_camera(camera: &Camera) -> Self {
        Self {
            view_proj: camera.view_projection_matrix().into(),
        }
    }
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

struct State<'a> {
    device: wgpu::Device,
    queue: wgpu::Queue,
    instance: wgpu::Instance,
    surface: wgpu::Surface<'a>,
    surface_config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    depth_texture: Texture,
    camera_bind_group: wgpu::BindGroup,
    camera: Camera,
    vertex_buffer: wgpu::Buffer,
    t0: Instant,
    num_vert: u32,
    camera_buffer: wgpu::Buffer,
    last_time: Instant,
    last_count: u32,
}
impl<'a> State<'a> {
    fn new(window: &'a winit::window::Window) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER
                | wgpu::InstanceFlags::DEBUG
                | wgpu::InstanceFlags::VALIDATION,
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
        let surface_caps = dbg!(surface.get_capabilities(&adapter));
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: 200,
            height: 200,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: dbg!(&surface_caps.alpha_modes)[0],
            view_formats: vec![],
        };

        dbg!(adapter.limits());
        dbg!(adapter.features());
        dbg!(adapter.get_info());

        let (device, queue) = match adapter.request_device(&default(), None).block_on() {
            Ok(o) => o,
            Err(e) => panic!("{}", e),
        };
        surface.configure(&device, &surface_config);

        let shader_source = include_str!("shader.wgsl");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let vert: Vec<_> = cube_march_cpu().into_iter().flatten().collect(); // VERT;
        let num_vert = vert.len() as u32;

        let _ = indexify(&vert);

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as _,
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
            contents: bytemuck::cast_slice(&[CameraUniform::from_camera(&camera)]),
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
        }
    }
    fn render(&mut self) {
        let output = self.surface.get_current_texture().unwrap();
        let view = &output.texture.create_view(&default());
        let mut encoder = self.device.create_command_encoder(&default());
        {
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
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.draw(0..self.num_vert, 0..1);
        }
        self.camera = Camera::new(&self.surface_config);
        let t = self.t0.elapsed().as_secs_f32();
        self.camera.eye = (
            2.0 * t.sin(),
            (t * 2.0_f32.sqrt() * 0.5).sin() * 0.1,
            2.0 * t.cos(),
        )
            .into();
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[CameraUniform::from_camera(&self.camera)]),
        );
        self.queue.submit([encoder.finish()]);
        output.present();
        let elapsed = self.last_time.elapsed().as_secs_f64();
        self.last_count += 1;
        if elapsed > 1.0 {
            let fps = self.last_count as f64 / elapsed;
            println!("FPS: {fps}");

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

    const VERT: &[Vertex] = &[
        Vertex {
            pos: [0.0, 0.5, 0.0],
        },
        Vertex {
            pos: [-0.5, -0.5, 0.0],
        },
        Vertex {
            pos: [0.5, -0.5, 0.0],
        },
        Vertex {
            pos: [0.5, 0.5, 0.0],
        },
        Vertex {
            pos: [1.0, -0.5, 0.0],
        },
        Vertex {
            pos: [0.0, -0.5, 0.0],
        },
    ];

    let vert = VERT;
    let mut state = State::new(window);

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
