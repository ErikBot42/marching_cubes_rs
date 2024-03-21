use std::{
    cmp::Eq,
    collections::{HashMap, VecDeque},
    hash::Hash,
    marker::Copy,
    mem::{transmute, MaybeUninit},
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
const fn arr_len<T: Copy, const N: usize>(a: [T; N]) -> usize {
    N
}

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


fn gen_cases() {
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
            verts.index(transformed)
        });
        v
    }

    //      z
    //      |
    //      |
    //      |
    //     .0------x
    //   .'
    //  y
    let transforms = [
        [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    ]
    .map(|m| make_transform(&VERTS, m));

    let edges: [[usize; 2]; 12] = [
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

    let s01: usize = 0;
    let s23: usize = 1;
    let s45: usize = 2;
    let s67: usize = 3;
    let s02: usize = 4;
    let s13: usize = 5;
    let s46: usize = 6;
    let s57: usize = 7;
    let s04: usize = 8;
    let s15: usize = 9;
    let s26: usize = 10;
    let s37: usize = 11;

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
        let cases = vec![
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

    let mut found: [_; 256] = std::array::from_fn(|_| None);
    let mut front = VecDeque::new();

    for (i, c) in cases {
        front.push_back((i, c));
    }

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
                                edges[edge_vertex].map(|vertex| transform[vertex]);
                            transformed.sort();
                            edges.index(transformed)
                        })
                    })
                    .collect();
                front.push_back((j, c));
            }
        }
        found[i] = Some(c);
    }
    dbg!(&found);
    let f: Vec<_> = found
        .iter()
        .enumerate()
        .filter_map(|(i, f)| f.is_none().then_some(i))
        .collect();
    dbg!(&f);
}

trait SearchExt<T: Copy + Eq> {
    fn index(self, t: T) -> usize;
}
impl<T: Copy + Eq> SearchExt<T> for &[T] {
    fn index(self, t: T) -> usize {
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

fn cube_march_cpu() {
    gen_cases();

    panic!();
    let size = 100;
    let cases = [
        vec![],
        vec![[8, 0, 3]],
        vec![[1, 0, 9]],
        vec![[8, 1, 3], [8, 9, 1]],
        vec![[10, 2, 1]],
        vec![[8, 0, 3], [1, 10, 2]],
        vec![[9, 2, 0], [9, 10, 2]],
        vec![[3, 8, 2], [2, 8, 10], [10, 8, 9]],
        vec![[3, 2, 11]],
        vec![[0, 2, 8], [2, 11, 8]],
        vec![[1, 0, 9], [2, 11, 3]],
        vec![[2, 9, 1], [11, 9, 2], [8, 9, 11]],
        vec![[3, 10, 11], [3, 1, 10]],
        vec![[1, 10, 0], [0, 10, 8], [8, 10, 11]],
        vec![[0, 11, 3], [9, 11, 0], [10, 11, 9]],
        vec![[8, 9, 11], [11, 9, 10]],
        vec![[7, 4, 8]],
        vec![[3, 7, 0], [7, 4, 0]],
        vec![[7, 4, 8], [9, 1, 0]],
        vec![[9, 1, 4], [4, 1, 7], [7, 1, 3]],
        vec![[7, 4, 8], [2, 1, 10]],
        vec![[4, 3, 7], [4, 0, 3], [2, 1, 10]],
        vec![[2, 0, 10], [0, 9, 10], [7, 4, 8]],
        vec![[9, 10, 4], [4, 10, 3], [3, 10, 2], [4, 3, 7]],
        vec![[4, 8, 7], [3, 2, 11]],
        vec![[7, 4, 11], [11, 4, 2], [2, 4, 0]],
        vec![[1, 0, 9], [2, 11, 3], [8, 7, 4]],
        vec![[2, 11, 1], [1, 11, 9], [9, 11, 7], [9, 7, 4]],
        vec![[10, 11, 1], [11, 3, 1], [4, 8, 7]],
        vec![[4, 0, 7], [7, 0, 10], [0, 1, 10], [7, 10, 11]],
        vec![[7, 4, 8], [0, 11, 3], [9, 11, 0], [10, 11, 9]],
        vec![[4, 11, 7], [9, 11, 4], [10, 11, 9]],
        vec![[9, 4, 5]],
        vec![[9, 4, 5], [0, 3, 8]],
        vec![[0, 5, 1], [0, 4, 5]],
        vec![[4, 3, 8], [5, 3, 4], [1, 3, 5]],
        vec![[5, 9, 4], [10, 2, 1]],
        vec![[8, 0, 3], [1, 10, 2], [4, 5, 9]],
        vec![[10, 4, 5], [2, 4, 10], [0, 4, 2]],
        vec![[3, 10, 2], [8, 10, 3], [5, 10, 8], [4, 5, 8]],
        vec![[9, 4, 5], [11, 3, 2]],
        vec![[11, 0, 2], [11, 8, 0], [9, 4, 5]],
        vec![[5, 1, 4], [1, 0, 4], [11, 3, 2]],
        vec![[5, 1, 4], [4, 1, 11], [1, 2, 11], [4, 11, 8]],
        vec![[3, 10, 11], [3, 1, 10], [5, 9, 4]],
        vec![[9, 4, 5], [1, 10, 0], [0, 10, 8], [8, 10, 11]],
        vec![[5, 0, 4], [11, 0, 5], [11, 3, 0], [10, 11, 5]],
        vec![[5, 10, 4], [4, 10, 8], [8, 10, 11]],
        vec![[9, 7, 5], [9, 8, 7]],
        vec![[0, 5, 9], [3, 5, 0], [7, 5, 3]],
        vec![[8, 7, 0], [0, 7, 1], [1, 7, 5]],
        vec![[7, 5, 3], [3, 5, 1]],
        vec![[7, 5, 8], [5, 9, 8], [2, 1, 10]],
        vec![[10, 2, 1], [0, 5, 9], [3, 5, 0], [7, 5, 3]],
        vec![[8, 2, 0], [5, 2, 8], [10, 2, 5], [7, 5, 8]],
        vec![[2, 3, 10], [10, 3, 5], [5, 3, 7]],
        vec![[9, 7, 5], [9, 8, 7], [11, 3, 2]],
        vec![[0, 2, 9], [9, 2, 7], [7, 2, 11], [9, 7, 5]],
        vec![[3, 2, 11], [8, 7, 0], [0, 7, 1], [1, 7, 5]],
        vec![[11, 1, 2], [7, 1, 11], [5, 1, 7]],
        vec![[3, 1, 11], [11, 1, 10], [8, 7, 9], [9, 7, 5]],
        vec![[11, 7, 0], [7, 5, 0], [5, 9, 0], [10, 11, 0], [1, 10, 0]],
        vec![[0, 5, 10], [0, 7, 5], [0, 8, 7], [0, 10, 11], [0, 11, 3]],
        vec![[10, 11, 5], [11, 7, 5]],
        vec![[5, 6, 10]],
        vec![[8, 0, 3], [10, 5, 6]],
        vec![[0, 9, 1], [5, 6, 10]],
        vec![[8, 1, 3], [8, 9, 1], [10, 5, 6]],
        vec![[1, 6, 2], [1, 5, 6]],
        vec![[6, 2, 5], [2, 1, 5], [8, 0, 3]],
        vec![[5, 6, 9], [9, 6, 0], [0, 6, 2]],
        vec![[5, 8, 9], [2, 8, 5], [3, 8, 2], [6, 2, 5]],
        vec![[3, 2, 11], [10, 5, 6]],
        vec![[0, 2, 8], [2, 11, 8], [5, 6, 10]],
        vec![[3, 2, 11], [0, 9, 1], [10, 5, 6]],
        vec![[5, 6, 10], [2, 9, 1], [11, 9, 2], [8, 9, 11]],
        vec![[11, 3, 6], [6, 3, 5], [5, 3, 1]],
        vec![[11, 8, 6], [6, 8, 1], [1, 8, 0], [6, 1, 5]],
        vec![[5, 0, 9], [6, 0, 5], [3, 0, 6], [11, 3, 6]],
        vec![[6, 9, 5], [11, 9, 6], [8, 9, 11]],
        vec![[7, 4, 8], [6, 10, 5]],
        vec![[3, 7, 0], [7, 4, 0], [10, 5, 6]],
        vec![[7, 4, 8], [6, 10, 5], [9, 1, 0]],
        vec![[5, 6, 10], [9, 1, 4], [4, 1, 7], [7, 1, 3]],
        vec![[1, 6, 2], [1, 5, 6], [7, 4, 8]],
        vec![[6, 1, 5], [2, 1, 6], [0, 7, 4], [3, 7, 0]],
        vec![[4, 8, 7], [5, 6, 9], [9, 6, 0], [0, 6, 2]],
        vec![[2, 3, 9], [3, 7, 9], [7, 4, 9], [6, 2, 9], [5, 6, 9]],
        vec![[2, 11, 3], [7, 4, 8], [10, 5, 6]],
        vec![[6, 10, 5], [7, 4, 11], [11, 4, 2], [2, 4, 0]],
        vec![[1, 0, 9], [8, 7, 4], [3, 2, 11], [5, 6, 10]],
        vec![[1, 2, 9], [9, 2, 11], [9, 11, 4], [4, 11, 7], [5, 6, 10]],
        vec![[7, 4, 8], [11, 3, 6], [6, 3, 5], [5, 3, 1]],
        vec![[11, 0, 1], [11, 4, 0], [11, 7, 4], [11, 1, 5], [11, 5, 6]],
        vec![[6, 9, 5], [0, 9, 6], [11, 0, 6], [3, 0, 11], [4, 8, 7]],
        vec![[5, 6, 9], [9, 6, 11], [9, 11, 7], [9, 7, 4]],
        vec![[4, 10, 9], [4, 6, 10]],
        vec![[10, 4, 6], [10, 9, 4], [8, 0, 3]],
        vec![[1, 0, 10], [10, 0, 6], [6, 0, 4]],
        vec![[8, 1, 3], [6, 1, 8], [6, 10, 1], [4, 6, 8]],
        vec![[9, 2, 1], [4, 2, 9], [6, 2, 4]],
        vec![[3, 8, 0], [9, 2, 1], [4, 2, 9], [6, 2, 4]],
        vec![[0, 4, 2], [2, 4, 6]],
        vec![[8, 2, 3], [4, 2, 8], [6, 2, 4]],
        vec![[4, 10, 9], [4, 6, 10], [2, 11, 3]],
        vec![[11, 8, 2], [2, 8, 0], [6, 10, 4], [4, 10, 9]],
        vec![[2, 11, 3], [1, 0, 10], [10, 0, 6], [6, 0, 4]],
        vec![[8, 4, 1], [4, 6, 1], [6, 10, 1], [11, 8, 1], [2, 11, 1]],
        vec![[3, 1, 11], [11, 1, 4], [1, 9, 4], [11, 4, 6]],
        vec![[6, 11, 1], [11, 8, 1], [8, 0, 1], [4, 6, 1], [9, 4, 1]],
        vec![[3, 0, 11], [11, 0, 6], [6, 0, 4]],
        vec![[4, 11, 8], [4, 6, 11]],
        vec![[6, 8, 7], [10, 8, 6], [9, 8, 10]],
        vec![[3, 7, 0], [0, 7, 10], [7, 6, 10], [0, 10, 9]],
        vec![[1, 6, 10], [0, 6, 1], [7, 6, 0], [8, 7, 0]],
        vec![[10, 1, 6], [6, 1, 7], [7, 1, 3]],
        vec![[9, 8, 1], [1, 8, 6], [6, 8, 7], [1, 6, 2]],
        vec![[9, 7, 6], [9, 3, 7], [9, 0, 3], [9, 6, 2], [9, 2, 1]],
        vec![[7, 6, 8], [8, 6, 0], [0, 6, 2]],
        vec![[3, 6, 2], [3, 7, 6]],
        vec![[3, 2, 11], [6, 8, 7], [10, 8, 6], [9, 8, 10]],
        vec![[7, 9, 0], [7, 10, 9], [7, 6, 10], [7, 0, 2], [7, 2, 11]],
        vec![[0, 10, 1], [6, 10, 0], [8, 6, 0], [7, 6, 8], [2, 11, 3]],
        vec![[1, 6, 10], [7, 6, 1], [11, 7, 1], [2, 11, 1]],
        vec![[1, 9, 6], [9, 8, 6], [8, 7, 6], [3, 1, 6], [11, 3, 6]],
        vec![[9, 0, 1], [11, 7, 6]],
        vec![[0, 11, 3], [6, 11, 0], [7, 6, 0], [8, 7, 0]],
        vec![[7, 6, 11]],
        vec![[11, 6, 7]],
        vec![[3, 8, 0], [11, 6, 7]],
        vec![[1, 0, 9], [6, 7, 11]],
        vec![[1, 3, 9], [3, 8, 9], [6, 7, 11]],
        vec![[10, 2, 1], [6, 7, 11]],
        vec![[10, 2, 1], [3, 8, 0], [6, 7, 11]],
        vec![[9, 2, 0], [9, 10, 2], [11, 6, 7]],
        vec![[11, 6, 7], [3, 8, 2], [2, 8, 10], [10, 8, 9]],
        vec![[2, 6, 3], [6, 7, 3]],
        vec![[8, 6, 7], [0, 6, 8], [2, 6, 0]],
        vec![[7, 2, 6], [7, 3, 2], [1, 0, 9]],
        vec![[8, 9, 7], [7, 9, 2], [2, 9, 1], [7, 2, 6]],
        vec![[6, 1, 10], [7, 1, 6], [3, 1, 7]],
        vec![[8, 0, 7], [7, 0, 6], [6, 0, 1], [6, 1, 10]],
        vec![[7, 3, 6], [6, 3, 9], [3, 0, 9], [6, 9, 10]],
        vec![[7, 8, 6], [6, 8, 10], [10, 8, 9]],
        vec![[8, 11, 4], [11, 6, 4]],
        vec![[11, 0, 3], [6, 0, 11], [4, 0, 6]],
        vec![[6, 4, 11], [4, 8, 11], [1, 0, 9]],
        vec![[1, 3, 9], [9, 3, 6], [3, 11, 6], [9, 6, 4]],
        vec![[8, 11, 4], [11, 6, 4], [1, 10, 2]],
        vec![[1, 10, 2], [11, 0, 3], [6, 0, 11], [4, 0, 6]],
        vec![[2, 9, 10], [0, 9, 2], [4, 11, 6], [8, 11, 4]],
        vec![[3, 4, 9], [3, 6, 4], [3, 11, 6], [3, 9, 10], [3, 10, 2]],
        vec![[3, 2, 8], [8, 2, 4], [4, 2, 6]],
        vec![[2, 4, 0], [6, 4, 2]],
        vec![[0, 9, 1], [3, 2, 8], [8, 2, 4], [4, 2, 6]],
        vec![[1, 2, 9], [9, 2, 4], [4, 2, 6]],
        vec![[10, 3, 1], [4, 3, 10], [4, 8, 3], [6, 4, 10]],
        vec![[10, 0, 1], [6, 0, 10], [4, 0, 6]],
        vec![[3, 10, 6], [3, 9, 10], [3, 0, 9], [3, 6, 4], [3, 4, 8]],
        vec![[9, 10, 4], [10, 6, 4]],
        vec![[9, 4, 5], [7, 11, 6]],
        vec![[9, 4, 5], [7, 11, 6], [0, 3, 8]],
        vec![[0, 5, 1], [0, 4, 5], [6, 7, 11]],
        vec![[11, 6, 7], [4, 3, 8], [5, 3, 4], [1, 3, 5]],
        vec![[1, 10, 2], [9, 4, 5], [6, 7, 11]],
        vec![[8, 0, 3], [4, 5, 9], [10, 2, 1], [11, 6, 7]],
        vec![[7, 11, 6], [10, 4, 5], [2, 4, 10], [0, 4, 2]],
        vec![[8, 2, 3], [10, 2, 8], [4, 10, 8], [5, 10, 4], [11, 6, 7]],
        vec![[2, 6, 3], [6, 7, 3], [9, 4, 5]],
        vec![[5, 9, 4], [8, 6, 7], [0, 6, 8], [2, 6, 0]],
        vec![[7, 3, 6], [6, 3, 2], [4, 5, 0], [0, 5, 1]],
        vec![[8, 1, 2], [8, 5, 1], [8, 4, 5], [8, 2, 6], [8, 6, 7]],
        vec![[9, 4, 5], [6, 1, 10], [7, 1, 6], [3, 1, 7]],
        vec![[7, 8, 6], [6, 8, 0], [6, 0, 10], [10, 0, 1], [5, 9, 4]],
        vec![[3, 0, 10], [0, 4, 10], [4, 5, 10], [7, 3, 10], [6, 7, 10]],
        vec![[8, 6, 7], [10, 6, 8], [5, 10, 8], [4, 5, 8]],
        vec![[5, 9, 6], [6, 9, 11], [11, 9, 8]],
        vec![[11, 6, 3], [3, 6, 0], [0, 6, 5], [0, 5, 9]],
        vec![[8, 11, 0], [0, 11, 5], [5, 11, 6], [0, 5, 1]],
        vec![[6, 3, 11], [5, 3, 6], [1, 3, 5]],
        vec![[10, 2, 1], [5, 9, 6], [6, 9, 11], [11, 9, 8]],
        vec![[3, 11, 0], [0, 11, 6], [0, 6, 9], [9, 6, 5], [1, 10, 2]],
        vec![[0, 8, 5], [8, 11, 5], [11, 6, 5], [2, 0, 5], [10, 2, 5]],
        vec![[11, 6, 3], [3, 6, 5], [3, 5, 10], [3, 10, 2]],
        vec![[3, 9, 8], [6, 9, 3], [5, 9, 6], [2, 6, 3]],
        vec![[9, 6, 5], [0, 6, 9], [2, 6, 0]],
        vec![[6, 5, 8], [5, 1, 8], [1, 0, 8], [2, 6, 8], [3, 2, 8]],
        vec![[2, 6, 1], [6, 5, 1]],
        vec![[6, 8, 3], [6, 9, 8], [6, 5, 9], [6, 3, 1], [6, 1, 10]],
        vec![[1, 10, 0], [0, 10, 6], [0, 6, 5], [0, 5, 9]],
        vec![[3, 0, 8], [6, 5, 10]],
        vec![[10, 6, 5]],
        vec![[5, 11, 10], [5, 7, 11]],
        vec![[5, 11, 10], [5, 7, 11], [3, 8, 0]],
        vec![[11, 10, 7], [10, 5, 7], [0, 9, 1]],
        vec![[5, 7, 10], [10, 7, 11], [9, 1, 8], [8, 1, 3]],
        vec![[2, 1, 11], [11, 1, 7], [7, 1, 5]],
        vec![[3, 8, 0], [2, 1, 11], [11, 1, 7], [7, 1, 5]],
        vec![[2, 0, 11], [11, 0, 5], [5, 0, 9], [11, 5, 7]],
        vec![[2, 9, 5], [2, 8, 9], [2, 3, 8], [2, 5, 7], [2, 7, 11]],
        vec![[10, 3, 2], [5, 3, 10], [7, 3, 5]],
        vec![[10, 0, 2], [7, 0, 10], [8, 0, 7], [5, 7, 10]],
        vec![[0, 9, 1], [10, 3, 2], [5, 3, 10], [7, 3, 5]],
        vec![[7, 8, 2], [8, 9, 2], [9, 1, 2], [5, 7, 2], [10, 5, 2]],
        vec![[3, 1, 7], [7, 1, 5]],
        vec![[0, 7, 8], [1, 7, 0], [5, 7, 1]],
        vec![[9, 5, 0], [0, 5, 3], [3, 5, 7]],
        vec![[5, 7, 9], [7, 8, 9]],
        vec![[4, 10, 5], [8, 10, 4], [11, 10, 8]],
        vec![[3, 4, 0], [10, 4, 3], [10, 5, 4], [11, 10, 3]],
        vec![[1, 0, 9], [4, 10, 5], [8, 10, 4], [11, 10, 8]],
        vec![[4, 3, 11], [4, 1, 3], [4, 9, 1], [4, 11, 10], [4, 10, 5]],
        vec![[1, 5, 2], [2, 5, 8], [5, 4, 8], [2, 8, 11]],
        vec![[5, 4, 11], [4, 0, 11], [0, 3, 11], [1, 5, 11], [2, 1, 11]],
        vec![[5, 11, 2], [5, 8, 11], [5, 4, 8], [5, 2, 0], [5, 0, 9]],
        vec![[5, 4, 9], [2, 3, 11]],
        vec![[3, 4, 8], [2, 4, 3], [5, 4, 2], [10, 5, 2]],
        vec![[5, 4, 10], [10, 4, 2], [2, 4, 0]],
        vec![[2, 8, 3], [4, 8, 2], [10, 4, 2], [5, 4, 10], [0, 9, 1]],
        vec![[4, 10, 5], [2, 10, 4], [1, 2, 4], [9, 1, 4]],
        vec![[8, 3, 4], [4, 3, 5], [5, 3, 1]],
        vec![[1, 5, 0], [5, 4, 0]],
        vec![[5, 0, 9], [3, 0, 5], [8, 3, 5], [4, 8, 5]],
        vec![[5, 4, 9]],
        vec![[7, 11, 4], [4, 11, 9], [9, 11, 10]],
        vec![[8, 0, 3], [7, 11, 4], [4, 11, 9], [9, 11, 10]],
        vec![[0, 4, 1], [1, 4, 11], [4, 7, 11], [1, 11, 10]],
        vec![[10, 1, 4], [1, 3, 4], [3, 8, 4], [11, 10, 4], [7, 11, 4]],
        vec![[9, 4, 1], [1, 4, 2], [2, 4, 7], [2, 7, 11]],
        vec![[1, 9, 2], [2, 9, 4], [2, 4, 11], [11, 4, 7], [3, 8, 0]],
        vec![[11, 4, 7], [2, 4, 11], [0, 4, 2]],
        vec![[7, 11, 4], [4, 11, 2], [4, 2, 3], [4, 3, 8]],
        vec![[10, 9, 2], [2, 9, 7], [7, 9, 4], [2, 7, 3]],
        vec![[2, 10, 7], [10, 9, 7], [9, 4, 7], [0, 2, 7], [8, 0, 7]],
        vec![[10, 4, 7], [10, 0, 4], [10, 1, 0], [10, 7, 3], [10, 3, 2]],
        vec![[8, 4, 7], [10, 1, 2]],
        vec![[4, 1, 9], [7, 1, 4], [3, 1, 7]],
        vec![[8, 0, 7], [7, 0, 1], [7, 1, 9], [7, 9, 4]],
        vec![[0, 7, 3], [0, 4, 7]],
        vec![[8, 4, 7]],
        vec![[9, 8, 10], [10, 8, 11]],
        vec![[3, 11, 0], [0, 11, 9], [9, 11, 10]],
        vec![[0, 10, 1], [8, 10, 0], [11, 10, 8]],
        vec![[11, 10, 3], [10, 1, 3]],
        vec![[1, 9, 2], [2, 9, 11], [11, 9, 8]],
        vec![[9, 2, 1], [11, 2, 9], [3, 11, 9], [0, 3, 9]],
        vec![[8, 2, 0], [8, 11, 2]],
        vec![[11, 2, 3]],
        vec![[2, 8, 3], [10, 8, 2], [9, 8, 10]],
        vec![[0, 2, 9], [2, 10, 9]],
        vec![[3, 2, 8], [8, 2, 10], [8, 10, 1], [8, 1, 0]],
        vec![[1, 2, 10]],
        vec![[3, 1, 8], [1, 9, 8]],
        vec![[9, 0, 1]],
        vec![[3, 0, 8]],
        vec![],
    ];
    let vertices = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ];

    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];

    fn sdf(x: usize, y: usize, z: usize) -> u8 {
        let x = x as f32 / 2.0;
        let y = y as f32 / 2.0;
        let z = z as f32 / 2.0;

        (x.sin() * y.cos() * z.sin().cos() > 0.0) as u8
    }
    let mut map: HashMap<u8, usize> = HashMap::new();

    for x in 0..size {
        for y in 0..size {
            for z in 0..size {
                let mut idx = 0;
                idx |= sdf(x + 0, y + 0, z + 0) << 0;
                idx |= sdf(x + 0, y + 0, z + 1) << 1;
                idx |= sdf(x + 0, y + 1, z + 0) << 2;
                idx |= sdf(x + 0, y + 1, z + 1) << 3;
                idx |= sdf(x + 1, y + 0, z + 0) << 4;
                idx |= sdf(x + 1, y + 0, z + 1) << 5;
                idx |= sdf(x + 1, y + 1, z + 0) << 6;
                idx |= sdf(x + 1, y + 1, z + 1) << 7;
                *map.entry(idx).or_default() += 1;

                // for face in &cases[idx] {

                // }
            }
        }
    }
    println!("{map:?}");
}

fn main() {
    env_logger::init();
    std::env::set_var("RUST_BACKTRACE", "1");
    cube_march_cpu();
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    let window = &window;

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

    // instance.request_adapter(&wgpu::RequestAdapterOptions {
    //     power_preference: todo!(),
    //     force_fallback_adapter: false,
    //     compatible_surface: todo!(),
    // });

    let surface = instance.create_surface(window).unwrap();
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
    let mut surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: 200,
        height: 200,
        present_mode: dbg!(&surface_caps.present_modes)[0],
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

    // device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //     label: None,
    //     contents: todo!(),
    //     usage: todo!(),
    // });
    // let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
    //     label: None,
    //     layout: Some(&render_pipeline_layout),
    //     vertex: todo!(),
    //     primitive: todo!(),
    //     depth_stencil: todo!(),
    //     multisample: todo!(),
    //     fragment: todo!(),
    //     multiview: todo!(),
    // });

    let shader_source = "
    struct VertexInput {
        @location(0) position: vec3<f32>,
        @builtin(vertex_index) in_vertex_index: u32,
    };
    struct VertexOutput {
        @builtin(position) clip_position: vec4<f32>,
        // @location(0) color: vec3<f32>,
    };
    
    @vertex
    fn vs_main(
        model: VertexInput,
    ) -> VertexOutput {
        var out: VertexOutput;
        out.clip_position = vec4<f32>(model.position, 1.0);
        return out;
    }
    
    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
        return vec4<f32>(0.3, 0.2, 0.1, 1.0);
    }
    
    ";

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

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
    ];

    let vertex_buffer_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as _,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
    };

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(VERT),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

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
        depth_stencil: None,
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
                            let output = surface.get_current_texture().unwrap();
                            let view = &output.texture.create_view(&default());
                            let mut encoder = device.create_command_encoder(&default());
                            {
                                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: None,
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view,
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
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                                render_pass.set_pipeline(&render_pipeline);
                                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                                render_pass.draw(0..3, 0..1);
                            }
                            queue.submit([encoder.finish()]);
                            output.present();
                        }
                        WindowEvent::Resized(size) => {
                            if size.width > 0 && size.height > 0 {
                                surface_config.width = size.width;
                                surface_config.height = size.height;
                                surface.configure(&device, &surface_config);
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
