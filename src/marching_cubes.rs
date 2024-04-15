use std::{array, collections::VecDeque};
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
    // println!("{:?}", &found);
    // println!("{:?}", found.clone().map(|f| f.len()));
    let flattened: Vec<_> = found.iter().cloned().flatten().collect();
    // println!("{:?}", &flattened);
    // println!("{:?}", flattened.len());
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

/// All the LUTs for cube maching.
/// The uniforms denormalize this.
pub(crate) struct CubeMarch {
    // case -> size
    pub(crate) case_to_size: [usize; 256],
    // case -> offset
    pub(crate) case_to_offset: [usize; 257],
    // offsets -> triangle
    pub(crate) offset_to_triangle: [usize; 732],
    // triangle -> 3*edge.
    pub(crate) triangle_to_edge: [[usize; 3]; 135],
    // edge -> 2*vertex.
    pub(crate) edge_to_corner: [[usize; 2]; 12],
    // vertex -> normalized position.
    pub(crate) corner_to_pos: [[i32; 3]; 8],
}
impl CubeMarch {
    pub(crate) fn new() -> Box<CubeMarch> {
        gen_cases()
    }
}
pub(crate) fn cube_march_cpu() -> Vec<[[f32; 3]; 3]> {
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

