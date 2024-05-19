use std::{
    array,
    collections::{HashMap, VecDeque},
    fmt::Debug,
    mem::replace,
};

#[allow(unused)]
use cgmath::{
    dot, AbsDiff, AbsDiffEq, Angle, Array, BaseFloat, BaseNum, Basis2, Basis3, Bounded, Decomposed,
    Deg, ElementWise, EuclideanSpace, Euler, InnerSpace, Matrix, Matrix2, Matrix3, Matrix4,
    MetricSpace, One, Ortho, Perspective, PerspectiveFov, Point1, Point2, Point3, Quaternion, Rad,
    Relative, RelativeEq, Rotation, Rotation2, Rotation3, SquareMatrix, Transform, Transform2,
    Transform3, Ulps, UlpsEq, Vector1, Vector2, Vector3, Vector4, VectorSpace, Zero,
};

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

    fn make_transform((mtx, flip): ([[i32; 3]; 3], bool)) -> ([usize; 8], bool) {
        let v = VERTS.map(|v| {
            let transformed = [
                mtx[0][0] * v[0] + mtx[0][1] * v[1] + mtx[0][2] * v[2],
                mtx[1][0] * v[0] + mtx[1][1] * v[1] + mtx[1][2] * v[2],
                mtx[2][0] * v[0] + mtx[2][1] * v[1] + mtx[2][2] * v[2],
            ];
            VERTS.indexof(transformed)
        });
        (v, flip)
    }

    let transforms = [
        ([[1, 0, 0], [0, 0, -1], [0, 1, 0]], false), /* rotate x */
        ([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], false), /* rotate y */
        ([[0, -1, 0], [1, 0, 0], [0, 0, 1]], false), /* rotate z */
        ([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], true),  /* flip x   */
        ([[1, 0, 0], [0, -1, 0], [0, 0, 1]], true),  /* flip y   */
        ([[1, 0, 0], [0, 1, 0], [0, 0, -1]], true),  /* flip z   */
    ]
    .map(make_transform);

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
        (0, 0, vec![]),
        (1, t2, vec![[s02, s26, s23]]/**/),
        (2, t2|t3, vec![[s02, s26, s13], [s26, s13, s37]]),

        (3, t2|t7, vec![[s02, s26, s23]/**/, [s67, s57, s37]]/**/),
        (4, t0|t1|t3, vec![[s02, s23, s37], [s02, s04, s37], [s04, s15, s37]]),
        //---
        (5, t0|t1|t2|t3, vec![[s04, s15, s26], [s15, s37, s26]]),
        (6, t6|t0|t1|t3, vec![[s46, s67, s26], [s04, s15, s37], [s37, s04, s02], [s23, s37, s02]]),
        (7, t4|t7|t2|t1, vec![[s02, s26, s23]/**/, [s46, s45, s04], [s67, s57, s37]/**/, [s01, s13, s15]]),
        (8, t4|t1|t0|t2, vec![[s45, s46, s26], [s26, s45, s23], [s45, s15, s23], [s15, s13, s23]]),
        (9, t4|t0|t1|t3, vec![[s45, s15, s46], [s46, s15, s23], [s46, s23, s02], [s23, s15, s37]]),
        //---
        (10, t2|t5, vec![[s45, s15, s57], [s02, s26, s23]/**/]),
        (11, t2|t3|t5, vec![[s45, s15, s57], [s02, s13, s26], [s13, s26, s37]]),
        (12, t6|t5|t3, vec![[s46, s67, s26], [s15, s57, s45], [s37, s13, s23]]),
        (13, t2|t6|t1|t5, vec![[s46, s02, s23], [s46, s67, s23], [s45, s57, s13], [s01, s13, s45]]),
        (14, t1|t5|t2|t0, vec![[s45, s57, s04], [s26, s23, s04], [s04, s57, s23], [s13, s23, s57]]),
        // ----
        // https://www.boristhebrave.com/2018/04/15/marching-cubes-3d-tutorial/
        (15, t6|t0|t2|t3|t5, vec![[s45, s57, s15], [s46, s04, s01], [s46, s67, s01], [s01, s13, s67], [s37, s13, s67]]),
        (16, t2|t3|t4|t5|t6, vec![[s57, s15, s67], [s15, s67, s04], [s04, s67, s02], [s02, s67, s13], [s13, s37, s67]]),
        (17, t1|t2|t3|t4|t6|t7, vec![[s04, s02, s45], [s45, s57, s02], [s57, s15, s02], [s15, s01, s02]]),
    ];

    let cases = cases.map(|(n, id, mut tris)| {
        for tri in &mut tris {
            tri.sort();
        }
        (n, id, tris)
    });

    let mut found: [Option<(usize, Vec<[usize; 3]>)>; 256] = std::array::from_fn(|_| None);

    if true {
        for (n, i, c) in cases.iter() {
            found[*i] = Some((*n, c.clone()));
        }
        let mut progress = false;
        loop {
            for (transform, flip) in transforms {
                for i in 0..256 {
                    let Some(c) = &found[i] else {
                        continue;
                    };
                    let mut j = 0;
                    for bit in 0..8 {
                        let mask = 1 << bit;
                        if i & mask != 0 {
                            j |= 1 << transform[bit];
                        }
                    }

                    let n = c.0;

                    if let Some((n2, _)) = found[j] {
                        if n != n2 {
                            println!("{n} <-> {n2}");
                        }
                        if n >= n2 {
                            continue;
                        }
                    }

                    let c: Vec<_> =
                        c.1.iter()
                            .copied()
                            .map(|tri| {
                                let mut tri = tri.map(|edge_vertex| {
                                    let mut transformed =
                                        EDGES[edge_vertex].map(|vertex| transform[vertex]);
                                    transformed.sort();
                                    EDGES.indexof(transformed)
                                });
                                if flip {
                                    tri.swap(1, 2);
                                }
                                tri
                            })
                            .collect();

                    found[j] = Some((n, c));
                    progress = true;
                }
                if progress {
                    break;
                }
            }
            if !progress {
                for i in 0..256 {
                    let j = i ^ 255;
                    let mirrored = found[i].clone().map(|(n, mut tris)| {
                        tris.iter_mut().for_each(|tri| tri.swap(1, 2));
                        (n, tris)
                    });
                    if found[j].is_none() {
                        found[j] = mirrored;
                        progress = true;
                    } else if found[i].is_some() && found[j] != mirrored && (i < j) {
                        // found[j] = mirrored;
                        // progress = true;
                    }
                }
            }
            if !replace(&mut progress, false) {
                break;
            }
        }
    }

    let found = found.map(|f| {
        let mut f = f.unwrap();
        for (id, tri) in f.1.iter_mut().enumerate() {
            tri.sort();
            //let min = *tri.iter().min().unwrap();
            //*tri = if min == tri[0] {
            //    [tri[0], tri[1], tri[2]]
            //} else if min == tri[1] {
            //    [tri[1], tri[2], tri[0]]
            //} else {
            //    [tri[2], tri[0], tri[1]]
            //}
            // tri.sort();
            // for &t in &*tri {
            //     let [e0, e1] = EDGES[t];
            //     //assert!((((1 << e0) & id) == 0) != (((1 << e1) & id) == 0));
            // }
            // let [e0, e1] = EDGES[tri[0]].map(|e| Vector3::from(VERTS[e].map(|e| e as f32)));
            // let [e2, e3] = EDGES[tri[1]].map(|e| Vector3::from(VERTS[e].map(|e| e as f32)));
            // let [e4, e5] = EDGES[tri[2]].map(|e| Vector3::from(VERTS[e].map(|e| e as f32)));

            // let v01 = (e0 + e1) / 2.0;
            // let v23 = (e2 + e3) / 2.0;
            // let v45 = (e4 + e5) / 2.0;

            // if (dot((v23 - v01).cross(v45 - v01), e0 - e1) > 0.0)
            //     != ((1 << EDGES[tri[0]][0]) & id == 0)
            // {
            //     tri.swap(1, 2);
            // }
        }

        f
    });
    let triangle_to_edge = {
        let mut f: Vec<_> = found.iter().map(|(_, t)| t).flatten().copied().collect();
        f.sort();
        f.dedup();
        collect_arr(f.into_iter())
    };
    let found = found.map(|(_, f)| {
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

fn collect_arr<const N: usize, T: Debug>(i: impl Iterator<Item = T>) -> [T; N] {
    let arr: Vec<_> = i.collect();
    dbg!(arr.len(), N);
    arr.try_into().unwrap()
}

/// All the LUTs for cube maching.
/// The uniforms denormalize this.
pub(crate) struct CubeMarch {
    // case -> size
    pub(crate) case_to_size: [usize; 256],
    // case -> offset
    pub(crate) case_to_offset: [usize; 257],
    // offsets -> triangle
    pub(crate) offset_to_triangle: [usize; 820], // 732
    // triangle -> 3*edge.
    pub(crate) triangle_to_edge: [[usize; 3]; 190], // 135
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
