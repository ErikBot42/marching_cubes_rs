
const num_vertex: array<u32> = [
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2, 
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 2, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2, 
    2, 3, 3, 4, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2, 3, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2, 
    2, 3, 3, 4, 3, 4, 4, 3, 3, 4, 2, 3, 4, 3, 3, 2, 3, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 
    2, 3, 3, 4, 3, 4, 4, 3, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 
    3, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 2, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0
];


fn sdf(x0: u32, y0: u32, z0: u32) -> f32 {
    let x = (f32(x0) / 2.0) % 30.0 - 15.0;
    let y = (f32(y0) / 2.0) % 30.0 - 15.0;
    let z = (f32(z0) / 2.0) % 30.0 - 15.0;

    return sqrt(x * x + y * y + z * z) - 10.0;
}

@group(0) @binding(0)
var<storage, read_write> sdf_data: array<f32>; // swap with u8 later


// 8 * 8 * 8 = 512
// work in cubes to share some (cache) work.
// start with 8 * 8, 8 long prefix sums.
@compute @workgroup_size(8, 8, 8)
fn cs_sdf(
    // position in workgroup
    @builtin(local_invocation_id) local_id: vec3<u32>,
    // what index in workgroup
    @builtin(local_invocation_index) local_index: u32,
    // global coords
    @builtin(global_invocation_id) global_id: vec3<u32>,
    // number of workgroups in each dimension
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    // what workgroup number we are in.
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let workgrup_size = vec3<u32>(8, 8, 8);
    let extent = workgroup_size * num_workgroups;
    let thread_id = global_id.x + global_id.y * extent.x + global_id.z * extent.x * extent.y;
    let x = global_id.x % SIZE;
    let y = (global_id.y / SIZE) % SIZE;
    let z = global_id.z / (SIZE * SIZE);

    let s = sdf(x, y, z);

    let extent_x = num_workgroups.x * workgroup_size.x;
    let extent_y = num_workgroups.y * workgroup_size.y;
    let extent_z = num_workgroups.z * workgroup_size.z;
    sdf_data[x + extent_x * y + extent_x * extent_y * z];
}

// TODO: move bindings to separate shaders.
@group(0) @binding(1)
var<storage, read_write> cell_tri_idx: array<u32>; // (replace with u8) what 8 bit idx to use

@group(0) @binding(2)
var<storage, read_write> cell_vertex_sum: array<u32>; // what we will run prefix sum on.


const WG_SIZE = 64;
var<workgroup> wg_prefix: array<u32, 64>;

// 1 instance/tri => 135 (8 bit, 121 unused) possible triangles, 24 position bits => 128 wide chunks.

// use "virtual" vertex data to distinguish between chunks in worst case.

// 32^3: 128 * 256 <- best
// ____ ____ ____ ____ ____ ____ ____ ____ 
// TTTT TTTT PPPP PPPP PPPP PPPA AAAA AAAA
// 8T 15P 9A

// 64^3: 512 * 512
// ____ ____ ____ ____ ____ ____ ____ ____ 
// TTTT TTTT PPPP PPPP PPPP PPPP PPAA AAAA
// 8T 18P 6A

// 128^3:
// ____ ____ ____ ____ ____ ____ ____ ____ 
// TTTT TTTT PPPP PPPP PPPP PPPP PPPP PAAA
// 8T 21P 3A

// * | | | | | | | | *

// 32^3: 128 * 256
// ________ _____ _____ _____ ___ ___ ___
// TTTTTTTT XXXXX YYYYY ZZZZZ AAA BBB CCC 
// 8T       5X    5Y    5Z    3A  3B  3C

// separate sums per tris per voxel -> instancing per size.
// 32^3 chunks only need 15 bit voxel ids.
// 32 sum pass, 32 sum pass, 32 sum pass, 32 sum writeback.
// use auxilary space for middle passes.

fn prefix_sum_wg(v0: u32, id: u32) -> u32 {
    var v: u32 = v0;
    wg_prefix[id] = v;
    for (var i: u32 = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
        workgroupBarrier();
        if id >= (1u << i)  {
            v += wg_prefix[id - (1u << i)];
        }
        workgroupBarrier();
        wg_prefix[id] = v;
    }
    return v;
}

@compute @workgroup_size(512, 1, 1) // prioritize prefix sum over access to sdf values?
fn cs_init_count(
    // position in workgroup
    @builtin(local_invocation_id) local_id: vec3<u32>,
    // what index in workgroup
    @builtin(local_invocation_index) local_index: u32,
    // global coords
    @builtin(global_invocation_id) global_id: vec3<u32>,
    // number of workgroups in each dimension
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    // what workgroup number we are in.
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {

    // dispatch size "*" workgroup_size = total size
    let workgrup_size = vec3<u32>(8, 8, 8);
    let extent = workgroup_size * num_workgroups;
    let thread_id = global_id.x + global_id.y * extent.x + global_id.z * extent.x * extent.y;
    let x = global_id.x % SIZE;
    let y = (global_id.y / SIZE) % SIZE;
    let z = global_id.z / (SIZE * SIZE);

    var idx = 0;
    for (var i = 0; i < 8; i += 1) {
        let x1 = x + i & 1u;
        let y1 = y + (i & 2u) >> 1;
        let z1 = z + (i & 4u) >> 2;
        let bit = sdf_data[x1 + extent.x * y1 + extent.x * extent.y * z1] > 0;
        idx |= u32(bit) << i;
    }

    let global_i = x + extent.x * y + extent.x * extent.y * z;

    cell_tri_idx[global_i] = idx;
    let size = num_vertex[idx];

// 0b1000000000 512
// 0b0100000000 256
// 0b0010000000 128
// 0b0001000000 64
// 0b0000100000 32
// 0b0000010000 16
// 0b0000001000 8
// 0b0000000100 4

    let wg_i = local_index;
    cell_vertex_sum[wg_i] = size;
    workgroupBarrier();
    if (wg_i & u1) {

    }

}






