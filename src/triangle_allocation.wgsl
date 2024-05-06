
struct TriAllocUniform {
    case_triangle_count: array<u32, 256>
}

@group(0) @binding(0)
var<storage, read> this_u: TriAllocUniform;

@group(0) @binding(1)
var<storage, read> sdf_data: array<f32>; 

@group(0) @binding(2)
var<storage, read_write> triangle_count_prefix: array<u32>; 

// @group(0) @binding(3)
// var<storage, read_write> something_unused: array<u32>; 

// var<workgroup> wg0: array<u32, 128>;
// var<workgroup> wg1: array<u32, 128>;

var<workgroup> wg0: array<u32, 128>;
var<workgroup> wg1: array<u32, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_id: u32,
) {
    let gid: u32 = global_id.x;

    var idx: u32 = 0;

    let dx: u32 = 1u;
    let dy: u32 = 33u;
    let dz: u32 = 33u*33u;

    let x: u32 = gid % 32u;
    let y: u32 = (gid / 32u) % 32u;
    let z: u32 = gid / 1024u;

    let id: u32 = x + y * 33u + z * 33u * 33u;
    idx |= u32(sdf_data[id + (0u + 0u + 0u)] > 0.0) << 0;
    idx |= u32(sdf_data[id + (dx + 0u + 0u)] > 0.0) << 1;
    idx |= u32(sdf_data[id + (0u + dy + 0u)] > 0.0) << 2;
    idx |= u32(sdf_data[id + (dx + dy + 0u)] > 0.0) << 3;
    idx |= u32(sdf_data[id + (0u + 0u + dz)] > 0.0) << 4;
    idx |= u32(sdf_data[id + (dx + 0u + dz)] > 0.0) << 5;
    idx |= u32(sdf_data[id + (0u + dy + dz)] > 0.0) << 6;
    idx |= u32(sdf_data[id + (dx + dy + dz)] > 0.0) << 7;

    let tris = this_u.case_triangle_count[idx];

    let i = local_id.x;

    var c = tris;

    // inclusive prefix-sum
    switch (subgroup_size) {
    default { 
        wg0[i] = c;
        workgroupBarrier(); if i >= 1u   { c += wg0[i - 1u];  } wg1[i] = c;
        workgroupBarrier(); if i >= 2u   { c += wg1[i - 2u];  } wg0[i] = c;
        workgroupBarrier(); if i >= 4u   { c += wg0[i - 4u];  } wg1[i] = c;
        workgroupBarrier(); if i >= 8u   { c += wg1[i - 8u];  } wg0[i] = c;
        workgroupBarrier(); if i >= 16u  { c += wg0[i - 16u]; } wg1[i] = c;
        workgroupBarrier(); if i >= 32u  { c += wg1[i - 32u]; } wg0[i] = c;
        workgroupBarrier(); if i >= 64u  { c += wg0[i - 64u]; }
    }
    case 16u: {
        c = subgroupInclusiveAdd(c);

        if ((i & (16u - 1u)) == (16u - 1u)) {
            wg0[i >> 4u] = c;
        }
        workgroupBarrier();
        if i < 16u {
            // gid >= 8u contains trash.
            wg0[i] = subgroupExclusiveAdd(wg0[i]);
        }
        workgroupBarrier();
        c += wg0[i / 16u];
    }
    }

    // pseudo-exclusive prefix sum
    triangle_count_prefix[gid+1] = c;
}
