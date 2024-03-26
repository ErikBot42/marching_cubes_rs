
const case_triangle_count: array<u32> = array(
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2, 
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 2, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2, 
    2, 3, 3, 4, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2, 3, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2, 
    2, 3, 3, 4, 3, 4, 4, 3, 3, 4, 2, 3, 4, 3, 3, 2, 3, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 
    2, 3, 3, 4, 3, 4, 4, 3, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 
    3, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 2, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0
);

@group(0) @binding(0)
var<storage, read_write> sdf_data: array<f32>; 

@group(0) @binding(1)
var<storage, read_write> triangle_count_prefix: array<u32>; 

var<workgroup> wg0: array<u32, 128>;
var<workgroup> wg1: array<u32, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let id = global_id.x;

    var idx = 0;

    idx |= u32(sdf_data[id]>0) << 0;
    idx |= u32(sdf_data[id+1]>0) << 1;
    idx |= u32(sdf_data[id+32]>0) << 2;
    idx |= u32(sdf_data[id+33]>0) << 3;
    idx |= u32(sdf_data[id+1024]>0) << 4;
    idx |= u32(sdf_data[id+1025]>0) << 5;
    idx |= u32(sdf_data[id+1056]>0) << 6;
    idx |= u32(sdf_data[id+1057]>0) << 7;

    let tris = case_triangle_count[idx];

    let i = local_id.x;

    var c = tris;
    wg0[i] = c;

    workgroupBarrier(); if i >= 1u   { c += wg0[i-1u];   } wg1[i] = c;
    workgroupBarrier(); if i >= 2u   { c += wg1[i-2u];   } wg0[i] = c;
    workgroupBarrier(); if i >= 4u   { c += wg0[i-4u];   } wg1[i] = c;
    workgroupBarrier(); if i >= 8u   { c += wg1[i-8u];   } wg0[i] = c;
    workgroupBarrier(); if i >= 16u  { c += wg0[i-16u];  } wg1[i] = c;
    workgroupBarrier(); if i >= 32u  { c += wg1[i-32u];  } wg0[i] = c;
    workgroupBarrier(); if i >= 64u  { c += wg0[i-64u];  }

    // pseudo-exclusive prefix sum
    triangle_count_prefix[id+1] = c;
}
