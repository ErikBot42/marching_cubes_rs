


@group(0) @binding(0)
var<storage, read_write> triangle_count_prefix: array<u32>; 


// var<workgroup> wg0: array<u32, 256>;
// var<workgroup> wg1: array<u32, 256>;
var<workgroup> wg0: array<u32, 64>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_id: u32,
) {
    let i = local_id.x;
    let rw_idx = i * 128u + 128u;
    var c = triangle_count_prefix[rw_idx];

    // wg0[i] = c;
    // workgroupBarrier(); if i >= 1u   { c += wg0[i-1u];   } wg1[i] = c;
    // workgroupBarrier(); if i >= 2u   { c += wg1[i-2u];   } wg0[i] = c;
    // workgroupBarrier(); if i >= 4u   { c += wg0[i-4u];   } wg1[i] = c;
    // workgroupBarrier(); if i >= 8u   { c += wg1[i-8u];   } wg0[i] = c;
    // workgroupBarrier(); if i >= 16u  { c += wg0[i-16u];  } wg1[i] = c;
    // workgroupBarrier(); if i >= 32u  { c += wg1[i-32u];  } wg0[i] = c;
    // workgroupBarrier(); if i >= 64u  { c += wg0[i-64u];  } wg1[i] = c;
    // workgroupBarrier(); if i >= 128u { c += wg1[i-128u]; } 
    c = subgroupInclusiveAdd(c);
    if ((i & (subgroup_size - 1u)) == (subgroup_size - 1u)) {
        wg0[i / subgroup_size] = c;
    }
    workgroupBarrier();
    if i < subgroup_size {
        wg0[i] = subgroupExclusiveAdd(wg0[i]);
    }
    workgroupBarrier();
    c += wg0[i / subgroup_size];

    triangle_count_prefix[rw_idx] = c;



}

