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

const case_triangle_offset: array<u32> = array(...);

const case_triangle_number: array<u32> = array(...);

// const case_triangle_edges: array<array<u32, 3>> = array(...);

// const case_edges: array<u32> = array(...);

@group(0) @binding(0)
var<storage, read_write> sdf_data: array<f32>; 

@group(0) @binding(1)
var<storage, read_write> triangle_count_prefix: array<u32>; 

@group(0) @binding(1)
var<storage, read_write> triangle_storage: array<u32>; 

struct Chunk {
    pos: vec3<i32>;
    write_offset: u32,
}

@group(0) @binding(2)
var<uniform> chunk: Chunk;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {

    let gid = global_id.x;
    var c = triangle_count_prefix[gid];
    let src_idx = (gid / 128u) * 128u
    if gid % 128u != 127u && src_idx != 0u {
        c += triangle_count_prefix[src_idx - 1u];
    }

    let write_start = c + chunk.write_offset;


    var idx = 0;


    let sd = array(
        sdf_data[id],
        sdf_data[id+1u],
        sdf_data[id+32u],
        sdf_data[id+33u],
        sdf_data[id+1024u], 
        sdf_data[id+1025u],
        sdf_data[id+1056u],
        sdf_data[id+1057u],
    );

    idx |= u32(sd[0]>0.0) << 0u;
    idx |= u32(sd[1]>0.0) << 1u;
    idx |= u32(sd[2]>0.0) << 2u;
    idx |= u32(sd[3]>0.0) << 3u;
    idx |= u32(sd[4]>0.0) << 4u;
    idx |= u32(sd[5]>0.0) << 5u;
    idx |= u32(sd[6]>0.0) << 6u;
    idx |= u32(sd[7]>0.0) << 7u;

    let tris = case_triangle_count[idx];

    let x = gid % 32u;
    let y = (gid / 32u) % 32u;
    let z = gid / 1024u;

    // ________ _____ _____ _____ ___ ___ ___
    // TTTTTTTT XXXXX YYYYY ZZZZZ AAA BBB CCC 
    // 8T       5X    5Y    5Z    3A  3B  3C
    // 0123
    for (var i: u32 = 0u; i < tris; i += 1) {
        let case_data_idx = case_triangle_offset[idx] + i;

        let triangle = case_triangle_number[case_data_idx];
        // let edges = case_triangle_edges[case_data_idx];

        var bit = 0;
        bit |= triangle;
        bit |= x >> 5;
        bit |= y >> 10;
        bit |= z >> 15;

        triangle_storage[i + write_start] = bit;
    }



}
