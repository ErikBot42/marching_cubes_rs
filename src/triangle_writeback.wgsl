
@group(0) @binding(0)
var<storage, read_write> sdf_data: array<f32>; 

@group(0) @binding(1)
var<storage, read_write> triangle_count_prefix: array<u32>; 

@group(0) @binding(2)
var<storage, read_write> triangle_storage: array<u32>; 

@group(0) @binding(3)
var<storage, read> this_u: TriWriteBackUniform;

@group(0) @binding(4)
var<uniform> chunk: Chunk;

struct Chunk {
    pos: vec3<i32>,
    write_offset: u32,
}

struct TriWriteBackUniform {
    offset_to_bitpack: array<u32, 732>,
    case_to_offset: array<u32, 257>,
    _unused0: u32,
    _unused1: u32,
    _unused2: u32,
}



@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {

    let gid = global_id.x;
    var c = triangle_count_prefix[gid];
    let src_idx = (gid / 128u) * 128u;
    if gid % 128u != 0u {
        c += triangle_count_prefix[src_idx];
    }


    let write_start: u32 = c + chunk.write_offset;


    let dx: u32 = 1u;
    let dy: u32 = 33u;
    let dz: u32 = 33u*33u;

    let x: u32 = gid % 32u;
    let y: u32 = (gid / 32u) % 32u;
    let z: u32 = gid / 1024u;

    let id: u32 = x + y * 33u + z * 33u * 33u;
    var sd: array<f32, 8> = array(
        sdf_data[id + (0u + 0u + 0u)],
        sdf_data[id + (dx + 0u + 0u)],
        sdf_data[id + (0u + dy + 0u)],
        sdf_data[id + (dx + dy + 0u)],
        sdf_data[id + (0u + 0u + dz)], 
        sdf_data[id + (dx + 0u + dz)],
        sdf_data[id + (0u + dy + dz)],
        sdf_data[id + (dx + dy + dz)],
    );

    var idx: u32 = 0u;
    idx |= u32(sd[0]>0.0) << 0u;
    idx |= u32(sd[1]>0.0) << 1u;
    idx |= u32(sd[2]>0.0) << 2u;
    idx |= u32(sd[3]>0.0) << 3u;
    idx |= u32(sd[4]>0.0) << 4u;
    idx |= u32(sd[5]>0.0) << 5u;
    idx |= u32(sd[6]>0.0) << 6u;
    idx |= u32(sd[7]>0.0) << 7u;

    let case_to_offset = this_u.case_to_offset[idx];
    let case_to_offset1 = this_u.case_to_offset[idx + 1u];

    let tris: u32 = case_to_offset1 - case_to_offset;


    // ________ _____ _____ _____ ___ ___ ___
    // TTTTTTTT XXXXX YYYYY ZZZZZ AAA BBB CCC 
    // 8T       5X    5Y    5Z    3A  3B  3C
    // 0123
    for (var i: u32 = 0u; i < tris; i += 1u) {
        let case_data_idx: u32 = this_u.case_to_offset[idx] + i;

        let bitpack = this_u.offset_to_bitpack[case_data_idx];
        let triangle = bitpack & 255u;// bitpack >> 18;
        // let edges = case_triangle_edges[case_data_idx];


        let c0 = sd[(bitpack >> 8u) & 7u];
        let c1 = sd[(bitpack >> 11u) & 7u];

        let t0 = interpolate_bits(c0, c1);

        let c2 = sd[(bitpack >> 14u) & 7u];
        let c3 = sd[(bitpack >> 17u) & 7u];

        let t1 = interpolate_bits(c2, c3);

        let c4 = sd[(bitpack >> 20u) & 7u];
        let c5 = sd[(bitpack >> 23u) & 7u];

        let t2 = interpolate_bits(c4, c5);

        var bit: u32 = 0u;
        bit |= triangle;
        bit |= x << 8;
        bit |= y << 13;
        bit |= z << 18;
        bit |= t0 << 23;
        bit |= t1 << 26;
        bit |= t2 << 29;

        triangle_storage[i + write_start] =  bit;
    }
}

fn interpolate_bits(a: f32, b: f32) -> u32 {
    let sa = abs(a);
    let sb = abs(b);
    let s = sa / (sa + sb);

    //return u32(clamp(i32(round(8 * s + 0.5)), 0, 7));
    return u32(clamp(i32(round(7 * s)), 0, 7));
    // return u32(clamp(i32(round(8 * s)), 0, 7));
}


