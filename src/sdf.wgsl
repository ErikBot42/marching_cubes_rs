


fn sdf(x0: i32, y0: i32, z0: i32) -> f32 {
    let x = (f32(x0) / 2.0);// % 30.0 - 8.0;
    let y = (f32(y0) / 2.0);// % 30.0 - 8.0;
    let z = (f32(z0) / 2.0);// % 30.0 - 8.0;
    
    let p = vec3<f32>(x, y, z) - 16.0;


    return length(p%32.0) - 16.0;

    // return sqrt(p.x * p.x + p.y * p.y + p.z * p.z) - 16.0;
    //let r = sin(p/2.0) + sin(p*1.0)/5.0;
    //return r.x + r.y + r.z*3.0 + sin(y * 0.1)*4.0 + sin(z * 0.134234)*2.0 + sin(x * 0.12377823)*3.0;
    // // let rad = 80.0;
    // // let t = (abs(p) % (rad * 4.0)) - rad;

    // // return length(t) - rad;
}



struct Chunk {
    pos: vec3<i32>,
    _unused: i32,
}

@group(0) @binding(0)
var<uniform> chunk: Chunk;

@group(0) @binding(1)
var<storage, read_write> sdf_data: array<f32>; // 33^3 + padding

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let i: i32 = i32(global_id.x);
    let pos: vec3<i32> = vec3<i32>(i % 33, (i / 33) % 33, i / (33 * 33)) + chunk.pos * 32;
    let s: f32 = sdf(pos.x, pos.y, pos.z);
    sdf_data[i] = s;
}

