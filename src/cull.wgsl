
struct DrawCall {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};


struct CameraUniform {
    world_view: mat4x4<f32>,  // AKA view
    view_world: mat4x4<f32>,  // AKA view_inv

    world_clipw: mat4x4<f32>, // AKA view_proj
    clipw_world: mat4x4<f32>, // AKA view_proj_inv
    clipw_view: mat4x4<f32>,

    world_light: mat4x4<f32>,
    light_world: mat4x4<f32>,

    time: f32,
    cull_radius: f32,
    fog_inv: f32,
    _unused2: f32,

    world_clipw_cull: mat4x4<f32>,

    fog_color: vec3<f32>,
    diffuse_color: vec3<f32>,
    specular_color: vec3<f32>,
    light_color: vec3<f32>,
    sun_color: vec3<f32>,

    base_offset: vec3<i32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var<storage, read> src_buffer: array<DrawCall>; 

@group(0) @binding(2)
var<storage, read_write> dst_buffer: array<DrawCall>; 

// buffer does not need to be zeroed, but bogus compute will occur.

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i: u32 = global_id.x;

    let src: DrawCall = src_buffer[i];

    let pos_mask = src.first_vertex / 3u;
    let pos_offset = vec4<f32>(vec3<f32>(
        vec3<i32>(vec3<u32>(pos_mask & 1023u, (pos_mask >> 10u) & 1023u, (pos_mask >> 20u) & 1023u))
        - camera.base_offset
    ) * 2.0 + 1.0, 1.0);

    let pos = camera.world_clipw_cull * pos_offset;

    var visible = true;

    // distance = pos.z / pos.w

    // distance < 10.0 <=> pos.z / pos.w < 10.0 <=> pos.z < pos.w * 10.0



    visible = visible && (abs(pos.x) < pos.w);
    visible = visible && (abs(pos.y) < pos.w);
    visible = visible && pos.z > 0.0;
    if visible {
        // distance cull
        let rad = camera.cull_radius;
        let pos_view = camera.world_view * pos_offset;
        visible = visible && pos_view.z > -rad;
        visible = visible && dot(pos_view.xyz, pos_view.xyz) < (rad * pos_view.w) * (rad * pos_view.w);
    }

    if visible {
        dst_buffer[i] = src;
    } else {
        var tmp: DrawCall;
        tmp.vertex_count = 0u;
        tmp.instance_count = 0u;
        tmp.first_vertex = 0u;
        tmp.first_instance = 0u;
        dst_buffer[i] = tmp;
    }
}
