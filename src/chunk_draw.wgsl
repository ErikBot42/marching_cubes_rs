


struct VertexInput {
    @location(0) mask: u32,
    @builtin(vertex_index) vertex_index: u32,
};
struct VertexOutput {
    // (x, y, z, w) -> (x/w, y/w, z/w)
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_pos: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) dbg: vec3<f32>,
};

struct CameraUniform {
    view: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    lmap: mat4x4<f32>,
    lmap_inv: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;


@group(0) @binding(1)
var<storage, read_write> render_case: array<u32>; 

@vertex
fn vs_main(
    data: VertexInput,
) -> VertexOutput {

    let vid = data.vertex_index % 3u;

    let pos_mask = (data.vertex_index - vid) / 3u;

    let pos_offset = vec3<f32>(vec3<u32>(pos_mask & 1023u, (pos_mask >> 10u) & 1023u, (pos_mask >> 20u) & 1023u)) * 2.0;

    let tri = data.mask & 255u;
    let x = (data.mask >> 8u) & 31u;
    let y = (data.mask >> 13u) & 31u;
    let z = (data.mask >> 18u) & 31u;

    let rcase = (render_case[tri] >> (vid * 6u)) & 63u;

    let w = ((data.mask >> (23u + vid * 3u)) & 7u) * 2u;
    let pos0 = vec3<u32>((rcase >> 0) & 1, (rcase >> 1) & 1, (rcase >> 2) & 1) * (14 - w);
    let pos1 = vec3<u32>((rcase >> 3) & 1, (rcase >> 4) & 1, (rcase >> 5) & 1) * (w);

    let c = vec3<u32>(x, y, z) * 2 * 7 + pos0 + pos1;

    let pos = (vec3<f32>(c)/7.0 - 32.0) * (1.0 / 32.0) + pos_offset;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.world_pos = pos;
    out.view_pos = (camera.view * vec4<f32>(pos, 1.0)).xyz;
    out.dbg = vec3<f32>(
        f32((data.mask >> 8) & 255) / 255.0,
        f32((data.mask >> 16) & 255) / 255.0,
        f32((data.mask >> 24) & 255) / 255.0,
    );
    return out;
}


// local 
// -model> 
// world 
// -view> 
// view 
// -projection> 
// clip 
// -viewport transform> 
// screen space

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    let view_pos_dx = dpdx(in.view_pos);
    let view_pos_dy = dpdy(in.view_pos);
    let normal = -normalize(cross(view_pos_dx, view_pos_dy));


    let radius = 4.0;

    let lmap = camera.lmap;   
    let light_space_pos = (lmap * vec4<f32>(in.world_pos, 1.0)).xyz;


    let light_space_light_pos = (floor(light_space_pos/radius)+0.5)*radius;

    let light_pos = (camera.view * (camera.lmap_inv * vec4<f32>(light_space_light_pos, 1.0))).xyz;

    // let light_pos_world = (floor((lmap * in.world_pos)/radius)+0.5)*radius;

    // let light_pos = (camera.view * vec4<f32>(light_pos_world, 1.0)).xyz;

    let light_vector = light_pos - in.view_pos;

    let light_dist = length(light_vector);

    let light_strength = max((1.0 / (light_dist * light_dist) - 4.0 / (radius * radius)) - 0.0, 0.0);

    let light_dir = normalize(light_vector);

    let view_dir = normalize(-in.view_pos); 

    let lambertian = max(dot(light_dir, normal), 0.0);

    let h = normalize(light_dir + view_dir);

    let spec_angle = max(dot(h, normal), 0.0);

    let specular = pow(spec_angle, 10.0);


    let color = 0.003 + lambertian * light_strength;

    
    // return vec4<f32>((light_pos_world + 1.0) % 0.9, 1.0);

    return vec4<f32>(color, 0.0*dot(light_dir, normal), 0.0, 1.0);
}

