


struct VertexInput {
    @location(0) mask: u32,
    @builtin(vertex_index) vertex_index: u32,
};
struct VertexOutput {
    // (x, y, z, w) -> (x/w, y/w, z/w)
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

struct CameraUniform {
    view: mat4x4<f32>, // world, view
    view_proj: mat4x4<f32>, // world, clip
    lmap: mat4x4<f32>, // world, light
    lmap_inv: mat4x4<f32>, // light, world
    time: f32,
    _unused0: f32,
    _unused1: f32,
    _unused2: f32,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;


@group(0) @binding(1)
var<storage, read> render_case: array<u32>; 

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

    let pos = (vec3<f32>(c)/7.0) * (1.0 / 32.0) + pos_offset;

    var out: VertexOutput;

    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.world_pos = pos;
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
fn fs_wire(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let r = 4.0;

    let world_view = camera.view;
    let world_light = camera.lmap;
    let light_world = camera.lmap_inv;

    let p_world = in.world_pos + camera.time * 0.1;
    let p_view = (world_view * vec4f(p_world, 1.0)).xyz;
    let p_light = (world_light * vec4f(p_world, 1.0)).xyz;
    let l_light = (floor(p_light / r) + 0.5) * r;

    let l_view = (world_view * (light_world * vec4<f32>(l_light, 1.0))).xyz;
    let lp_view = l_view - p_view;

    let light_strength = max(1.0 / dot(lp_view, lp_view) - 4.0 / (r * r), 0.0);
    let lp_view_norm = normalize(lp_view);
    let view_dir = normalize(-p_view); 

    let view_pos_dx = dpdx(p_view);
    let view_pos_dy = dpdy(p_view);
    let normal = -normalize(cross(view_pos_dx, view_pos_dy));

    let lambertian = max(dot(lp_view_norm, normal), 0.0);
    let h = normalize(lp_view_norm + view_dir);
    let spec_angle = max(dot(h, normal), 0.0);
    let specular = pow(spec_angle, 10.0);
    let color = 0.003 + lambertian * light_strength;
    
    // return vec4<f32>((light_pos_world + 1.0) % 0.9, 1.0);

    return vec4<f32>(color, 1.0/length(p_view), 0.0, 1.0);
}

