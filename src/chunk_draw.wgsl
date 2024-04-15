


struct VertexInput {
    @location(0) mask: u32,
    @builtin(vertex_index) vertex_index: u32,
};
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
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
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    //return vec4<f32>(abs(sin(in.clip_position.z * 10.0)), 0.2, 0.1, 1.0);

    let c: vec3<f32> = in.clip_position.xyz;

    let dx = dpdx(in.world_pos);
    let dy = dpdy(in.world_pos);

    let v = -normalize(cross(dx, dy));

    let light = normalize(vec3<f32>(1.0, 1.0, 1.0));//normalize(camera.view_proj * vec4<f32>(1.0, 1.0, 1.0, 0.0)).xyz;

    let l = max(dot(v, light), 0.0);
    let r = max(-dot(v, light), 0.0);

    return vec4<f32>(l, r, 0.0, 1.0);

    //return vec4<f32>(sin(in.clip_position.z * 400.0) * 0.5 + 0.25, 0.0, 0.0, 1.0);
}

