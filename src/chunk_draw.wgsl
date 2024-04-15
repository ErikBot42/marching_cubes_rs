


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

    let tri = data.mask & 255u;
    let x = (data.mask >> 8) & 31;
    let y = (data.mask >> 13) & 31;
    let z = (data.mask >> 18) & 31;

    let rcase = (render_case[tri] >> ((data.vertex_index) * 6u)) & 63u;

    let w = ((data.mask >> (23 + data.vertex_index*3u)) & 7) * 2;
    let pos0 = vec3<u32>((rcase >> 0) & 1, (rcase >> 1) & 1, (rcase >> 2) & 1) * (14 - w);
    let pos1 = vec3<u32>((rcase >> 3) & 1, (rcase >> 4) & 1, (rcase >> 5) & 1) * (w);

    let c = vec3<u32>(x, y, z) * 2 * 7 + pos0 + pos1;

    let pos = (vec3<f32>(c)/7.0 - 32.0) * (1.0 / 32.0);

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

