
struct VertexInput {
    @location(0) position: vec3<f32>,
    @builtin(vertex_index) in_vertex_index: u32,
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


@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.world_pos = model.position;
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

    return vec4<f32>(l, 0.0, 0.0, 1.0);

    //return vec4<f32>(sin(in.clip_position.z * 400.0) * 0.5 + 0.25, 0.0, 0.0, 1.0);
}

