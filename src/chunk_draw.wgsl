


struct VertexInput {
    @location(0) mask: u32,
    @builtin(vertex_index) vertex_index: u32,
};
struct VertexOutput {
    // (x, y, z, w) -> (x/w, y/w, z/w)
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_pos: vec3<f32>,
};

struct CameraUniform {
    world_view: mat4x4<f32>,  // AKA view
    view_world: mat4x4<f32>,  // AKA view

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
var<storage, read> render_case: array<u32>; 

@vertex
fn vs_main(
    data: VertexInput,
) -> VertexOutput {

    let vid = data.vertex_index % 3u;

    let pos_mask = (data.vertex_index - vid) / 3u;

    let pos_offset = vec3<f32>(
        vec3<i32>(vec3<u32>(pos_mask & 1023u, (pos_mask >> 10u) & 1023u, (pos_mask >> 20u) & 1023u))
        - camera.base_offset
    ) * 2.0;

    let tri = data.mask & 255u;
    let x = (data.mask >> 8u) & 31u;
    let y = (data.mask >> 13u) & 31u;
    let z = (data.mask >> 18u) & 31u;

    let rcase = (render_case[tri] >> (vid * 6u)) & 63u;

    let w = ((data.mask >> (23u + vid * 3u)) & 7u) * 2u;
    let pos0 = vec3<u32>((rcase >> 0) & 1, (rcase >> 1) & 1, (rcase >> 2) & 1) * (14 - w);
    let pos1 = vec3<u32>((rcase >> 3) & 1, (rcase >> 4) & 1, (rcase >> 5) & 1) * (w);

    let c = vec3<u32>(x, y, z) * 2 * 7 + pos0 + pos1;

    let pos = (vec3<f32>(c)/7.0) * (1.0 / 32.0) + pos_offset; // world space

    var out: VertexOutput;

    out.clip_position = camera.world_clipw * vec4<f32>(pos, 1.0);
    out.view_pos = (camera.world_view * vec4<f32>(pos, 1.0)).xyz;
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
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return shade(in.view_pos);
}

@fragment
fn fs_depth(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@vertex
fn vs_deferred(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    return vec4<f32>(pos[vertex_index], 0.0, 1.0);
}

@group(1) @binding(0) var depth_texture: texture_depth_2d;

@fragment
fn fs_deferred(@builtin(position) screen_coords: vec4<f32>) -> @location(0) vec4<f32> {
    let depth = textureLoad(depth_texture, vec2<i32>(floor(screen_coords.xy)), 0);
    if depth >= 1.0 { discard; }

    let uv = screen_coords.xy / vec2<f32>(textureDimensions(depth_texture));
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);

    let view_posw = camera.clipw_view * clip;
    let view_pos = view_posw.xyz / view_posw.w;
    return shade(view_pos);
}

// view space -> color
fn blinn_phong(
    surface_camera: vec3<f32>, // |v| = 1
    surface_light: vec3<f32>, // |v| = 1
    normal: vec3<f32>, // |v| = 1
    color: vec3<f32>, // diffuse, specular, hardness
    light: vec3<f32>, // 0..1 (light color * light strength)
) -> vec3<f32> {
    let diffuse = saturate(dot(normal, surface_light));
    let specular = pow(saturate(dot(normal, normalize(surface_light + surface_camera))), color.z);
    return dot(vec2<f32>(diffuse, specular), color.xy) * light;
}

fn shade(p_view0: vec3<f32>) -> vec4<f32> {

    let view_dir = normalize(-p_view0); 

    let normal = -normalize(cross(dpdx(p_view0), dpdy(p_view0)));

    var acc: vec3<f32>;
    acc += blinn_phong(
        view_dir, 
        (camera.world_view * vec4<f32>(normalize(vec3<f32>(1.0, 1.0, 1.0)), 0.0)).xyz,
        normal,
        vec3<f32>(0.5, 1.0, 100.0),
        vec3<f32>(0.98, 0.77, 0.40),
    );
    acc += blinn_phong(
        view_dir, 
        (camera.world_view * vec4<f32>(normalize(vec3<f32>(0.0, 1.0, 0.0)), 0.0)).xyz,
        normal,
        vec3<f32>(0.5, 0.0, 0.0),
        camera.fog_color,
    );
    acc += blinn_phong(
        view_dir, 
        (camera.world_view * vec4<f32>(normalize(vec3<f32>(-1.0, 0.0, -1.0)), 0.0)).xyz,
        normal,
        vec3<f32>(0.2, 0.0, 0.0),
        vec3<f32>(0.1, 0.2, 0.30),
    );
    
    let up = (camera.world_view * vec4<f32>(normalize(vec3<f32>(0.0, 1.0, 0.0)), 0.0)).xyz;
    let color = acc;

    let fog_factor = pow(min(dot(p_view0, p_view0) * camera.fog_inv, 1.0), 2.0);

    //let dir = camera.view_world
    
    let interpolated = mix(color, background((vec4<f32>(-view_dir, 0.0) * camera.view_world).xyz), fog_factor);

    return vec4<f32>(interpolated, 1.0);
}

fn background(dir: vec3<f32>) -> vec3<f32> {
    return camera.fog_color;// * saturate(0.2 + dot(dir, vec3<f32>(1.0, 1.0, 1.0)));
}

fn hash(p: vec3<i32>) -> f32 {
    let s = abs(dot(p, vec3<i32>(5659, 7333, 3037)));
    return fract(f32(s) / 10000.0);
}

fn noise_texture(p0: vec3<f32>) -> f32 {
    var t = 0.0;
    let p = p0 * 0.5;
    let dp = dot(fwidth(p0), vec3<f32>(1.0));
    //t += hash(vec3<i32>(p * 1.0));
    //t += hash(vec3<i32>(p * 4.0));
    //t += hash(vec3<i32>(p * 16.0));

    var n = 0;
    {
        let s = 16.0;
        let f = smoothstep(0.0, s, 1.0/dp);
        t += hash(vec3<i32>(p * s)) * f + 0.5 * (1.0 - f);
        n += 1;
    }
    {
        let s = 32.0;
        let f = smoothstep(0.0, s, 1.0/dp);
        t += hash(vec3<i32>(p * s)) * f + 0.5 * (1.0 - f);
        n += 1;
    }
    {
        let s = 64.0;
        let f = smoothstep(0.0, s, 1.0/dp);
        t += hash(vec3<i32>(p * s)) * f + 0.5 * (1.0 - f);
        n += 1;
    }
    {
        let s = 128.0;
        let f = smoothstep(0.0, s, 1.0/dp);
        t += hash(vec3<i32>(p * s)) * f + 0.5 * (1.0 - f);
        n += 1;
    }
    {
        let s = 256.0;
        let f = smoothstep(0.0, s, 1.0/dp);
        t += hash(vec3<i32>(p * s)) * f + 0.5 * (1.0 - f);
        n += 1;
    }
    {
        let s = 512.0;
        let f = smoothstep(0.0, s, 1.0/dp);
        t += hash(vec3<i32>(p * s)) * f + 0.5 * (1.0 - f);
        n += 1;
    }
    {
        let s = 1024.0;
        let f = smoothstep(0.0, s, 1.0/dp);
        t += hash(vec3<i32>(p * s)) * f + 0.5 * (1.0 - f);
        n += 1;
    }
    {
        let s = 2048.0;
        let f = smoothstep(0.0, s, 1.0/dp);
        t += hash(vec3<i32>(p * s)) * f + 0.5 * (1.0 - f);
        n += 1;
    }
    {
        let s = 4096.0;
        let f = smoothstep(0.0, s, 1.0/dp);
        t += hash(vec3<i32>(p * s)) * f + 0.5 * (1.0 - f);
        n += 1;
    }
    t /= f32(n);
    t *= 2.0;
    //t += hash(vec3<i32>(p * 256.0));
    return t;
}
