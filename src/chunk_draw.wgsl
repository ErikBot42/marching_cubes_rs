


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
    ssao_radius: f32,

    world_clipw_cull: mat4x4<f32>,

    fog_color: vec3<f32>,
    diffuse_color: vec3<f32>,
    specular_color: vec3<f32>,
    light_color: vec3<f32>,
    sun_color: vec3<f32>,

    base_offset: vec3<i32>,
    view_clipw: mat4x4<f32>,
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
    return shade(in.view_pos, 1.0);
}

@fragment
fn fs_depth(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

struct DeferredVertexOutput {
    @builtin(position) builtin_clip_pos: vec4<f32>,
    @location(0) clip_pos: vec4<f32>,
}

@vertex
fn vs_deferred(@builtin(vertex_index) vertex_index: u32) -> DeferredVertexOutput {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    let clip_pos = vec4<f32>(pos[vertex_index], 0.99, 1.0);
    var out: DeferredVertexOutput;
    out.builtin_clip_pos = clip_pos;
    out.clip_pos = clip_pos;
    return out;
}

@group(1) @binding(0) var depth_texture: texture_depth_2d;

fn xorshift3(x0: u32) -> u32 {
    var x: u32 = x0;
    x ^= x >> 17;
    x ^= x << 5;
    x ^= x << 13;
    return x;
}

fn xorshift2(x0: u32) -> u32 {
    var x: u32 = x0;
    x ^= x << 13;
    x ^= x << 5;
    x ^= x >> 17;
    return x;
}

fn xorshift(x0: u32) -> u32 {
    var x: u32 = x0;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// xy screen space -> xyzw view space
fn lookup_coord(c: vec2<i32>, depth: f32) -> vec3<f32> {
    let uv = vec2<f32>(c) / vec2<f32>(textureDimensions(depth_texture));
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let view_posw = camera.clipw_view * clip;

    return view_posw.xyz / view_posw.w;
}

fn share_y(polarity: u32, val: vec3<f32>) -> vec3<f32> {
    var v = val;
    if (polarity & 1u) == 0u {
        v *= -1.0;
    }
    let dv = dpdy(v); // v0 + v1
    return dv - val;
}
fn share_x(polarity: u32, val: vec3<f32>) -> vec3<f32> {
    var v = val;
    if (polarity & 1u) == 0u {
        v *= -1.0;
    }
    let dv = dpdx(v); // v0 + v1
    return dv - val;
}

@fragment
fn fs_deferred(in: DeferredVertexOutput) -> @location(0) vec4<f32> {
    let screen_coords = vec2<i32>(floor(in.builtin_clip_pos.xy));

    var depth = textureLoad(depth_texture, screen_coords, 0);
    let cull = depth >= 1.0;
    if cull { depth = 0.33; }
    

    var clip = in.clip_pos; clip.z = depth;

    let view_posw = camera.clipw_view * clip;
    var view_pos = view_posw.xyz / view_posw.w;
    if cull {

        //view_pos = -normalize(view_pos);
        let view_dir = normalize(view_pos);
        return vec4<f32>(background(-view_dir), 1.0);
    }

    let ddx = dpdx(view_pos);
    let ddy = dpdy(view_pos);
    let normal = -normalize(cross(ddx, ddy));
    let dx = (ddx);
    let dy = (ddy);

    var cc = 0.0;
    let samples = 4u;
    let rad = camera.ssao_radius;

    // SSAO sampling, blue noise.

    let c = screen_coords;
    var cx = u32(c.x);
    var cy = u32(c.y);
    let state_base = xorshift(cy | (cx << 16)) ^ u32(fract(camera.time) * 37485.348589);
    cc = 0.0;
    {
        // sampling kernel constant per frame.
        
        let angle_prop = 6.283185307179586 / f32(samples);
        for (var i = 0u; i < samples; i = i + 1) {
            

            let state = xorshift(
                (cy | (cx << 16)) ^ 
                xorshift((i << 16u) ^ u32(fract(camera.time) * 37485.348589))
            );

            let angle = angle_prop * f32(i);//(state>>16) % 0xFFFF);
            //let angle = f32(i);
            let radius = (f32(state % 0xFFFF) % rad) * (0.9 / view_pos.z);
            //let radius = (f32((i+1)) / f32(samples))  * (f32(state % 0xFFFF) % rad) * (0.9 / view_pos.z);
            

            let disc_sample = radius * vec2<f32>(sin(angle), cos(angle));

            let p = c + vec2<i32>(disc_sample);

            let s = textureLoad(depth_texture, p, 0);

            // fn lookup_coord(c: vec2<i32>, depth: f32) -> vec3<f32> {
            //     let uv = vec2<f32>(c) / vec2<f32>(textureDimensions(depth_texture));
            //     let clip = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
            //     let view_posw = camera.clipw_view * clip;

            //     return view_posw.xyz / view_posw.w;
            // }

            let uv = vec2<f32>(p) / vec2<f32>(textureDimensions(depth_texture));
            let clip = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, s, 1.0);
            let sample_vieww = camera.clipw_view * clip;
            let sample_view = sample_vieww.xyz / sample_vieww.w;

            //let view2sample = normalize(view_pos - lookup_coord(p, s));
            //let view2sample = normalize(view_pos - sample_view);
            //let view2sample = normalize(view_pos - sample_vieww.xyz / sample_vieww.w);
            //let view2sample = normalize(view_pos - sample_vieww.xyz / abs(sample_vieww.w));
            let view2sample = normalize(view_pos  * abs(sample_vieww.w) - sample_vieww.xyz);

            let d = dot(view2sample, normal);
            if d < 0.0 { 
                cc -= d;
            }


            // {
            //     let disc_sample_view = view_pos + (disc_sample.x * dx + disc_sample.y * dy) * 1;
            //     
            //     let disc_sample_clipw = camera.view_clipw * vec4<f32>(disc_sample_view, 1.0);
            //     let disc_sample_clip = disc_sample_clipw.xyz / disc_sample_clipw.w;

            //     let uv: vec2<f32> = vec2<f32>((disc_sample_clip.x + 1.0) * 0.5, (disc_sample_clip.y + 1.0) * 0.5 - 1.0);
            //     let disc_sample_coord = vec2<i32>(floor(uv * vec2<f32>(textureDimensions(depth_texture))));

            //     let p = c + vec2<i32>(disc_sample);

            //     let s = textureLoad(depth_texture, p, 0);

            //     {
            //         let view2sample = normalize(view_pos - lookup_coord(p, s));
            //         if dot(view2sample, normal) > -0.01 { cc += 1.0; }
            //     }
            // }
        }
    }

    
    // let parity_x = 2 * (screen_coords.x & 1) - 1;
    // let parity_y = 2 * (screen_coords.y & 1) - 1;


    let ssao = pow(saturate((f32(samples) - cc) / f32(samples)), 4.0);//pow(cc / f32(samples), 2.0);
    //let ssao = pow(saturate((f32(samples) - cc) / f32(samples)), 2.0);//pow(cc / f32(samples), 2.0);
    
    //return vec4<f32>(normalize(view_pos), 1.0);
    //return vec4<f32>(ssao, 0.0, 0.0, 1.0);
    //return vec4<f32>(0.1/depth, 0.0, 0.0, 1.0);
    return shade(view_pos, ssao);
}

@fragment
fn fs_background(in: DeferredVertexOutput) -> @location(0) vec4<f32> {
    var clip = in.clip_pos;
    clip.z = 0.34;
    let view_posw = camera.clipw_view * clip;
    let view_pos = view_posw.xyz / view_posw.w;
    return vec4<f32>(background(normalize(view_pos)), 1.0);
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

fn shade(p_view0: vec3<f32>, ssao: f32) -> vec4<f32> {

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
    let color = acc * ssao;

    let fog_factor = pow(min(dot(p_view0, p_view0) * camera.fog_inv, 1.0), 2.0);

    //let dir = camera.view_world
    
    let interpolated = mix(color, background(view_dir), fog_factor);

    //return vec4<f32>(background(view_dir), 1.0);
    return vec4<f32>(interpolated, 1.0);
}

fn background(view_dir: vec3<f32>) -> vec3<f32> {
    let world_dir = (camera.view_world * vec4<f32>(view_dir, 0.0)).xyz;
    let light_sky = camera.fog_color * (0.4 + 0.6 * saturate(0.2 + dot(-world_dir, vec3<f32>(0.0, 1.0, 0.0))));
    let light_sun = vec3<f32>(0.98, 0.77, 0.40) * pow(saturate(
        -0.5 + dot(-world_dir, normalize(vec3<f32>(1.0, 1.0, 1.0))
    )), 4.0) * (1.0 / (1.0 - 0.7));
    return 3.0 * light_sun + light_sky * 1.9;
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
