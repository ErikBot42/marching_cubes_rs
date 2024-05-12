

fn sd_mandelbox_optim2(p00: vec3<f32>, scale: f32) -> f32 { 
    let iters = 20;
    let mr2 = 0.5 * 0.5;
    let scalevec = vec4<f32>(scale, scale, scale, abs(scale)) / mr2;
    let c1 = abs(scale- 1.0);
    let c2 = pow(abs(scale), 1.0 - f32(iters));

    var p = vec4<f32>(p00, 1.0);
    let p0 = p;
    for (var i=0; i<iters; i++) {
        p = vec4<f32>(clamp(p.xyz, - vec3<f32>(1.0), vec3<f32>(1.0)) * 2.0 - p.xyz, p.w);
        p = (p * clamp(max(mr2/dot(p.xyz, p.xyz), mr2), 0.0, 1.0)) * scalevec + p0; 
    }
    return (length(p.xyz) - c1) / p.w - c2;
}

fn sdf(x0: i32, y0: i32, z0: i32) -> f32 {
    // let x = (f32(x0) / 2.0);// % 30.0 - 8.0;
    // let y = (f32(y0) / 2.0);// % 30.0 - 8.0;
    // let z = (f32(z0) / 2.0);// % 30.0 - 8.0;

    let sc = 32.0 * 2.0;

    var pi = vec3<i32>(x0, y0, z0);


    let layer = pi.y / (4 * 32);
    pi %= 4 * 32;


    var p = (vec3<f32>(pi) - sc) / (sc * 2.0);

    var mandel_scale = -3.0 + (f32(layer % 9) * 0.5);
    var scale = 0.3; 
    if (mandel_scale > 0.0) {
        mandel_scale+=2.0; 
        scale/=(mandel_scale+1.0)/(mandel_scale- 1.0);
    } else {
        mandel_scale-=1.0;
    }
    return sd_mandelbox_optim2(p / scale, mandel_scale) * scale - 0.001;
    //let p = vec3<f32>(vec3<i32>(x0, y0, z0) - 32);

    //return -sd_mandelbulb3(p * (1.0 / 32.0));


/*
    let rad = 4 * 32;

    let p = vec3<f32>(vec3<i32>(x0, y0, z0)) * (1.0/f32(32));
    let l = length(p);
    let f = 1.75;

    if false {

    } else if l < 1.15 {
        return -sd_mandelbulb3(p);
    } else if l < 1.15 *       f {
        return  sd_mandelbulb3(p / f);
    } else if l < 1.15 *       f * f {
        return -sd_mandelbulb3(p / f / f);
    } else if l < 1.15 *       f * f * f {
        return  sd_mandelbulb3(p / f / f / f);
    } else if l < 1.15 *       f * f * f * f {
        return -sd_mandelbulb3(p / f / f / f / f);
    } else if l < 1.15 *       f * f * f * f * f {
        return  sd_mandelbulb3(p / f / f / f / f / f);
    } else if l < 1.15 *       f * f * f * f * f * f {
        return -sd_mandelbulb3(p / f / f / f / f / f / f);
    } else if l < 1.15 *       f * f * f * f * f * f * f{
        return  sd_mandelbulb3(p / f / f / f / f / f / f / f);
    } else {

    }

    return -1.0;
*/
}

fn sd_mandelbulb3(p: vec3<f32>) -> f32 {
    let iterations = 8;
    let power = 8.0;

    let half_power = (power - 1.0) * 0.5;
    let bailout = pow(2.0, power);

    var z = p;
    var r2 = dot(z, z);
    var dz = 1.0;

    for (var i: i32 = 0; i < iterations; i++) {
        if (r2 > bailout) { break; }

        dz *= power * pow(r2, half_power);
        dz += 1.0;
        let r = length(z);
        let theta = power * acos(z.z / r);
        let phi = power * atan2(z.y, z.x);
        z = p + pow(r, power) * 
            vec3<f32>(
                    sin(theta) * cos(phi),
                    sin(theta) * sin(phi), 
                    cos(theta) 
                    );

        r2 = dot(z, z);
    }

    return 0.25 * log(r2) * sqrt(r2) / dz;
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
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i: i32 = i32(global_id.x);
    let pos: vec3<i32> = vec3<i32>(i % 33, (i / 33) % 33, i / (33 * 33)) + chunk.pos * 32;
    let s: f32 = sdf(pos.x, pos.y, pos.z);
    sdf_data[i] = s;
}

