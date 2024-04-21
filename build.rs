use std::env;
use std::fs;
use std::path::Path;
use std::time::SystemTime;

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("compile_time.rs");
    fs::write(
        &dest_path,
        format!(
            "const COMPILE_TIME: Duration = std::time::Duration::from_millis({}_u64);",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ),
    )
    .unwrap();
    println!("cargo::rerun-if-changed=src/chunk_draw.wgsl");
    println!("cargo::rerun-if-changed=build.rs");
}
