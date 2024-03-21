{ pkgs ? import <nixpkgs> { }, lib ? pkgs.lib }:
pkgs.mkShell rec {
name = "rust-env";
buildInputs = with pkgs; [
linuxPackages.perf
cmake
gcc
mesa
clang
openssl
pkg-config
git

xorg.libX11
xorg.libXcursor
xorg.libXrandr
xorg.libXi

libxkbcommon
#libGl

# alsaLib
# freetype
# expat
freetype
glfw

vulkan-tools
vulkan-loader
vulkan-validation-layers
vulkan-tools-lunarg
vulkan-extension-layer

wayland

shaderc
shaderc.bin
shaderc.static
shaderc.dev
shaderc.lib
];
LD_LIBRARY_PATH = "${lib.makeLibraryPath buildInputs}";
VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
XDG_DATA_DIRS = builtins.getEnv "XDG_DATA_DIRS";
XDG_RUNTIME_DIR = builtins.getEnv "XDG_RUNTIME_DIR";
}
