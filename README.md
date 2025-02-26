# Filtering-Based Reconstruction for Gradient-Domain Rendering

## Build

### Clone

Clone the repository and fetch all submodules recursively.

```
git clone https://github.com/lastmc/FRGR.git
cd FRGR
git submodule update --init --recursive
```

### Requirements

OpenCV with `contrib` and `openexr` module.

For example, OpenCV could be installed through `vcpkg` in Windows:
```
vcpkg install opencv[contrib,openexr]:x64-windows
```

### Build by CMake

```
cmake -S . -B build
cmake --build build
```
The built executable should be in folder `build/bin`
## Run

An example run script is:
```
build/bin/main.exe -i color.exr --image_pt color_pt.exr -v var.exr --var_pt var_pt.exr dx.exr dy.exr -a albedo.exr -n normal.exr --albedo_var albedoVariance.exr --normal_var normalVariance.exr --lambda 0.5 -A 1 -N 0.01 -r 3 -G 1 -l 2 -s 64 -d example-scenes/kitchen --output build/test.exr
```

An example scene is provided in `example-scenes/kitchen`. The example scene is 64spp, thus the folder path is `example-scenes/kitchen/64spp`. The `spp` parameter is provided by `-s`.

Set the `--output` flag to choose where to save the output file. Also, the enhanced gradients are outputed to `example-scenes/kitchen/outputs` in the provided example.
