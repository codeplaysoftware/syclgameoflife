# SYCL Game of Life

An implementation of Conway's Game of Life which makes use of hierarchical kernels and local memory in SYCL.

## Dependencies

The code compiles with DPC++. Installation instructions can be found [here](https://intel.github.io/llvm-docs/GetStartedGuide.html).

If compiling for the CUDA backend, the [CUDA runtime](https://intel.github.io/llvm-docs/GetStartedGuide.html#build-dpc-toolchain-with-support-for-nvidia-cuda) must be installed.

The DPC++ OpenCL backend requires an [OpenCL runtime](https://intel.github.io/llvm-docs/GetStartedGuide.html#install-low-level-runtime).

Graphics are rendered with SDL2. SDL can be installed with apt: `sudo apt install libsdl2-dev`.

## Build Instructions

Build configuration is carried out using CMake. 

The option -DSYCL_BACKEND allows you to select which backend to build for ("spir", "cuda" or "hip"). By default, it builds for spir. 

When enabled, the -DBENCHMARK_MODE option builds a headless version which can be run without SDL. 

The code has been tested on OpenCL and CUDA backends (both on devices running Ubuntu). The user also has the option to compile for HIP, although this hasn't been tested. 

```
$ git clone https://github.com/codeplaysoftware/syclgameoflife.git
$ cd syclgameoflife
$ mkdir build && cd build
$ cmake -DCMAKE_CXX_COMPILER=path/to/llvm/build/bin/clang++ -DBENCHMARK_MODE=off -DSYCL_BACKEND=spir ..
$ cmake --build .
$ ./GoL
```

## Configuration

Users can change the work-group size, the render scale and the dimensions of the life board by altering the constexprs at the top of `main.cpp`

See the following guide for more information on how to select the best work-group size: https://codeplay.com/portal/blogs/2020/01/09/sycl-performance-post-choosing-a-good-work-group-size-for-sycl.html
