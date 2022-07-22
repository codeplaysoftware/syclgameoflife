# SYCL Game of Life

An implementation of Conway's Game of Life which makes use of hierarchical kernels and local memory in SYCL.

## Build Instructions

The code compiles with [DPC++](https://intel.github.io/llvm-docs/GetStartedGuide.html)

Graphics are rendered with SDL2

```
$ git clone https://github.com/codeplaysoftware/syclgameoflife.git
$ cd syclgameoflife
$ mkdir build && cd build
$ cmake -DCMAKE_CXX_COMPILER=path/to/llvm/build/bin/clang++ -DBENCHMARK_MODE=off ..
$ cmake --build .
$ ./GoL
```


