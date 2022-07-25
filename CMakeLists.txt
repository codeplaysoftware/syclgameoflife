cmake_minimum_required (VERSION 3.16.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project (GameOfLife)

set(SYCL_BACKEND "SYCL_BACKEND" CACHE STRING "Backend chosen by the user at CMake configure time")
set_property(CACHE SYCL_BACKEND PROPERTY STRINGS spir cuda hip)

if(SYCL_BACKEND STREQUAL hip)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=amdgcn-amd-amdhsa")
elseif(SYCL_BACKEND STREQUAL cuda)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda")
else()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=spir64")
endif()

OPTION(BENCHMARK_MODE "Enable benchmarking" off)
if(BENCHMARK_MODE)
    add_definitions(-DBENCHMARK_MODE)
else()
    find_package(SDL2 REQUIRED)
    include_directories(${SDL2_INCLUDE_DIRS})
endif()

add_executable(GoL src/main.cpp)
if(NOT BENCHMARK_MODE)
    target_link_libraries(GoL ${SDL2_LIBRARIES}) 
endif()
