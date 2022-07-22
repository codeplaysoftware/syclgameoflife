cmake_minimum_required (VERSION 3.16.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project (GameOfLife)

OPTION(BENCHMARK_MODE "Enable benchmarking" off)
if(BENCHMARK_MODE)
    add_definitions(-DBENCHMARK_MODE)
else()
    find_package(SDL2 REQUIRED)
    include_directories(${SDL2_INCLUDE_DIRS})
endif()

add_executable(GoL src/main.cpp)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fsycl")

if(NOT BENCHMARK_MODE)
    target_link_libraries(GoL ${SDL2_LIBRARIES}) 
endif()
