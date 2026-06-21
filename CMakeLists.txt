cmake_minimum_required(VERSION 3.20)
project(blackwell_moe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "89;90;100" CACHE STRING "CUDA architectures")

option(BUILD_TESTS "Build unit tests" ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include_directories(include)

file(GLOB_RECURSE MOE_SRC "src/*.cpp" "csrc/cuda/*.cu")

add_library(blackwell_moe SHARED ${MOE_SRC})
target_include_directories(blackwell_moe PUBLIC include)
target_compile_options(blackwell_moe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3 --use_fast_math -lineinfo --ptxas-options=-v
    >
)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
