cmake_minimum_required(VERSION 3.18)
project(gnss_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 QUIET)

add_library(gnss_gpu_core
  src/dummy.cu
)
target_include_directories(gnss_gpu_core PUBLIC include)
target_link_libraries(gnss_gpu_core CUDA::cudart)

if(pybind11_FOUND)
  pybind11_add_module(_gnss_gpu python/gnss_gpu/_bindings.cpp)
  target_link_libraries(_gnss_gpu PRIVATE gnss_gpu_core)
endif()
