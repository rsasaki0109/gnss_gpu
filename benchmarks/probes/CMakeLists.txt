cmake_minimum_required(VERSION 3.24)
project(gnss_gpu_benchmark_probes LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# A read-only probe for existing PF handles. No production PF code is linked
# or replaced; the actual baseline/candidate extension owns and runs the state.
pybind11_add_module(_pf_systematic_probe pf_systematic_probe.cpp)
target_include_directories(_pf_systematic_probe PRIVATE ../../include)
target_link_libraries(_pf_systematic_probe PRIVATE CUDA::cudart)
target_compile_features(_pf_systematic_probe PRIVATE cxx_std_20)
if(MSVC)
  target_compile_options(_pf_systematic_probe PRIVATE /utf-8)
endif()
