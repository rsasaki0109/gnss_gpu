cmake_minimum_required(VERSION 3.24)
project(pf_scan_cache_probe LANGUAGES CXX)
# This optional diagnostic targets the rejected scan-cache variant. Reconstruct
# its recorded candidate header before configuring this target.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
pybind11_add_module(_pf_scan_cache_probe ../pf_scan_cache_probe.cpp)
target_include_directories(_pf_scan_cache_probe PRIVATE ../../../include)
target_link_libraries(_pf_scan_cache_probe PRIVATE CUDA::cudart)
target_compile_features(_pf_scan_cache_probe PRIVATE cxx_std_20)
if(MSVC)
  target_compile_options(_pf_scan_cache_probe PRIVATE /utf-8)
endif()
