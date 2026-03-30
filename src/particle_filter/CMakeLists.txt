# Particle filter CUDA sources - to be added to the root CMakeLists.txt
#
# Add the following to the root CMakeLists.txt:
#
# 1. Add particle filter sources to gnss_gpu_core:
#
#   target_sources(gnss_gpu_core PRIVATE
#     src/particle_filter/predict.cu
#     src/particle_filter/weight.cu
#     src/particle_filter/resampling.cu
#   )
#
# 2. Add pybind11 module for particle filter bindings:
#
#   if(pybind11_FOUND)
#     pybind11_add_module(_gnss_gpu_pf python/gnss_gpu/_particle_filter_bindings.cpp)
#     target_link_libraries(_gnss_gpu_pf PRIVATE gnss_gpu_core)
#   endif()
#

# Standalone usage (if included via add_subdirectory):
if(TARGET gnss_gpu_core)
  target_sources(gnss_gpu_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/predict.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/weight.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/resampling.cu
  )
endif()
