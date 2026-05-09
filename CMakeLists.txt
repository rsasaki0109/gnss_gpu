cmake_minimum_required(VERSION 3.18)
project(gnss_gpu LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 QUIET)

# --- Core library (positioning + coordinates) ---
add_library(gnss_gpu_core
  src/utils/coordinates.cu
  src/positioning/wls.cu
  src/positioning/fgo.cu
  src/positioning/multi_gnss.cu
)
target_include_directories(gnss_gpu_core PUBLIC include)
target_link_libraries(gnss_gpu_core CUDA::cudart)

# --- Ray tracing module ---
add_library(gnss_gpu_raytrace
  src/raytrace/raytrace.cu
)
target_include_directories(gnss_gpu_raytrace PUBLIC include)
target_link_libraries(gnss_gpu_raytrace CUDA::cudart)

# --- BVH-accelerated ray tracing module ---
add_library(gnss_gpu_bvh
  src/raytrace/bvh.cu
)
target_include_directories(gnss_gpu_bvh PUBLIC include)
target_link_libraries(gnss_gpu_bvh CUDA::cudart)

# --- Multipath simulation module ---
add_library(gnss_gpu_multipath
  src/multipath/multipath.cu
)
target_include_directories(gnss_gpu_multipath PUBLIC include)
target_link_libraries(gnss_gpu_multipath CUDA::cudart)

# --- Skyplot / vulnerability map module ---
add_library(gnss_gpu_skyplot
  src/skyplot/skyplot.cu
)
target_include_directories(gnss_gpu_skyplot PUBLIC include)
target_link_libraries(gnss_gpu_skyplot CUDA::cudart)

# --- Signal acquisition module (requires cuFFT) ---
add_library(gnss_gpu_acquisition
  src/acquisition/acquisition.cu
)
target_include_directories(gnss_gpu_acquisition PUBLIC include)
target_link_libraries(gnss_gpu_acquisition CUDA::cudart CUDA::cufft)

# --- Interference detection module (requires cuFFT) ---
add_library(gnss_gpu_interference
  src/interference/interference.cu
)
target_include_directories(gnss_gpu_interference PUBLIC include)
target_link_libraries(gnss_gpu_interference CUDA::cudart CUDA::cufft)

# --- Vector tracking module ---
add_library(gnss_gpu_tracking
  src/tracking/tracking.cu
)
target_include_directories(gnss_gpu_tracking PUBLIC include)
target_link_libraries(gnss_gpu_tracking CUDA::cudart gnss_gpu_core)

# --- Signal simulator module (requires cuRAND) ---
add_library(gnss_gpu_signal_sim
  src/signal_sim/signal_sim.cu
)
target_include_directories(gnss_gpu_signal_sim PUBLIC include)
target_link_libraries(gnss_gpu_signal_sim CUDA::cudart CUDA::curand)

# --- Particle filter module (requires cuRAND) ---
add_library(gnss_gpu_pf
  src/particle_filter/predict.cu
  src/particle_filter/weight.cu
  src/particle_filter/resampling.cu
)
target_include_directories(gnss_gpu_pf PUBLIC include)
target_link_libraries(gnss_gpu_pf CUDA::cudart CUDA::curand)

# --- SVGD module (Stein Variational Gradient Descent for particle filter) ---
add_library(gnss_gpu_svgd
  src/particle_filter/svgd.cu
)
target_include_directories(gnss_gpu_svgd PUBLIC include)
target_link_libraries(gnss_gpu_svgd CUDA::cudart CUDA::curand)

# --- 3D-aware particle filter (raytrace + PF) ---
add_library(gnss_gpu_pf3d
  src/particle_filter/weight_3d.cu
)
target_include_directories(gnss_gpu_pf3d PUBLIC include)
target_link_libraries(gnss_gpu_pf3d CUDA::cudart)

# --- BVH-accelerated 3D-aware particle filter ---
add_library(gnss_gpu_pf3d_bvh
  src/particle_filter/weight_3d_bvh.cu
)
target_include_directories(gnss_gpu_pf3d_bvh PUBLIC include)
target_link_libraries(gnss_gpu_pf3d_bvh CUDA::cudart)

# --- Device-resident particle filter (CUDA streams + pinned memory) ---
add_library(gnss_gpu_pf_device
  src/particle_filter/pf_device.cu
)
target_include_directories(gnss_gpu_pf_device PUBLIC include)
target_link_libraries(gnss_gpu_pf_device CUDA::cudart CUDA::curand)

# --- Ephemeris module ---
add_library(gnss_gpu_ephemeris
  src/ephemeris/ephemeris.cu
)
target_include_directories(gnss_gpu_ephemeris PUBLIC include)
target_link_libraries(gnss_gpu_ephemeris CUDA::cudart)

# --- Atmosphere module (tropospheric + ionospheric corrections) ---
add_library(gnss_gpu_atmosphere
  src/atmosphere/atmosphere.cu
)
target_include_directories(gnss_gpu_atmosphere PUBLIC include)
target_link_libraries(gnss_gpu_atmosphere CUDA::cudart)

# --- RAIM / FDE module ---
add_library(gnss_gpu_raim
  src/raim/raim.cu
)
target_include_directories(gnss_gpu_raim PUBLIC include)
target_link_libraries(gnss_gpu_raim CUDA::cudart gnss_gpu_core)

# --- EKF module ---
add_library(gnss_gpu_ekf
  src/ekf/ekf.cu
)
target_include_directories(gnss_gpu_ekf PUBLIC include)
target_link_libraries(gnss_gpu_ekf CUDA::cudart)

# --- Doppler velocity module ---
add_library(gnss_gpu_doppler
  src/doppler/doppler.cu
)
target_include_directories(gnss_gpu_doppler PUBLIC include)
target_link_libraries(gnss_gpu_doppler CUDA::cudart)

# --- RTK module ---
add_library(gnss_gpu_rtk
  src/rtk/rtk.cu
)
target_include_directories(gnss_gpu_rtk PUBLIC include)
target_link_libraries(gnss_gpu_rtk CUDA::cudart)

# --- Python bindings ---
if(pybind11_FOUND)
  # Original bindings (positioning + coordinates)
  pybind11_add_module(_gnss_gpu python/gnss_gpu/_bindings.cpp)
  target_link_libraries(_gnss_gpu PRIVATE gnss_gpu_core)

  # Multi-GNSS bindings (ISB estimation)
  pybind11_add_module(_gnss_gpu_multi_gnss python/gnss_gpu/_multi_gnss_bindings.cpp)
  target_link_libraries(_gnss_gpu_multi_gnss PRIVATE gnss_gpu_core)

  # Ray tracing bindings
  pybind11_add_module(_raytrace python/gnss_gpu/_raytrace_bindings.cpp)
  target_link_libraries(_raytrace PRIVATE gnss_gpu_raytrace)

  # Multipath bindings
  pybind11_add_module(_gnss_gpu_multipath python/gnss_gpu/_multipath_bindings.cpp)
  target_link_libraries(_gnss_gpu_multipath PRIVATE gnss_gpu_multipath)

  # Skyplot bindings
  pybind11_add_module(_gnss_gpu_skyplot python/gnss_gpu/_skyplot_bindings.cpp)
  target_link_libraries(_gnss_gpu_skyplot PRIVATE gnss_gpu_skyplot)

  # Acquisition bindings
  pybind11_add_module(_gnss_gpu_acq python/gnss_gpu/_acquisition_bindings.cpp)
  target_link_libraries(_gnss_gpu_acq PRIVATE gnss_gpu_acquisition)

  # Interference bindings
  pybind11_add_module(_gnss_gpu_interference python/gnss_gpu/_interference_bindings.cpp)
  target_link_libraries(_gnss_gpu_interference PRIVATE gnss_gpu_interference)

  # Tracking bindings
  pybind11_add_module(_gnss_gpu_tracking python/gnss_gpu/_tracking_bindings.cpp)
  target_link_libraries(_gnss_gpu_tracking PRIVATE gnss_gpu_tracking)

  # Particle filter bindings
  pybind11_add_module(_gnss_gpu_pf python/gnss_gpu/_particle_filter_bindings.cpp)
  target_link_libraries(_gnss_gpu_pf PRIVATE gnss_gpu_pf)

  # SVGD bindings
  pybind11_add_module(_gnss_gpu_svgd python/gnss_gpu/_svgd_bindings.cpp)
  target_link_libraries(_gnss_gpu_svgd PRIVATE gnss_gpu_svgd)

  # Doppler bindings
  pybind11_add_module(_gnss_gpu_doppler python/gnss_gpu/_doppler_bindings.cpp)
  target_link_libraries(_gnss_gpu_doppler PRIVATE gnss_gpu_doppler)

  # RTK bindings
  pybind11_add_module(_gnss_gpu_rtk python/gnss_gpu/_rtk_bindings.cpp)
  target_link_libraries(_gnss_gpu_rtk PRIVATE gnss_gpu_rtk)

  # Ephemeris bindings
  pybind11_add_module(_gnss_gpu_ephemeris python/gnss_gpu/_ephemeris_bindings.cpp)
  target_link_libraries(_gnss_gpu_ephemeris PRIVATE gnss_gpu_ephemeris)

  # 3D-aware PF bindings
  pybind11_add_module(_gnss_gpu_pf3d python/gnss_gpu/_pf3d_bindings.cpp)
  target_link_libraries(_gnss_gpu_pf3d PRIVATE gnss_gpu_pf3d)

  # Atmosphere bindings
  pybind11_add_module(_gnss_gpu_atmosphere python/gnss_gpu/_atmosphere_bindings.cpp)
  target_link_libraries(_gnss_gpu_atmosphere PRIVATE gnss_gpu_atmosphere)

  # EKF bindings
  pybind11_add_module(_gnss_gpu_ekf python/gnss_gpu/_ekf_bindings.cpp)
  target_link_libraries(_gnss_gpu_ekf PRIVATE gnss_gpu_ekf)

  # RAIM / FDE bindings
  pybind11_add_module(_gnss_gpu_raim python/gnss_gpu/_raim_bindings.cpp)
  target_link_libraries(_gnss_gpu_raim PRIVATE gnss_gpu_raim)

  # BVH ray tracing bindings
  pybind11_add_module(_bvh python/gnss_gpu/_bvh_bindings.cpp)
  target_link_libraries(_bvh PRIVATE gnss_gpu_bvh)

  # BVH-accelerated 3D-aware PF bindings
  pybind11_add_module(_gnss_gpu_pf3d_bvh python/gnss_gpu/_pf3d_bvh_bindings.cpp)
  target_link_libraries(_gnss_gpu_pf3d_bvh PRIVATE gnss_gpu_pf3d_bvh)

  # Device-resident particle filter bindings (CUDA streams)
  pybind11_add_module(_gnss_gpu_pf_device python/gnss_gpu/_pf_device_bindings.cpp)
  target_link_libraries(_gnss_gpu_pf_device PRIVATE gnss_gpu_pf_device)

  # Signal simulator bindings
  pybind11_add_module(_gnss_gpu_signal_sim python/gnss_gpu/_signal_sim_bindings.cpp)
  target_link_libraries(_gnss_gpu_signal_sim PRIVATE gnss_gpu_signal_sim)

  # --- RTKLIB SPP wrapper (pure C, no CUDA) ---
  # Requires RTKLIB demo5 sources at ../ref/RTKLIB-demo5/src/
  set(RTKLIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ref/RTKLIB-demo5/src)
  if(EXISTS ${RTKLIB_DIR}/rtklib.h)
    set(RTKLIB_SRCS
      ${RTKLIB_DIR}/rtkcmn.c
      ${RTKLIB_DIR}/rinex.c
      ${RTKLIB_DIR}/pntpos.c
      ${RTKLIB_DIR}/ephemeris.c
      ${RTKLIB_DIR}/ionex.c
      ${RTKLIB_DIR}/tides.c
      ${RTKLIB_DIR}/sofa.c
      ${RTKLIB_DIR}/trace.c
      ${RTKLIB_DIR}/options.c
      ${RTKLIB_DIR}/solution.c
      ${RTKLIB_DIR}/lambda.c
      ${RTKLIB_DIR}/geoid.c
      ${RTKLIB_DIR}/sbas.c
      ${RTKLIB_DIR}/preceph.c
      ${RTKLIB_DIR}/rtkpos.c
      ${RTKLIB_DIR}/postpos.c
      ${RTKLIB_DIR}/ppp.c
      ${RTKLIB_DIR}/ppp_ar.c
      ${RTKLIB_DIR}/rtcm.c
      ${RTKLIB_DIR}/rtcm2.c
      ${RTKLIB_DIR}/rtcm3.c
      ${RTKLIB_DIR}/rtcm3e.c
    )
    add_library(rtklib_c STATIC ${RTKLIB_SRCS} src/rtklib_spp/rtklib_spp.c)
    target_include_directories(rtklib_c PUBLIC ${RTKLIB_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(rtklib_c PRIVATE TRACE ENAGLO ENAQZS ENAGAL ENACMP ENAIRN NFREQ=4 NEXOBS=3)
    target_compile_options(rtklib_c PRIVATE -std=c99 -Wno-unused-but-set-variable)
    set_target_properties(rtklib_c PROPERTIES LINKER_LANGUAGE C)

    pybind11_add_module(_gnss_gpu_rtklib_spp python/gnss_gpu/_rtklib_spp_bindings.cpp)
    target_link_libraries(_gnss_gpu_rtklib_spp PRIVATE rtklib_c m)
    target_include_directories(_gnss_gpu_rtklib_spp PRIVATE ${RTKLIB_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
    install(TARGETS _gnss_gpu_rtklib_spp DESTINATION gnss_gpu)
  else()
    message(STATUS "RTKLIB demo5 sources not found at ${RTKLIB_DIR} -- skipping rtklib_spp binding")
  endif()

  # Install all pybind11 modules into the gnss_gpu package
  install(TARGETS
    _gnss_gpu _gnss_gpu_multi_gnss _raytrace _gnss_gpu_multipath
    _gnss_gpu_skyplot _gnss_gpu_acq _gnss_gpu_interference
    _gnss_gpu_tracking _gnss_gpu_pf _gnss_gpu_svgd _gnss_gpu_pf3d
    _gnss_gpu_atmosphere _gnss_gpu_ekf _gnss_gpu_raim _bvh
    _gnss_gpu_pf3d_bvh _gnss_gpu_pf_device _gnss_gpu_doppler
    _gnss_gpu_rtk _gnss_gpu_ephemeris _gnss_gpu_signal_sim
    DESTINATION gnss_gpu
  )
endif()
