cmake_minimum_required(VERSION 3.24)
project(rbpf_compiled_moves LANGUAGES CXX)
find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Module)
if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "This experimental probe requires Windows x64 CPython 3.12")
endif()
if(NOT DEFINED CYTHON_ROOT)
    message(FATAL_ERROR "Pass an explicit isolated Cython 3.1.4 package directory")
endif()
set(generated "${CMAKE_CURRENT_BINARY_DIR}/_rbpf_compiled_moves.cpp")
add_custom_command(OUTPUT "${generated}"
    COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${CYTHON_ROOT}" "PYTHONDONTWRITEBYTECODE=1"
        "${Python3_EXECUTABLE}" -B -m cython --cplus -3
        -o "${generated}" "${CMAKE_CURRENT_SOURCE_DIR}/_rbpf_compiled_moves.pyx"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/_rbpf_compiled_moves.pyx"
    VERBATIM)
Python3_add_library(_rbpf_compiled_moves MODULE WITH_SOABI "${generated}")
target_compile_features(_rbpf_compiled_moves PRIVATE cxx_std_17)
target_compile_options(_rbpf_compiled_moves PRIVATE /O2 /fp:strict /W3 /bigobj)
