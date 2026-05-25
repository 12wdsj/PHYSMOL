cmake_minimum_required(VERSION 3.16)
project(physmol LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Find pybind11
find_package(pybind11 REQUIRED)

# Compiler flags
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Detect SIMD support
include(CheckCCompilerFlag)
check_c_compiler_flag("-mavx2" HAS_AVX2)
if(HAS_AVX2)
    add_compile_options(-mavx2 -mfma)
    message(STATUS "AVX2+FMA SIMD enabled")
else()
    message(STATUS "AVX2 not available, using scalar fallback")
endif()

# === C Core Libraries ===

# VSA core
add_library(vsa_core STATIC
    src/core/vsa.c
)
target_include_directories(vsa_core PUBLIC src/core)

# SNN core
add_library(snn_core STATIC
    src/core/snn.c
    src/core/causal.c
)
target_include_directories(snn_core PUBLIC src/core)

# LNN core
add_library(lnn_core STATIC
    src/core/lnn.c
)
target_include_directories(lnn_core PUBLIC src/core)

# Memory core
add_library(memory_core STATIC
    src/core/memory.c
)
target_include_directories(memory_core PUBLIC src/core)

# === pybind11 Modules ===

# VSA Python extension
pybind11_add_module(_vsa src/bindings/vsa_py.cpp)
target_link_libraries(_vsa PRIVATE vsa_core)
target_include_directories(_vsa PRIVATE src)

# SNN Python extension
pybind11_add_module(_snn src/bindings/snn_py.cpp)
target_link_libraries(_snn PRIVATE snn_core)
target_include_directories(_snn PRIVATE src)

# LNN Python extension
pybind11_add_module(_lnn src/bindings/lnn_py.cpp)
target_link_libraries(_lnn PRIVATE lnn_core memory_core)
target_include_directories(_lnn PRIVATE src)

# === Install ===
install(TARGETS _vsa _snn _lnn
    LIBRARY DESTINATION src/python/physmol
)
