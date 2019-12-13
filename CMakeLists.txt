#
# CMakeLists.txt for software package MUGIQ
#
# Author: C. Kallidonis, William & Mary
# Dec. 2019

# basic setup for cmake
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set Disable-in-source-builds to ON in order to avoid overwriting the existing makefiles
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# Directory for cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#--------------------------------------------------------------

# Project name: MUGIQ
project("MUGIQ" VERSION 0.1.0 LANGUAGES)

message(STATUS "")
message(STATUS "${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "cmake version: ${CMAKE_VERSION}")
message(STATUS "Source location: ${CMAKE_SOURCE_DIR}")
message(STATUS "Build location: ${CMAKE_BINARY_DIR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
#--------------------------------------------------------------

# Host system CPU architecture TODO: Rename to MUGIQ_CPU_ARCH
set(CPU_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})

# Host system GPU architecture
if(DEFINED ENV{MUGIQ_GPU_ARCH})
  set(MUGIQ_DEFAULT_GPU_ARCH $ENV{MUGIQ_GPU_ARCH})
else()
  set(MUGIQ_DEFAULT_GPU_ARCH sm_35)
endif()
if(NOT MUGIQ_GPU_ARCH)
  message(STATUS "Building MUGIQ for GPU-architecture " "${MUGIQ_DEFAULT_GPU_ARCH}")
endif()
message(STATUS "")

set(MUGIQ_GPU_ARCH
    ${MUGIQ_DEFAULT_GPU_ARCH}
    CACHE STRING "Set the GPU architecture (sm_20, sm_21, sm_30, sm_35, sm_37, sm_50, sm_52, sm_60, sm_70, sm_75)")
set_property(CACHE MUGIQ_GPU_ARCH PROPERTY STRINGS sm_20 sm_21 sm_30 sm_35 sm_37 sm_50 sm_52 sm_60 sm_70 sm_75)
#--------------------------------------------------------------


# Package Build Options
# QIO
set(MUGIQ_QIO OFF CACHE BOOL "Link with QIO library")
set(MUGIQ_QIO_HOME "" CACHE PATH "path to QIO")

# MAGMA
set(MUGIQ_MAGMA OFF CACHE BOOL "Link with MAGMA Library")
set(MUGIQ_MAGMA_HOME "" CACHE PATH "path to MAGMA, if not set, pkg-config will be attempted")
set(MUGIQ_MAGMA_LIBS "" CACHE STRING "additional linker flags required to link against magma")

# PRIMME
set(MUGIQ_PRIMME OFF CACHE BOOL "Link with PRIMME Library")
set(MUGIQ_PRIMME_HOME "" CACHE PATH "path to MAGMA, if not set, pkg-config will be attempted")

# Multi-GPU options
set(MUGIQ_QMP OFF CACHE BOOL "Build the QMP multi-GPU code")
set(MUGIQ_QMP_HOME "" CACHE PATH "path to QMP")
set(MUGIQ_MPI OFF CACHE BOOL "Build the MPI multi-GPU code")

# Whether to build all tests
set(MUGIQ_BUILD_ALL_TESTS ON CACHE BOOL "build tests by default")

# OpenMP and CXX Standards
set(MUGIQ_CXX_STANDARD 14 CACHE STRING "set the CXX Standard (11, 14 or 17)")
set_property(CACHE MUGIQ_CXX_STANDARD PROPERTY STRINGS 11 14 17)
set(CMAKE_CXX_STANDARD ${MUGIQ_CXX_STANDARD})

# Whether to use CXX Flags on CUDA Host compiler
# TODO: Rename to MUGIQ_USE_CXX_FLAGS_WITH_CUDA
set(MUGIQ_PROPAGATE_CXX_FLAGS ON CACHE BOOL "use CXX_FLAGS with CUDA host compiler (for cmake >= 3.8)")

# define CXX FLAGS
set(CMAKE_CXX_FLAGS_STANDARD
    "${OpenMP_CXX_FLAGS} -g -O3 -Wall"
    CACHE STRING "Flags used by the C++ compiler for standard builds.")
enable_language(CXX)

# define C FLAGS
set(CMAKE_C_FLAGS_STANDARD
    "-g -O3 -Wall"
    CACHE STRING "Flags used by the C compiler during standard builds.")
enable_language(C)


# TODO: MPI / QMP options






# TODO: Set this if MUGIQ_QIO is set to ON
set(MUGIQ_LIMEHOME "" CACHE PATH "path to LIME")

