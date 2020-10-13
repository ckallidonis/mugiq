#
# CMakeLists.txt for software package MUGIQ
#
# C. Kallidonis, William & Mary
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
#--------------------------------------------------------------


# MPI/QMP options
# Use MPI Comms by default
if(DEFINED ENV{MUGIQ_COMMS_TYPE})
  set(DEFCOMMS $ENV{MUGIQ_COMMS_TYPE})
else()
  set(DEFCOMMS "MPI")
endif()


set(VALID_COMMS_TYPES MPI QMP)
set(CMAKE_COMMS_TYPE "${DEFCOMMS}" CACHE STRING "Type of communications, must be the same as QUDA library. Options are: ${VALID_COMMS_TYPES}")
set_property(CACHE CMAKE_COMMS_TYPE PROPERTY STRINGS MPI QMP)


string(TOUPPER ${CMAKE_COMMS_TYPE} CHECK_COMMS_TYPE)
list(FIND VALID_COMMS_TYPES ${CHECK_COMMS_TYPE} COMMS_TYPE_VALID)

if(COMMS_TYPE_VALID LESS 0)
  message(SEND_ERROR "A valid CMAKE_COMMS_TYPE type must be specified. Valid comms types are:" "${VALID_COMMS_TYPES}")
endif()
#--------------------------------------------------------------


# CMake Build Type (Only DEVEL for now...)
if(DEFINED ENV{MUGIQ_BUILD_TYPE})
  set(DEFBUILD $ENV{MUGIQ_BUILD_TYPE})
else()
  set(DEFBUILD "DEVEL")
endif()

set(VALID_BUILD_TYPES DEVEL)
set(CMAKE_BUILD_TYPE "${DEFBUILD}" CACHE STRING "The type of build, options are: ${VALID_BUILD_TYPES}")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS DEVEL)

string(TOUPPER ${CMAKE_BUILD_TYPE} CHECK_BUILD_TYPE)
list(FIND VALID_BUILD_TYPES ${CHECK_BUILD_TYPE} BUILD_TYPE_VALID)

if(BUILD_TYPE_VALID LESS 0)
  message(SEND_ERROR "A valid CMAKE_BUILD_TYPE type must be specified. Valid build types are:" "${VALID_BUILD_TYPES}")
endif()
#--------------------------------------------------------------



# Project name: MUGIQ
project("MUGIQ" VERSION 0.1.0 LANGUAGES)

message(STATUS "")
message(STATUS "${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "cmake version: ${CMAKE_VERSION}")
message(STATUS "Source location: ${CMAKE_SOURCE_DIR}")
message(STATUS "Build location: ${CMAKE_BINARY_DIR}")
message(STATUS "Comms type: ${CMAKE_COMMS_TYPE}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

#--------------------------------------------------------------
#--------------------------------------------------------------


# Host system CPU architecture
set(CPU_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})

# Host system GPU architecture
if(DEFINED ENV{MUGIQ_GPU_ARCH})
  set(MUGIQ_DEFAULT_GPU_ARCH $ENV{MUGIQ_GPU_ARCH})
else()
  set(MUGIQ_DEFAULT_GPU_ARCH sm_70)
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
#--------------------------------------------------------------


# Package Build Options

set(BUILD_MUGIQ_CUDA_LIB OFF CACHE BOOL "Build the MuGiq Library" FORCE)

# QUDA (foremost)
set(MUGIQ_QUDA OFF CACHE BOOL "Whether to Link with QUDA library")
set(MUGIQ_QUDA_HOME "" CACHE PATH "Install path to QUDA")
set(MUGIQ_QUDA_SOURCE "" CACHE PATH "Source code path to QUDA")

# QIO / Lime
set(MUGIQ_QIO OFF CACHE BOOL "Link with QIO library")
set(MUGIQ_QIO_HOME "" CACHE PATH "path to QIO")
set(MUGIQ_LIME_HOME "" CACHE PATH "path to LIME")

# MAGMA
set(MUGIQ_MAGMA OFF CACHE BOOL "Link with MAGMA Library")
set(MUGIQ_MAGMA_HOME "" CACHE PATH "path to MAGMA, if not set, pkg-config will be attempted")
set(MUGIQ_MAGMA_LIBS "" CACHE STRING "additional linker flags required to link against magma")

# PRIMME
set(MUGIQ_PRIMME OFF CACHE BOOL "Link with PRIMME Library")
set(MUGIQ_PRIMME_HOME "" CACHE PATH "path to PRIMME, if not set, pkg-config will be attempted")

# QMP
set(MUGIQ_QMP_HOME "" CACHE PATH "path to QMP")

# HDF5
set(MUGIQ_HDF5 ON CACHE BOOL "Link with HDF5 Library")
set(MUGIQ_HDF5_HOME "" CACHE PATH "path to HDF5, if not set, pkg-config will be attempted")
set(MUGIQ_HDF5_LIBS "" CACHE STRING "additional linker flags required to link against HDF5")

# Whether to build all tests
set(MUGIQ_BUILD_ALL_TESTS ON CACHE BOOL "build tests by default")
#--------------------------------------------------------------
#--------------------------------------------------------------


# Compilers and Flags

# OpenMP and CXX Standards
set(MUGIQ_CXX_STANDARD 14 CACHE STRING "set the CXX Standard (11, 14 or 17)")
set_property(CACHE MUGIQ_CXX_STANDARD PROPERTY STRINGS 11 14 17)
set(CMAKE_CXX_STANDARD ${MUGIQ_CXX_STANDARD})

# Whether to use CXX Flags on CUDA Host compiler
set(MUGIQ_USE_CXX_FLAGS_WITH_CUDA ON CACHE BOOL "use CXX_FLAGS with CUDA host compiler (for cmake >= 3.8)")

# define CXX FLAGS
set(CMAKE_CXX_FLAGS_DEVEL "${OpenMP_CXX_FLAGS} -g -O3 -Wall" CACHE STRING "Flags used by the C++ compiler for DEVEL (standard) builds.")
enable_language(CXX)

# define C FLAGS
set(CMAKE_C_FLAGS_DEVEL "-g -O3 -Wall" CACHE STRING "Flags used by the C compiler during DEVEL (standard) builds.")
enable_language(C)

# Check for MPI C/C++ compilers
if((NOT MPI_CXX_COMPILER) AND DEFINED ENV{MPICXX})
  set(MPI_CXX_COMPILER $ENV{MPICXX})
  set(mpimessage True)
  message(STATUS "Found environment variable MPICXX. Using it for MPI detection: $ENV{MPICXX}")
endif()
if((NOT MPI_C_COMPILER) AND DEFINED ENV{MPICC})
  message(STATUS "Found environment variable MPICC. Using it for MPI detection: $ENV{MPICC}")
  set(MPI_C_COMPILER $ENV{MPICC})
  set(mpimessage True)
endif()

if(mpimessage)
  message("Found MPICC/MPICXX environment variables.")
endif()
find_package(MPI)
#--------------------------------------------------------------
#--------------------------------------------------------------


# Communication Options

# MPI Comms
if("${CMAKE_COMMS_TYPE}" STREQUAL "MPI")
  add_definitions(-DMULTI_GPU -DMPI_COMMS)
  include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
endif()

# QMP Comms
if("${CMAKE_COMMS_TYPE}" STREQUAL "QMP")
  find_library (LIB_QMP qmp HINTS ${MUGIQ_QMP_HOME}/lib NO_DEFAULT_PATH)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${LIB_QMP})

  if("${MUGIQ_QMP_HOME}" STREQUAL "")
    message(FATAL_ERROR "MUGIQ_QMP_HOME must be specified if CMAKE_COMMS_TYPE is set to QMP.")
  endif()
  execute_process(COMMAND ${MUGIQ_QMP_HOME}/bin/qmp-config --cflags  OUTPUT_VARIABLE MUGIQ_QMP_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${MUGIQ_QMP_HOME}/bin/qmp-config --ldflags OUTPUT_VARIABLE MUGIQ_QMP_LDFLAGS_INTERNAL OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${MUGIQ_QMP_HOME}/bin/qmp-config --libs    OUTPUT_VARIABLE MUGIQ_QMP_LIBS_INTERNAL OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(MUGIQ_QMP_LDFLAGS ${MUGIQ_QMP_LDFLAGS_INTERNAL} CACHE STRING "LDFLAGS for QMP - should be derived from qmp-config --ldflags")
  set(MUGIQ_QMP_LIBS ${MUGIQ_QMP_LIBS_INTERNAL} CACHE STRING "LIBS for QMP - should be derived from qmp-config --libs")
  
  add_definitions(-DMULTI_GPU -DQMP_COMMS)
  
  include_directories(SYSTEM ${MUGIQ_QMP_HOME}/include)
  include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
endif()
#--------------------------------------------------------------
#--------------------------------------------------------------


# Package/library-related options

# QUDA options
if(MUGIQ_QUDA)

  if(MUGIQ_QUDA_HOME STREQUAL "")
    message(FATAL_ERROR "MUGIQ_QUDA_HOME must be specified")
  endif()
  if(MUGIQ_QUDA_SOURCE STREQUAL "")
    message(FATAL_ERROR "MUGIQ_QUDA_SOURCE must be specified")
  endif()
  
  find_library (LIB_QUDA libquda.a HINTS ${MUGIQ_QUDA_HOME}/lib NO_DEFAULT_PATH)
  find_library (LIB_QUDATEST libquda_test.a HINTS ${MUGIQ_QUDA_HOME}/tests NO_DEFAULT_PATH)

  if(NOT LIB_QUDA)
    message(FATAL_ERROR "QUDA Library not found")
  endif()
  if(NOT LIB_QUDATEST)
    message(FATAL_ERROR "QUDA_TEST Library not found")
  endif()

  set(MUGIQ_QUDA_LDFLAGS "-L${MUGIQ_QUDA_HOME}/lib -L${MUGIQ_QUDA_HOME}/tests" CACHE STRING "LDFLAGS for QUDA library")
  set(MUGIQ_QUDA_LIBS "-lquda -lquda_test" CACHE STRING "LIBS for QUDA library")
  
  add_definitions(-DHAVE_QUDA)

  include_directories(SYSTEM ${MUGIQ_QUDA_HOME}/include)
  include_directories(SYSTEM ${MUGIQ_QUDA_HOME}/include/externals)
  include_directories(SYSTEM ${MUGIQ_QUDA_SOURCE}/tests)
  include_directories(SYSTEM ${MUGIQ_QUDA_SOURCE}/lib)

  set(BUILD_MUGIQ_CUDA_LIB ON CACHE BOOL "Build the MuGiq Library" FORCE)
endif()


# HDF5 options
if(MUGIQ_HDF5)  
  if("${MUGIQ_HDF5_HOME}" STREQUAL "")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIB_HDF5 hdf5)
    include_directories(SYSTEM ${HDF5_INCLUDEDIR})
    message("${HDF5_INCLUDEDIR}")
    find_library(LIB_HDF5 ${HDF5_LIBRARIES} PATH ${HDF5_LIBRARY_DIRS})
  else()
    find_library(LIB_HDF5 libhdf5.a hdf5 ${MUGIQ_HDF5_HOME}/lib) # Prefer Static library
  endif()
  
  if(NOT LIB_HDF5)
    message(FATAL_ERROR "HDF5 Library not found")
  endif()
  
  set(MUGIQ_HDF5_LDFLAGS "-L${MUGIQ_HDF5_HOME}/lib" CACHE STRING "LDFLAGS for HDF5 library")
  set(MUGIQ_HDF5_LIBS "-lhdf5" CACHE STRING "LIBS for HDF5 library")
  
  add_definitions(-DHDF5_LIB)
  
  include_directories(SYSTEM ${MUGIQ_HDF5_HOME}/include)
endif()


# QIO Options
if(MUGIQ_QIO)
  # If using QIO, we MUST compile with QMP (the same for QUDA.)
  if(NOT "${CMAKE_COMMS_TYPE}" STREQUAL "QMP")
    message(FATAL_ERROR "If MUGIQ_QIO is ON then QMP MUST be used. The same must hold for QUDA")
  endif()
  find_library (LIB_QIO qio HINTS ${MUGIQ_QIO_HOME}/lib NO_DEFAULT_PATH)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${LIB_QIO})
  
  if("${MUGIQ_QIO_HOME}" STREQUAL "")
    message(FATAL_ERROR "MUGIQ_QIO_HOME must be specified when MUGIQ_QIO is set to ON")
  endif()
  
  if(NOT "${MUGIQ_LIME_HOME}" STREQUAL "")
    find_library (LIB_LIME lime HINTS ${MUGIQ_LIME_HOME}/lib NO_DEFAULT_PATH)
    if(NOT LIB_LIME)
      message(FATAL_ERROR "LIME library not found inside ${MUGIQ_LIME_HOME}. Please specify correct MUGIQ_LIME_HOME")
    endif() 
    set(MUGIQ_LIME_HOME "${MUGIQ_LIME_HOME}" CACHE PATH "Path to LIME" FORCE)
  else()
    find_library (LIB_LIME lime HINTS ${MUGIQ_QIO_HOME}/lib NO_DEFAULT_PATH)
    if(LIB_LIME)
      set(MUGIQ_LIME_HOME "${MUGIQ_QIO_HOME}" CACHE PATH "Path to LIME" FORCE)
      if("${MUGIQ_LIME_HOME}" STREQUAL "")
        message(FATAL_ERROR "MUGIQ_LIME_HOME is not set")
      endif()
    elseif(NOT LIB_LIME)
      message(FATAL_ERROR "LIME library not found inside ${MUGIQ_QIO_HOME} as expected. Please specify MUGIQ_LIME_HOME")
    endif(LIB_LIME)
  endif()  
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${LIB_LIME})
  
  execute_process(COMMAND ${MUGIQ_QIO_HOME}/bin/qio-config --cflags  OUTPUT_VARIABLE MUGIQ_QIO_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${MUGIQ_QIO_HOME}/bin/qio-config --ldflags OUTPUT_VARIABLE MUGIQ_QIO_LDFLAGS_INTERNAL OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${MUGIQ_QIO_HOME}/bin/qio-config --libs    OUTPUT_VARIABLE MUGIQ_QIO_LIBS_INTERNAL OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(MUGIQ_QIO_LDFLAGS ${MUGIQ_QIO_LDFLAGS_INTERNAL} CACHE STRING "LDFLAGS for QIO - should be derived from qio-config --ldflags")
  set(MUGIQ_QIO_LIBS ${MUGIQ_QIO_LIBS_INTERNAL} CACHE STRING "LIBS for QIO - should be derived from qio-config --libs")
  
  add_definitions(-DHAVE_QIO)
  
  include_directories(SYSTEM ${MUGIQ_QIO_HOME}/include)
  include_directories(SYSTEM ${MUGIQ_LIME_HOME}/include)
endif(MUGIQ_QIO)


# MAGMA Options
if(MUGIQ_MAGMA)
  find_package(OpenMP)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  if("${MUGIQ_MAGMA_HOME}" STREQUAL "")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIB_MAGMA magma)
    include_directories(SYSTEM ${MAGMA_INCLUDEDIR})
    message("${MAGMA_INCLUDEDIR}")
    find_library(LIB_MAGMA ${MAGMA_LIBRARIES} PATH ${MAGMA_LIBRARY_DIRS})
    LIST(APPEND LIB_MAGMA ${BLAS_LIBRARIES})
    LIST(APPEND LIB_MAGMA ${LAPACK_LIBRARIES})
  else()
    find_library(LIB_MAGMA libmagma.a magma ${MUGIQ_MAGMA_HOME}/lib) # Prefer Static library

    # Additional libraries required by magma
    list(APPEND LIB_MAGMA ${CUDA_cublas_LIBRARY})
    list(APPEND LIB_MAGMA ${CUDA_cusparse_LIBRARY})
    list(APPEND LIB_MAGMA ${MUGIQ_MAGMA_LIBS})

    # Additional OpenMP linker flags
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
    include_directories(SYSTEM ${MUGIQ_MAGMA_HOME}/include)

    # add BLAS and LAPACK libraries
    LIST(APPEND LIB_MAGMA ${BLAS_LIBRARIES})
    LIST(APPEND LIB_MAGMA ${LAPACK_LIBRARIES})
  endif()

  add_definitions(-DMAGMA_LIB -DADD_ -DMAGMA_SETAFFINITY -DGPUSHMEM=300 -DHAVE_CUBLAS -DMAGMA_LIB)
endif()


# PRIMME options
if(MUGIQ_PRIMME)
  if(NOT MUGIQ_MAGMA)
      message(FATAL_ERROR "PRIMME requires MAGMA. Please set MUGIQ_MAGMA ON")
  endif()
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIB_PRIMME QUIET primme)
  if(NOT LIB_PRIMME OR MUGIQ_PRIMME_HOME)
    find_library(LIB_PRIMME libprimme.a primme PATH ${MUGIQ_PRIMME_HOME}/lib) # Prefer Static Library
  else()
    find_library(LIB_PRIMME ${PRIMME_LIBRARIES} PATH ${PRIMME_LIBRARY_DIRS})
  endif()
  add_definitions(-DPRIMME_LIB)

  include_directories(SYSTEM ${MUGIQ_PRIMME_HOME}/include)
  LIST(APPEND LIB_PRIMME ${LIB_PRIMME})
endif()
#--------------------------------------------------------------
#--------------------------------------------------------------


# CUDA related options
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "Host compiler to be used by nvcc")
set(CMAKE_CUDA_STANDARD ${MUGIQ_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_DISABLE_XCOMPILER_WARNINGS "-Wno-unknown-pragmas,-Wno-unused-function,-Wno-unused-local-typedef,-Wno-unused-private-field")

# NVCC FLAGS
set(MUGIQ_NVCC_FLAGS "-ftz=true -prec-div=false -prec-sqrt=false")
set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets -arch=${MUGIQ_GPU_ARCH} --expt-relaxed-constexpr" CACHE STRING "Flags used by the CUDA compiler" FORCE)

set(CMAKE_CUDA_FLAGS_DEVEL
  "${MUGIQ_NVCC_FLAGS} -lineinfo -g -O3 -Xcompiler ${CMAKE_CUDA_DISABLE_XCOMPILER_WARNINGS}"
  CACHE STRING "Flags used by the CUDA compiler during DEVEL (standard) builds.")


# CUDA Wrapper (Take this from QUDA)
if(MUGIQ_QUDA AND NOT ${MUGIQ_QUDA_SOURCE} STREQUAL "")
  list(APPEND CMAKE_MODULE_PATH "${MUGIQ_QUDA_SOURCE}/cmake")
  find_package(CUDAWrapper)
endif()

# Threading
find_package(Threads REQUIRED)

# Compiler options and BUILD types
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(include)
include_directories(lib)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# add tests and mugiq library folders
add_subdirectory(lib)
add_subdirectory(tests)
