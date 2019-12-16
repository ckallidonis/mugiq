#
# CMakeLists.txt for ./lib folder of MUGIQ
#
# C. Kallidonis, William & Mary
# Dec. 2019


# CPP objects
set(MUGIQ_CPP_OBJS
  # cmake-format: sortable
  interface_mugiq.cpp)
# cmake-format: on

#--------------------------------------------------------------


# generate cmake library for all cpp objects
add_library(mugiq_cpp OBJECT ${MUGIQ_CPP_OBJS})

# The MUGIQ library
if(BUILD_MUGIQ_CUDA_LIB)

  # NVCC/CUDA objects
  set(MUGIQ_CUDA_OBJS "" )

  cuda_add_library(mugiq STATIC $<TARGET_OBJECTS:mugiq_cpp> ${MUGIQ_CUDA_OBJS})

  # Include Directories
  target_include_directories(mugiq PRIVATE .)
  target_include_directories(mugiq_cpp PRIVATE .)
  target_include_directories(mugiq PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> $<INSTALL_INTERFACE:include>)


  # If using the CXX flags with the CUDA host compiler
  if(${MUGIQ_USE_CXX_FLAGS_WITH_CUDA})
    target_compile_options(mugiq PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: $<$<CONFIG:DEVEL>:-Xcompiler ${CMAKE_CXX_FLAGS_DEVEL}> >)
  endif()
  #--------------------------------------------------------------


  # Package options

  if(MUGIQ_QUDA)
    target_link_libraries(mugiq INTERFACE ${MUGIQ_QUDA_LDFLAGS} ${MUGIQ_QUDA_LIBS})
  endif()

  if(MUGIQ_QIO)
    target_link_libraries(mugiq INTERFACE ${MUGIQ_QIO_LDFLAGS} ${MUGIQ_QIO_LIBS})
  endif()

  if("${CMAKE_COMMS_TYPE}" STREQUAL "QMP")
    target_link_libraries(mugiq INTERFACE ${MUGIQ_QMP_LDFLAGS} ${MUGIQ_QMP_LIBS} ${MPI_CXX_LIBRARIES})
  endif()

  if("${CMAKE_COMMS_TYPE}" STREQUAL "MPI")
    target_link_libraries(mugiq INTERFACE ${MPI_CXX_LIBRARIES})
  endif()

  if(MUGIQ_MAGMA)
    if(MUGIQ_PRIMME)
      target_link_libraries(mugiq PRIVATE ${LIB_PRIMME})
    endif()
    target_link_libraries(mugiq PRIVATE ${LIB_MAGMA})
  endif()
  #--------------------------------------------------------------

  # Find the CUDA_cuda_library
  target_link_libraries(mugiq PUBLIC ${CUDA_cuda_LIBRARY})

  # Copy the include directory to the build directory
  add_custom_command(TARGET mugiq POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)


  install(TARGETS mugiq EXPORT mugiqTargets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib INCLUDES DESTINATION include)

  install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
  include(CMakePackageConfigHelpers)
  write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/mugiqConfigVersion.cmake"
    VERSION ${MUGIQ_VERSION}
    COMPATIBILITY AnyNewerVersion)

  export(EXPORT mugiqTargets FILE "${CMAKE_CURRENT_BINARY_DIR}/mugiqTargets.cmake" NAMESPACE mugiq::)
  set(ConfigPackageLocation lib/cmake/mugiq/)
  install(EXPORT mugiqTargets NAMESPACE mugiq:: DESTINATION ${ConfigPackageLocation})

endif() # BUILD_MUGIQ_CUDA_LIB

