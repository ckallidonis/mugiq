#
# CMakeLists.txt for ./tests folder of MUGIQ
#
# C. Kallidonis, William & Mary
# Dec. 2019

# Link with libmugiq.a
set(EXE_LIBS mugiq)

# Macro to exclude test for building, if NOT building all tests
macro(MUGIQ_CHECKTEST TEST_TARGET MUGIQ_BUILD_ALL_TESTS_BOOL)
  if(NOT ${MUGIQ_BUILD_ALL_TESTS_BOOL})
    set_property(TARGET ${TEST_TARGET} PROPERTY EXCLUDE_FROM_ALL 1)
  endif()
endmacro()


# Test targets

cuda_add_executable(eigensolve_test_QUDA eigensolve_test_QUDA.cpp)
#wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp clover_reference.cpp blas_reference.cpp)
target_link_libraries(eigensolve_test_QUDA ${EXE_LIBS})
mugiq_checktest(eigensolve_test_QUDA MUGIQ_BUILD_ALL_TESTS)
