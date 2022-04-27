# RPATH stuff
# see https://cmake.org/Wiki/CMake_RPATH_handling
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(_rpath_portable_origin "@loader_path")
else()
  set(_rpath_portable_origin $ORIGIN)
endif(APPLE)

# Use separate rpaths during build and install phases
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# Don't use the install-rpath during the build phase
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${_rpath_portable_origin}")
# Automatically add all linked folders that are NOT in the build directory to
# the rpath (per library?)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# ---[ CUDA
if(USE_CUDA)
  # public/*.cmake uses CAFFE2_USE_*
  set(CAFFE2_USE_CUDA ${USE_CUDA})
  include(${CMAKE_CURRENT_LIST_DIR}/public/cuda.cmake)
  if(CAFFE2_USE_CUDA)
    # A helper variable recording the list of Caffe2 dependent libraries
    # torch::cudart is dealt with separately, due to CUDA_ADD_LIBRARY
    # design reason (it adds CUDA_LIBRARIES itself).
    set(Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS
      caffe2::cufft caffe2::curand caffe2::cublas)
  else()
    message(WARNING
      "Not compiling with CUDA. Suppress this warning with "
      "-DUSE_CUDA=OFF.")
    caffe2_update_option(USE_CUDA OFF)
    set(CAFFE2_USE_CUDA OFF)
  endif()
endif()

# ---[ Threads
include(${CMAKE_CURRENT_LIST_DIR}/public/threads.cmake)
if(TARGET caffe2::Threads)
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS caffe2::Threads)
else()
  message(FATAL_ERROR
      "Cannot find threading library. Caffe2 requires Threads to compile.")
endif()

# ---[ gflags
if (USE_GFLAGS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/gflags)
  # 不需要设置头文件的位置也可以找到，gflags的头文件
  # include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/gflags)
endif()
list(APPEND Caffe2_DEPENDENCY_LIBS gflags)

# ---[ Googletest and benchmark
if(BUILD_TEST)
  # Preserve build options.
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

  # We will build gtest as static libs and embed it directly into the binary.
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)

  # For gtest, we will simply embed it into our test binaries, so we won't
  # need to install it.
  set(INSTALL_GTEST OFF CACHE BOOL "Install gtest." FORCE)
  set(BUILD_GMOCK ON CACHE BOOL "Build gmock." FORCE)
  # For Windows, we will check the runtime used is correctly passed in.
  if(NOT CAFFE2_USE_MSVC_STATIC_RUNTIME)
      set(gtest_force_shared_crt ON CACHE BOOL "force shared crt on gtest" FORCE)
  endif()

  # Add googletest subdirectory but make sure our INCLUDE_DIRECTORIES
  # don't bleed into it. This is because libraries installed into the root conda
  # env (e.g. MKL) add a global /opt/conda/include directory, and if there's
  # gtest installed in conda, the third_party/googletest/**.cc source files
  # would try to include headers from /opt/conda/include/gtest/**.h instead of
  # its own. Once we have proper target-based include directories,
  # this shouldn't be necessary anymore.
  get_property(INC_DIR_temp DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
  set_property(DIRECTORY PROPERTY INCLUDE_DIRECTORIES "")
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest)
  set_property(DIRECTORY PROPERTY INCLUDE_DIRECTORIES ${INC_DIR_temp})

  include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest/googletest/include)
  include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest/googlemock/include)

  # We will not need to test benchmark lib itself.
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing as we don't need it.")
  # We will not need to install benchmark since we link it statically.
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")
  if(NOT USE_SYSTEM_BENCHMARK)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/benchmark)
  else()
    add_library(benchmark SHARED IMPORTED)
    find_library(BENCHMARK_LIBRARY benchmark)
    if(NOT BENCHMARK_LIBRARY)
      message(FATAL_ERROR "Cannot find google benchmark library")
    endif()
    message("-- Found benchmark: ${BENCHMARK_LIBRARY}")
    set_property(TARGET benchmark PROPERTY IMPORTED_LOCATION ${BENCHMARK_LIBRARY})
  endif()
  include_directories(${CMAKE_CURRENT_LIST_DIR}/../third_party/benchmark/include)

  # Recover build options.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)

endif()