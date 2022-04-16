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
  set(CAFFE2_USE_CUDNN ${USE_CUDNN})
  set(CAFFE2_USE_NVRTC ${USE_NVRTC})
  set(CAFFE2_USE_TENSORRT ${USE_TENSORRT})
  include(${CMAKE_CURRENT_LIST_DIR}/public/cuda.cmake)
  if(CAFFE2_USE_CUDA)
    # A helper variable recording the list of Caffe2 dependent libraries
    # torch::cudart is dealt with separately, due to CUDA_ADD_LIBRARY
    # design reason (it adds CUDA_LIBRARIES itself).
    set(Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS
      caffe2::cufft caffe2::curand caffe2::cublas)
    if(CAFFE2_USE_NVRTC)
      list(APPEND Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS caffe2::cuda caffe2::nvrtc)
    else()
      caffe2_update_option(USE_NVRTC OFF)
    endif()
    if(CAFFE2_USE_CUDNN)
      list(APPEND Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS caffe2::cudnn-public)
    else()
      caffe2_update_option(USE_CUDNN OFF)
    endif()
    if(CAFFE2_USE_TENSORRT)
      list(APPEND Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS caffe2::tensorrt)
    else()
      caffe2_update_option(USE_TENSORRT OFF)
    endif()
  else()
    message(WARNING
      "Not compiling with CUDA. Suppress this warning with "
      "-DUSE_CUDA=OFF.")
    caffe2_update_option(USE_CUDA OFF)
    caffe2_update_option(USE_CUDNN OFF)
    caffe2_update_option(USE_NVRTC OFF)
    caffe2_update_option(USE_TENSORRT OFF)
    set(CAFFE2_USE_CUDA OFF)
    set(CAFFE2_USE_CUDNN OFF)
    set(CAFFE2_USE_NVRTC OFF)
    set(CAFFE2_USE_TENSORRT OFF)
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

# ---[ protobuf
if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  if(USE_LITE_PROTO)
    set(CAFFE2_USE_LITE_PROTO 1)
  endif()
endif()