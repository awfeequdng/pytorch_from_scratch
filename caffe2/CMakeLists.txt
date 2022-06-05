# ---[ Generate and install header and cpp files
include(../cmake/Codegen.cmake)

# 在根目录的CMakelists.txt中已经包含了该文件
# include(../cmake/public/utils.cmake)

set(TORCH_SRC_DIR "${PROJECT_SOURCE_DIR}/torch")
set(TORCH_ROOT "${PROJECT_SOURCE_DIR}")

# This is required for older versions of CMake, which don't allow
# specifying add_library() without a list of source files
set(DUMMY_EMPTY_FILE ${CMAKE_BINARY_DIR}/empty.cpp)
set(DUMMY_FILE_CONTENT "")
file(WRITE ${DUMMY_EMPTY_FILE} ${DUMMY_FILE_CONTENT})

# Wrapper library for people who link against torch and expect both CPU and CUDA support
# Contains "torch_cpu" and "torch_cuda"
add_library(torch ${DUMMY_EMPTY_FILE})

if(HAVE_SOVERSION)
  set_target_properties(torch PROPERTIES
      VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
endif()



set(Caffe2_CPU_SRCS ${DUMMY_EMPTY_FILE})


# 在如下目录中有Caffe2_CPU_SRCS的定义
add_subdirectory(core)
add_subdirectory(utils)

add_library(torch_cpu ${Caffe2_CPU_SRCS})
target_link_libraries(torch_cpu PUBLIC c10)

# Compile exposed libraries.
if(USE_CUDA)
  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)

  add_library(torch_cuda
      ${Caffe2_GPU_SRCS} ${Caffe2_GPU_SRCS_W_SORT_BY_KEY}
      ${Caffe2_GPU_CU_SRCS} ${Caffe2_GPU_CU_SRCS_W_SORT_BY_KEY})

  set(CUDA_LINK_LIBRARIES_KEYWORD)
  torch_compile_options(torch_cuda)  # see cmake/public/utils.cmake
  target_compile_definitions(torch_cuda PRIVATE USE_CUDA)

  # if(USE_NCCL)
  #   target_link_libraries(torch_cuda PRIVATE __caffe2_nccl)
  #   target_compile_definitions(torch_cuda PRIVATE USE_NCCL)
  # endif()
endif()

caffe2_interface_library(torch torch_library)

# install(TARGETS torch torch_library EXPORT Caffe2Targets DESTINATION "${TORCH_INSTALL_LIB_DIR}")

set(TH_CPU_INCLUDE
    # dense
    aten/src/TH
    ${CMAKE_CURRENT_BINARY_DIR}/aten/src/TH
    ${TORCH_ROOT}/aten/src
    ${CMAKE_CURRENT_BINARY_DIR}/aten/src
    ${CMAKE_BINARY_DIR}/aten/src)

set(ATen_CPU_INCLUDE
    ${TORCH_ROOT}/aten/src
    ${CMAKE_CURRENT_BINARY_DIR}/../aten/src
    ${CMAKE_BINARY_DIR}/aten/src)

target_include_directories(torch_cpu PRIVATE ${TH_CPU_INCLUDE})

# torch_cpu_library 链接torch_cpu
caffe2_interface_library(torch_cpu torch_cpu_library)

if(USE_CUDA)
  caffe2_interface_library(torch_cuda torch_cuda_library)
endif()

target_link_libraries(torch PUBLIC torch_cpu_library)

if(USE_CUDA)
  target_link_libraries(torch PUBLIC torch_cuda_library)
endif()

# ---[ CUDA library.
if(USE_CUDA)
  target_link_libraries(torch_cuda INTERFACE torch::cudart)
  # target_link_libraries(torch_cuda PUBLIC c10_cuda torch::nvtoolsext)
  target_include_directories(torch_cuda PRIVATE ${TH_CPU_INCLUDE})
  target_include_directories(
      torch_cuda INTERFACE $<INSTALL_INTERFACE:include>)
  target_include_directories(
      torch_cuda PRIVATE ${Caffe2_GPU_INCLUDE})
  target_link_libraries(
      torch_cuda PRIVATE ${Caffe2_CUDA_DEPENDENCY_LIBS})

  # These public dependencies must go after the previous dependencies, as the
  # order of the libraries in the linker call matters here when statically
  # linking; libculibos and cublas must be last.
  target_link_libraries(torch_cuda PUBLIC torch_cpu_library ${Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS})
endif()

add_subdirectory(../torch torch)

# ---[ Python.
if(BUILD_CAFFE2)
target_compile_definitions(torch PRIVATE BUILD_CAFFE2)
target_compile_definitions(torch_python PRIVATE BUILD_CAFFE2)
endif()