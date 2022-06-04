# ---[ cuda

# Poor man's include guard
if(TARGET torch::cudart)
  return()
endif()

# Enable CUDA language support
# set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# CMake 3.18 adds integrated support for architecture selection, but we can't rely on it
set(CMAKE_CUDA_ARCHITECTURES OFF)

# message(STATUS "Caffe2: CUDA detected: " ${CUDA_VERSION})
# message(STATUS "Caffe2: CUDA nvcc is: " ${CUDA_NVCC_EXECUTABLE})
# message(STATUS "Caffe2: CUDA toolkit directory: " ${CUDA_TOOLKIT_ROOT_DIR})
# if(CUDA_VERSION VERSION_LESS 10.2)
  # message(FATAL_ERROR "PyTorch requires CUDA 10.2 or above.")
# endif()

# find libcuda.so and lbnvrtc.so
# For libcuda.so, we will find it under lib, lib64, and then the
# stubs folder, in case we are building on a system that does not
# have cuda driver installed. On windows, we also search under the
# folder lib/x64.
find_library(CUDA_CUDA_LIB cuda
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/stubs lib64/stubs lib/x64)
find_library(CUDA_NVRTC_LIB nvrtc
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
if(CUDA_NVRTC_LIB AND NOT CUDA_NVRTC_SHORTHASH)
  if("${PYTHON_EXECUTABLE}" STREQUAL "")
    set(_python_exe "python")
  else()
    set(_python_exe "${PYTHON_EXECUTABLE}")
  endif()
  execute_process(
    COMMAND "${_python_exe}" -c
    "import hashlib;hash=hashlib.sha256();hash.update(open('${CUDA_NVRTC_LIB}','rb').read());print(hash.hexdigest()[:8])"
    RESULT_VARIABLE _retval
    OUTPUT_VARIABLE CUDA_NVRTC_SHORTHASH)
  if(NOT _retval EQUAL 0)
    message(WARNING "Failed to compute shorthash for libnvrtc.so")
    set(CUDA_NVRTC_SHORTHASH "XXXXXXXX")
  else()
    string(STRIP "${CUDA_NVRTC_SHORTHASH}" CUDA_NVRTC_SHORTHASH)
    message(STATUS "${CUDA_NVRTC_LIB} shorthash is ${CUDA_NVRTC_SHORTHASH}")
  endif()
endif()

# Create new style imported libraries.
# Several of these libraries have a hardcoded path if CAFFE2_STATIC_LINK_CUDA
# is set. This path is where sane CUDA installations have their static
# libraries installed. This flag should only be used for binary builds, so
# end-users should never have this flag set.

# cuda
add_library(caffe2::cuda UNKNOWN IMPORTED)
set_property(
    TARGET caffe2::cuda PROPERTY IMPORTED_LOCATION
    ${CUDA_CUDA_LIB})
set_property(
    TARGET caffe2::cuda PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# cudart. CUDA_LIBRARIES is actually a list, so we will make an interface
# library.
add_library(torch::cudart INTERFACE IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA)
    set_property(
        TARGET torch::cudart PROPERTY INTERFACE_LINK_LIBRARIES
        "${CUDA_cudart_static_LIBRARY}")

    set_property(
        TARGET torch::cudart APPEND PROPERTY INTERFACE_LINK_LIBRARIES
        rt dl)
else()
    set_property(
        TARGET torch::cudart PROPERTY INTERFACE_LINK_LIBRARIES
        ${CUDA_LIBRARIES})
endif()
set_property(
    TARGET torch::cudart PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})


# cublas. CUDA_CUBLAS_LIBRARIES is actually a list, so we will make an
# interface library similar to cudart.
add_library(caffe2::cublas INTERFACE IMPORTED)
target_link_libraries(caffe2::cublas ${CUDA_cublas_LIBRARY})
# if(CAFFE2_STATIC_LINK_CUDA)
#     set_property(
#         TARGET caffe2::cublas PROPERTY INTERFACE_LINK_LIBRARIES
#         "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas_static.a")
#     set_property(
#       TARGET caffe2::cublas APPEND PROPERTY INTERFACE_LINK_LIBRARIES
#       "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublasLt_static.a")
#     # Add explicit dependency to cudart_static to fix
#     # libcublasLt_static.a.o): undefined reference to symbol 'cudaStreamWaitEvent'
#     # error adding symbols: DSO missing from command line
#     set_property(
#       TARGET caffe2::cublas APPEND PROPERTY INTERFACE_LINK_LIBRARIES
#       "${CUDA_cudart_static_LIBRARY}" rt dl)
# else()
#     set_property(
#         TARGET caffe2::cublas PROPERTY INTERFACE_LINK_LIBRARIES
#         ${CUDA_CUBLAS_LIBRARIES})
# endif()
set_property(
    TARGET caffe2::cublas PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
message(STATUS "Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA}")

# disable some nvcc diagnostic that appears in boost, glog, glags, opencv, etc.
foreach(diag cc_clobber_ignored integer_sign_change useless_using_declaration
             set_but_not_used field_without_dll_interface
             base_class_has_different_dll_interface
             dll_interface_conflict_none_assumed
             dll_interface_conflict_dllexport_assumed
             implicit_return_from_non_void_function
             unsigned_compare_with_zero
             declared_but_not_referenced
             bad_friend_decl)
  list(APPEND SUPPRESS_WARNING_FLAGS --diag_suppress=${diag})
endforeach()
string(REPLACE ";" "," SUPPRESS_WARNING_FLAGS "${SUPPRESS_WARNING_FLAGS}")
list(APPEND CUDA_NVCC_FLAGS -Xcudafe ${SUPPRESS_WARNING_FLAGS})

# Set expt-relaxed-constexpr to suppress Eigen warnings
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

# Set expt-extended-lambda to support lambda on device
list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")

foreach(FLAG ${CUDA_NVCC_FLAGS})
  string(FIND "${FLAG}" " " flag_space_position)
  if(NOT flag_space_position EQUAL -1)
    message(FATAL_ERROR "Found spaces in CUDA_NVCC_FLAGS entry '${FLAG}'")
  endif()
  string(APPEND CMAKE_CUDA_FLAGS " ${FLAG}")
endforeach()
