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
        caffe2::cublas)
    # set(Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS
    #   caffe2::cufft caffe2::curand caffe2::cublas)
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
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  set(TEMP_BUILD_STATIC_LIBS ${BUILD_STATIC_LIBS})
  set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libs" FORCE)
  set(BUILD_STATIC_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/gflags)
  # 不需要设置头文件的位置也可以找到，gflags的头文件
  # include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/gflags)

  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)
  set(BUILD_STATIC_LIBS ${TEMP_BUILD_STATIC_LIBS} CACHE BOOL "Build shared libs" FORCE)
endif()
list(APPEND Caffe2_DEPENDENCY_LIBS gflags)

# ---[ gflags
if (USE_GLOG)
  # Preserve build options.
  # set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  set(TEMP_WITH_CUSTOM_PREFIX ${WITH_CUSTOM_PREFIX})
  set(TEMP_WITH_GFLAGS ${WITH_GFLAGS})
  set(TEMP_WITH_GTEST ${WITH_GTEST})
  set(TEMP_WITH_PKGCONFIG ${WITH_PKGCONFIG})
  set(TEMP_WITH_SYMBOLIZE ${WITH_SYMBOLIZE})
  # set(WITH_UNWIND ${WITH_UNWIND})
  # set(WITH_TLS ${WITH_TLS})

  # We will build gtest as static libs and embed it directly into the binary.
  # set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
  set(WITH_CUSTOM_PREFIX OFF CACHE BOOL " " FORCE)
  set(WITH_GFLAGS OFF CACHE BOOL " " FORCE)
  set(WITH_GTEST OFF CACHE BOOL " " FORCE)
  set(WITH_PKGCONFIG OFF CACHE BOOL " " FORCE)
  set(WITH_SYMBOLIZE OFF CACHE BOOL " " FORCE)

# option (WITH_CUSTOM_PREFIX "Enable support for user-generated message prefixes" ON)
# option (WITH_GFLAGS "Use gflags" ON)
# option (WITH_GTEST "Use Google Test" ON)
# option (WITH_PKGCONFIG "Enable pkg-config support" ON)
# option (WITH_SYMBOLIZE "Enable symbolize module" ON)
# option (WITH_THREADS "Enable multithreading support" ON)
# option (WITH_TLS "Enable Thread Local Storage (TLS) support" ON)
# option (WITH_UNWIND "Enable libunwind support" ON)

  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/glog)
  # include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/gflags)

  # Recover build options.
  # set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)
  set(WITH_CUSTOM_PREFIX ${TEMP_WITH_CUSTOM_PREFIX} CACHE BOOL " " FORCE)
  set(WITH_GFLAGS ${TEMP_WITH_GFLAGS} CACHE BOOL " " FORCE)
  set(WITH_GTEST ${TEMP_WITH_GTEST} CACHE BOOL " " FORCE)
  set(WITH_PKGCONFIG ${TEMP_WITH_PKGCONFIG} CACHE BOOL " " FORCE)
  set(WITH_SYMBOLIZE ${TEMP_WITH_SYMBOLIZE} CACHE BOOL " " FORCE)
endif()
list(APPEND Caffe2_DEPENDENCY_LIBS glog::glog)


# ---[ NUMA
if(USE_NUMA)
  if(LINUX)
    find_package(Numa)
    if(NUMA_FOUND)
      include_directories(SYSTEM ${Numa_INCLUDE_DIR})
      list(APPEND Caffe2_DEPENDENCY_LIBS ${Numa_LIBRARIES})
    else()
      message(WARNING "Not compiling with NUMA. Suppress this warning with -DUSE_NUMA=OFF")
      caffe2_update_option(USE_NUMA OFF)
    endif()
  else()
    message(WARNING "NUMA is currently only supported under Linux.")
    caffe2_update_option(USE_NUMA OFF)
  endif()
endif()

# ---[ Googletest and benchmark
if(BUILD_TEST)
  # Preserve build options.
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

  # We will build gtest as static libs and embed it directly into the binary.
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)

  # For gtest, we will simply embed it into our test binaries, so we won't
  # need to install it.
  set(INSTALL_GTEST OFF CACHE BOOL "Install gtest." FORCE)
  # set(BUILD_GMOCK ON CACHE BOOL "Build gmock." FORCE)

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
  # include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest/googlemock/include)

  # # We will not need to test benchmark lib itself.
  # set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing as we don't need it.")
  # # We will not need to install benchmark since we link it statically.
  # set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")
  # if(NOT USE_SYSTEM_BENCHMARK)
  #   add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/benchmark)
  # else()
  #   add_library(benchmark SHARED IMPORTED)
  #   find_library(BENCHMARK_LIBRARY benchmark)
  #   if(NOT BENCHMARK_LIBRARY)
  #     message(FATAL_ERROR "Cannot find google benchmark library")
  #   endif()
  #   message("-- Found benchmark: ${BENCHMARK_LIBRARY}")
  #   set_property(TARGET benchmark PROPERTY IMPORTED_LOCATION ${BENCHMARK_LIBRARY})
  # endif()
  # include_directories(${CMAKE_CURRENT_LIST_DIR}/../third_party/benchmark/include)

  # Recover build options.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)
endif()


# ---[ Python + Numpy
if(BUILD_PYTHON)
  message(STATUS "BUILD_PYTHON")
  # If not given a Python installation, then use the current active Python
  if(NOT PYTHON_EXECUTABLE)
    execute_process(
      COMMAND "which" "python" RESULT_VARIABLE _exitcode OUTPUT_VARIABLE _py_exe)
    if(${_exitcode} EQUAL 0)
      if(NOT MSVC)
        string(STRIP ${_py_exe} PYTHON_EXECUTABLE)
      endif()
      message(STATUS "Setting Python to ${PYTHON_EXECUTABLE}")
    endif()
  endif()

  # Check that Python works
  set(PYTHON_VERSION)
  if(DEFINED PYTHON_EXECUTABLE)
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "--version"
        RESULT_VARIABLE _exitcode OUTPUT_VARIABLE PYTHON_VERSION)
    if(NOT _exitcode EQUAL 0)
      message(FATAL_ERROR "The Python executable ${PYTHON_EXECUTABLE} cannot be run. Make sure that it is an absolute path.")
    endif()
    if(PYTHON_VERSION)
      string(REGEX MATCH "([0-9]+)\\.([0-9]+)" PYTHON_VERSION ${PYTHON_VERSION})
    endif()
  endif()

  # Seed PYTHON_INCLUDE_DIR and PYTHON_LIBRARY to be consistent with the
  # executable that we already found (if we didn't actually find an executable
  # then these will just use "python", but at least they'll be consistent with
  # each other).
  if(NOT PYTHON_INCLUDE_DIR)
    # TODO: Verify that sysconfig isn't inaccurate
    message(STATUS "not define PYTHON_INCLUDE_DIR")
    pycmd_no_exit(_py_inc _exitcode "import sysconfig; print(sysconfig.get_path('include'))")
    if("${_exitcode}" EQUAL 0 AND IS_DIRECTORY "${_py_inc}")
    set(PYTHON_INCLUDE_DIR "${_py_inc}")
    message(STATUS "Setting Python's include dir to ${_py_inc} from sysconfig")
    else()
    message(WARNING "Could not set Python's include dir to ${_py_inc} from sysconfig")
    endif()
    endif(NOT PYTHON_INCLUDE_DIR)

    if(NOT PYTHON_LIBRARY)
    message(STATUS "not define PYTHON_LIBRARY")
    pycmd_no_exit(_py_lib _exitcode "import sysconfig; print(sysconfig.get_path('stdlib'))")
    if("${_exitcode}" EQUAL 0 AND EXISTS "${_py_lib}" AND EXISTS "${_py_lib}")
      set(PYTHON_LIBRARY "${_py_lib}")
      if(MSVC)
        string(REPLACE "Lib" "libs" _py_static_lib ${_py_lib})
        link_directories(${_py_static_lib})
      endif()
      message(STATUS "Setting Python's library to ${PYTHON_LIBRARY}")
    endif()
  endif(NOT PYTHON_LIBRARY)

  # These should fill in the rest of the variables, like versions, but resepct
  # the variables we set above
  set(Python_ADDITIONAL_VERSIONS ${PYTHON_VERSION} 3.9 3.8 3.7)
  find_package(PythonInterp 3.0)
  find_package(PythonLibs 3.0)

  if(${PYTHONLIBS_VERSION_STRING} VERSION_LESS 3)
    message(FATAL_ERROR
      "Found Python libraries version ${PYTHONLIBS_VERSION_STRING}. Python 2 has reached end-of-life and is no longer supported by PyTorch.")
  endif()
  if(${PYTHONLIBS_VERSION_STRING} VERSION_LESS 3.7)
    message(FATAL_ERROR
      "Found Python libraries version ${PYTHONLIBS_VERSION_STRING}. Python 3.6 is no longer supported by PyTorch.")
  endif()

  # When building pytorch, we pass this in directly from setup.py, and
  # don't want to overwrite it because we trust python more than cmake
  # if(NUMPY_INCLUDE_DIR)
  #   set(NUMPY_FOUND ON)
  # elseif(USE_NUMPY)
  #   find_package(NumPy)
  #   if(NOT NUMPY_FOUND)
  #     message(WARNING "NumPy could not be found. Not building with NumPy. Suppress this warning with -DUSE_NUMPY=OFF")
  #   endif()
  # endif()

  # if(PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)
    add_library(python::python INTERFACE IMPORTED)
    set_property(TARGET python::python PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${PYTHON_INCLUDE_DIRS})
    set_property(TARGET python::python PROPERTY
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${PYTHON_INCLUDE_DIRS})
    # if(WIN32)
      set_property(TARGET python::python PROPERTY
          INTERFACE_LINK_LIBRARIES ${PYTHON_LIBRARIES})
    # endif()

    # caffe2_update_option(USE_NUMPY OFF)
    # if(NUMPY_FOUND)
    #   caffe2_update_option(USE_NUMPY ON)
    #   add_library(numpy::numpy INTERFACE IMPORTED)
    #   set_property(TARGET numpy::numpy PROPERTY
    #       INTERFACE_INCLUDE_DIRECTORIES ${NUMPY_INCLUDE_DIR})
    #   set_property(TARGET numpy::numpy PROPERTY
    #       INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${NUMPY_INCLUDE_DIR})
    # endif()
    # Observers are required in the python build
    caffe2_update_option(USE_OBSERVERS ON)
  # else()
  #   message(WARNING "Python dependencies not met. Not compiling with python. Suppress this warning with -DBUILD_PYTHON=OFF")
  #   caffe2_update_option(BUILD_PYTHON OFF)
  # endif()
endif()

# ---[ pybind11
if(USE_SYSTEM_BIND11)
  message(STATUS "USE_SYSTEM_BIND11")
  find_package(pybind11 CONFIG)
  if(NOT pybind11_FOUND)
    find_package(pybind11)
  endif()
  if(NOT pybind11_FOUND)
    message(FATAL "Cannot find system pybind11")
  endif()
else()
    message(STATUS "Using third_party/pybind11.")
    set(pybind11_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../third_party/pybind11/include)
    install(DIRECTORY ${pybind11_INCLUDE_DIRS}
            DESTINATION ${CMAKE_INSTALL_PREFIX}
            FILES_MATCHING PATTERN "*.h")
endif()
message(STATUS "pybind11 include dirs: " "${pybind11_INCLUDE_DIRS}")
add_library(pybind::pybind11 INTERFACE IMPORTED)
set_property(TARGET pybind::pybind11 PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${pybind11_INCLUDE_DIRS})
set_property(TARGET pybind::pybind11 PROPERTY
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${pybind11_INCLUDE_DIRS})
set_property(TARGET pybind::pybind11 PROPERTY
    INTERFACE_LINK_LIBRARIES python::python)

# ---[ BLAS

# setting default preferred BLAS options if not already present.
# if(NOT INTERN_BUILD_MOBILE)
#   set(BLAS "MKL" CACHE STRING "Selected BLAS library")
# else()
set(BLAS "Eigen" CACHE STRING "Selected BLAS library")
set(AT_MKLDNN_ENABLED 0)
set(AT_MKL_ENABLED 0)
# endif()
set_property(CACHE BLAS PROPERTY STRINGS "ATLAS;BLIS;Eigen;FLAME;Generic;MKL;OpenBLAS;vecLib")
message(STATUS "Trying to find preferred BLAS backend of choice: " ${BLAS})

if(BLAS STREQUAL "Eigen")
  # Eigen is header-only and we do not have any dependent libraries
  set(CAFFE2_USE_EIGEN_FOR_BLAS ON)
elseif(BLAS STREQUAL "ATLAS")
  find_package(Atlas REQUIRED)
  include_directories(SYSTEM ${ATLAS_INCLUDE_DIRS})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${ATLAS_LIBRARIES})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS cblas)
  set(BLAS_INFO "atlas")
  set(BLAS_FOUND 1)
  set(BLAS_LIBRARIES ${ATLAS_LIBRARIES} cblas)
elseif(BLAS STREQUAL "OpenBLAS")
  find_package(OpenBLAS REQUIRED)
  include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${OpenBLAS_LIB})
  set(BLAS_INFO "open")
  set(BLAS_FOUND 1)
  set(BLAS_LIBRARIES ${OpenBLAS_LIB})
elseif(BLAS STREQUAL "BLIS")
  find_package(BLIS REQUIRED)
  include_directories(SYSTEM ${BLIS_INCLUDE_DIR})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${BLIS_LIB})
elseif(BLAS STREQUAL "MKL")
  if(BLAS_SET_BY_USER)
    find_package(MKL REQUIRED)
  else()
    find_package(MKL QUIET)
  endif()
  include(${CMAKE_CURRENT_LIST_DIR}/public/mkl.cmake)
  if(MKL_FOUND)
    message(STATUS "MKL libraries: ${MKL_LIBRARIES}")
    message(STATUS "MKL include directory: ${MKL_INCLUDE_DIR}")
    message(STATUS "MKL OpenMP type: ${MKL_OPENMP_TYPE}")
    message(STATUS "MKL OpenMP library: ${MKL_OPENMP_LIBRARY}")
    include_directories(AFTER SYSTEM ${MKL_INCLUDE_DIR})
    list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS caffe2::mkl)
    set(CAFFE2_USE_MKL ON)
    set(BLAS_INFO "mkl")
    set(BLAS_FOUND 1)
    set(BLAS_LIBRARIES ${MKL_LIBRARIES})
  else()
    message(WARNING "MKL could not be found. Defaulting to Eigen")
    set(CAFFE2_USE_EIGEN_FOR_BLAS ON)
    set(CAFFE2_USE_MKL OFF)
  endif()
elseif(BLAS STREQUAL "vecLib")
  find_package(vecLib REQUIRED)
  include_directories(SYSTEM ${vecLib_INCLUDE_DIR})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${vecLib_LINKER_LIBS})
  set(BLAS_INFO "veclib")
  set(BLAS_FOUND 1)
  set(BLAS_LIBRARIES ${vecLib_LINKER_LIBS})
elseif(BLAS STREQUAL "FlexiBLAS")
  find_package(FlexiBLAS REQUIRED)
  include_directories(SYSTEM ${FlexiBLAS_INCLUDE_DIR})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${FlexiBLAS_LIB})
elseif(BLAS STREQUAL "Generic")
  # On Debian family, the CBLAS ABIs have been merged into libblas.so
  find_library(BLAS_LIBRARIES blas)
  message("-- Using BLAS: ${BLAS_LIBRARIES}")
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${BLAS_LIBRARIES})
  set(GENERIC_BLAS_FOUND TRUE)
  set(BLAS_INFO "generic")
  set(BLAS_FOUND 1)
else()
  message(FATAL_ERROR "Unrecognized BLAS option: " ${BLAS})
endif()

# ---[ EIGEN
# Due to license considerations, we will only use the MPL2 parts of Eigen.
set(EIGEN_MPL2_ONLY 1)
if(USE_SYSTEM_EIGEN_INSTALL)
  find_package(Eigen3)
  if(EIGEN3_FOUND)
    message(STATUS "Found system Eigen at " ${EIGEN3_INCLUDE_DIR})
  else()
    message(STATUS "Did not find system Eigen. Using third party subdirectory.")
    set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/eigen)
    caffe2_update_option(USE_SYSTEM_EIGEN_INSTALL OFF)
  endif()
else()
  message(STATUS "Using third party subdirectory Eigen.")
  set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/eigen)
endif()
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})