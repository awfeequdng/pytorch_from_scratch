cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# ---[ Project and semantic versioning.
project(Torch CXX C)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(LINUX TRUE)
else()
  set(LINUX FALSE)
endif()

set(CMAKE_INSTALL_MESSAGE NEVER)

# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
message("env_cxx_standard: ${env_cxx_standard}")
if(env_cxx_standard GREATER -1)
  message(
    WARNING "C++ standard version definition detected in environment variable."
      "PyTorch requires -std=c++14. Please remove -std=c++ settings in your environment.")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# One variable that determines whether the current cmake process is being run
# with the main Caffe2 library. This is useful for building modules - if
# modules are built with the main Caffe2 library then one does not need to do
# find caffe2 in the cmake script. One can usually guard it in some way like
#    if(NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
#      find_package(Caffe2 REQUIRED)
#    endif()
set(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO ON)

set(CPU_INTEL OFF)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(AMD64|x86_64)")
  set(CPU_INTEL ON)
endif()

# ---[ Options.
# Note to developers: if you add an option below, make sure you also add it to
# cmake/Summary.cmake so that the summary prints out the option values.
include(CMakeDependentOption)
option(BUILD_CUSTOM_PROTOBUF "Build and use Caffe2's own protobuf under third_party" ON)
option(BUILD_PYTHON "Build Python binaries" ON)

option(BUILD_CAFFE2 "Master flag to build Caffe2" OFF)
cmake_dependent_option(
    BUILD_CAFFE2_OPS "Build Caffe2 operators" ON
    "BUILD_CAFFE2" OFF)

option(BUILD_SHARED_LIBS "Build libcaffe2.so" ON)
cmake_dependent_option(
    CAFFE2_LINK_LOCAL_PROTOBUF "If set, build protobuf inside libcaffe2.so." ON
    "BUILD_SHARED_LIBS AND BUILD_CUSTOM_PROTOBUF" OFF)

option(COLORIZE_OUTPUT "Colorize output during compilation" ON)

option(USE_CUDA "Use CUDA" OFF)
# BUILD_SPLIT_CUDA must also be exported as an environment variable before building, with
# `export BUILD_SPLIT_CUDA=1` because cpp_extension.py can only work properly if this variable
# also exists in the environment.
# This option is incompatible with CUDA_SEPARABLE_COMPILATION.
cmake_dependent_option(
    BUILD_SPLIT_CUDA "Split torch_cuda library into torch_cuda_cu and torch_cuda_cpp" OFF
    "USE_CUDA AND NOT CUDA_SEPARABLE_COMPILATION" OFF)
cmake_dependent_option(
    BUILD_LAZY_CUDA_LINALG "Build cuda linalg ops as separate library" ON "USE_CUDA AND LINUX AND BUILD_PYTHON" OFF)

option(CAFFE2_STATIC_LINK_CUDA "Statically link CUDA libraries" OFF)
cmake_dependent_option(
    USE_CUDNN "Use cuDNN" ON
    "USE_CUDA" OFF)
cmake_dependent_option(
    USE_STATIC_CUDNN "Use cuDNN static libraries" OFF
    "USE_CUDNN" OFF)
cmake_dependent_option(
    BUILD_NVFUSER_BENCHMARK "Build C++ binaries for nvfuser benchmarks" ON
    "USE_CUDA;BUILD_TEST" OFF)
cmake_dependent_option(
    USE_EXPERIMENTAL_CUDNN_V8_API "Use experimental cuDNN v8 API" OFF
    "USE_CUDNN" OFF)

option(USE_FBGEMM "Use FBGEMM (quantized 8-bit server operators)" ON)
option(USE_KINETO "Use Kineto profiling library" ON)
option(USE_BREAKPAD "Use breakpad crash dump library" ON)
option(USE_GFLAGS "Use GFLAGS" ON)
option(USE_GLOG "Use GLOG" ON)
option(USE_LEVELDB "Use LEVELDB" OFF)
option(USE_NATIVE_ARCH "Use -march=native" OFF)
cmake_dependent_option(
    USE_NCCL "Use NCCL" ON
    "USE_CUDA OR USE_ROCM;UNIX;NOT APPLE" OFF)
cmake_dependent_option(USE_RCCL "Use RCCL" ON
    USE_NCCL OFF)
cmake_dependent_option(
    USE_STATIC_NCCL "Use static NCCL" OFF
    "USE_NCCL" OFF)
cmake_dependent_option(
    USE_SYSTEM_NCCL "Use system-wide NCCL" OFF
    "USE_NCCL" OFF)
cmake_dependent_option(
    USE_NUMA "Use NUMA. Only available on Linux." ON
    "LINUX" OFF)
cmake_dependent_option(
    USE_NVRTC "Use NVRTC. Only available if USE_CUDA is on." OFF
    "USE_CUDA" OFF)
option(USE_OPENMP "Use OpenMP for parallel code" ON)
cmake_dependent_option(
    USE_PRECOMPILED_HEADERS "Use pre-compiled headers to accelerate build. Requires cmake >= 3.16." OFF
    "CMAKE_VERSION VERSION_GREATER_EQUAL \"3.16\"" OFF)
option(USE_ZMQ "Use ZMQ" OFF)
option(USE_ZSTD "Use ZSTD" OFF)

option(USE_STATIC_MKL "Prefer to link with MKL statically (Unix only)" OFF)
option(USE_DISTRIBUTED "Use distributed" ON)

cmake_dependent_option(
    USE_MPI "Use MPI for Caffe2. Only available if USE_DISTRIBUTED is on." ON
    "USE_DISTRIBUTED" OFF)
cmake_dependent_option(
    USE_GLOO "Use Gloo. Only available if USE_DISTRIBUTED is on." ON
    "USE_DISTRIBUTED" OFF)
cmake_dependent_option(
  USE_GLOO_WITH_OPENSSL "Use Gloo with OpenSSL. Only available if USE_GLOO is on." OFF
    "USE_GLOO AND LINUX AND NOT INTERN_BUILD_MOBILE" OFF)
cmake_dependent_option(
    USE_C10D_GLOO "USE C10D GLOO" ON "USE_DISTRIBUTED;USE_GLOO" OFF)
cmake_dependent_option(
    USE_C10D_NCCL "USE C10D NCCL" ON "USE_DISTRIBUTED;USE_NCCL" OFF)
cmake_dependent_option(
    USE_NCCL_WITH_UCC "Enable UCC support for ProcessGroupNCCL. Only available if USE_C10D_NCCL is on." OFF
    "USE_C10D_NCCL" OFF)
cmake_dependent_option(
    USE_C10D_MPI "USE C10D MPI" ON "USE_DISTRIBUTED;USE_MPI" OFF)
option(HAVE_SOVERSION "Whether to add SOVERSION to the shared objects" OFF)
option(BUILD_LIBTORCH_CPU_WITH_DEBUG "Enable RelWithDebInfo for libtorch_cpu target only" OFF)
cmake_dependent_option(
    USE_DEPLOY "Build embedded torch::deploy interpreter.  See torch/csrc/deploy/README.md for more info." OFF
    "BUILD_PYTHON" OFF)
cmake_dependent_option(USE_CCACHE "Attempt using CCache to wrap the compilation" ON "UNIX" OFF)
option(WERROR "Build with -Werror supported by the compiler" OFF)
option(USE_PER_OPERATOR_HEADERS "Whether ATen should generate separate headers for each operator" ON)
cmake_dependent_option(
    BUILD_LAZY_TS_BACKEND "Build the lazy Torchscript backend, not compatible with mobile builds" ON
    "NOT INTERN_BUILD_MOBILE" OFF)

if(USE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher")
    else()
        message(STATUS "Could not find ccache. Consider installing ccache to speed up compilation.")
    endif()
endif()


if(USE_GLOO_WITH_OPENSSL)
  set(USE_TCP_OPENSSL_LOAD ON CACHE STRING "")
endif()


# Linux distributions do not want too many embedded sources, in that sense we
# need to be able to build pytorch with an (almost) empty third_party
# directory.
# USE_SYSTEM_LIBS is a shortcut variable to toggle all the # USE_SYSTEM_*
# variables on. Individual USE_SYSTEM_* variables can be toggled with
# USE_SYSTEM_LIBS being "OFF".
option(USE_SYSTEM_LIBS "Use all available system-provided libraries." OFF)
option(USE_SYSTEM_CPUINFO "Use system-provided cpuinfo." OFF)
option(USE_SYSTEM_SLEEF "Use system-provided sleef." OFF)
option(USE_SYSTEM_GLOO "Use system-provided gloo." OFF)
option(USE_SYSTEM_FP16 "Use system-provided fp16." OFF)
option(USE_SYSTEM_PYBIND11 "Use system-provided PyBind11." OFF)
option(USE_SYSTEM_PTHREADPOOL "Use system-provided pthreadpool." OFF)
option(USE_SYSTEM_PSIMD "Use system-provided psimd." OFF)
option(USE_SYSTEM_FXDIV "Use system-provided fxdiv." OFF)
option(USE_GOLD_LINKER "Use ld.gold to link" OFF)
if(USE_SYSTEM_LIBS)
  set(USE_SYSTEM_CPUINFO ON)
  set(USE_SYSTEM_SLEEF ON)
  set(USE_SYSTEM_GLOO ON)
  set(BUILD_CUSTOM_PROTOBUF OFF)
  set(USE_SYSTEM_EIGEN_INSTALL ON)
  set(USE_SYSTEM_FP16 ON)
  set(USE_SYSTEM_PTHREADPOOL ON)
  set(USE_SYSTEM_PSIMD ON)
  set(USE_SYSTEM_PYBIND11 ON)
  if(USE_TBB)
    set(USE_SYSTEM_TBB ON)
  endif()
endif()

# Used when building Caffe2 through setup.py
option(BUILDING_WITH_TORCH_LIBS "Tell cmake if Caffe2 is being built alongside torch libs" ON)

# This is a fix for a rare build issue on Ubuntu:
# symbol lookup error: miniconda3/envs/pytorch-py3.7/lib/libmkl_intel_lp64.so: undefined symbol: mkl_blas_dsyrk
# https://software.intel.com/en-us/articles/symbol-lookup-error-when-linking-intel-mkl-with-gcc-on-ubuntu
if(LINUX)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")
endif()

string(APPEND CMAKE_CUDA_FLAGS " -Xfatbin -compress-all")
string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -g -lineinfo --source-in-ptx")
string(APPEND CMAKE_CUDA_FLAGS_RELWITHDEBINFO " -g -lineinfo --source-in-ptx")

set(INTERN_BUILD_ATEN_OPS ON)

# todo: 该文件中的方法没有完全实现
# ---[ Utils
include(cmake/public/utils.cmake)

# ---[ Version numbers for generated libraries
file(READ version.txt TORCH_DEFAULT_VERSION)
# Strip trailing newline
string(REGEX REPLACE "\n$" "" TORCH_DEFAULT_VERSION "${TORCH_DEFAULT_VERSION}")
if("${TORCH_DEFAULT_VERSION} " STREQUAL " ")
  message(WARNING "Could not get version from base 'version.txt'")
  # If we can't get the version from the version file we should probably
  # set it to something non-sensical like 0.0.0
  set(TORCH_DEFAULT_VERSION, "0.0.0")
endif()

set(TORCH_BUILD_VERSION "${TORCH_DEFAULT_VERSION}" CACHE STRING "Torch build version")
if(DEFINED ENV{PYTORCH_BUILD_VERSION})
  set(TORCH_BUILD_VERSION "$ENV{PYTORCH_BUILD_VERSION}"
    CACHE STRING "Torch build version" FORCE)
endif()
if(NOT TORCH_BUILD_VERSION)
  # An empty string was specified so force version to the default
  set(TORCH_BUILD_VERSION "${TORCH_DEFAULT_VERSION}"
    CACHE STRING "Torch build version" FORCE)
endif()
caffe2_parse_version_str(TORCH ${TORCH_BUILD_VERSION})
caffe2_parse_version_str(CAFFE2 ${TORCH_BUILD_VERSION})
set(TORCH_SOVERSION "${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR}")

caffe_print_list(${TORCH_SOVERSION} ${TORCH_VERSION} ${CAFFE2_VERSION})

# ---[ CMake scripts + modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# ---[ CMake build directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

enable_testing()


# ---[ Build variables set within the cmake tree
include(cmake/BuildVariables.cmake)
set(CAFFE2_ALLOWLIST "" CACHE STRING "A allowlist file of files that one should build.")

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set - defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage." FORCE)
endif()

# ---[ Misc checks to cope with various compiler modes
include(cmake/MiscCheck.cmake)

# External projects
include(ExternalProject)

# ---[ Dependencies
# ---[ FBGEMM doesn't work on x86 32bit and CMAKE_SYSTEM_PROCESSOR thinks its 64bit
if(USE_FBGEMM AND ((CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" AND CMAKE_SIZEOF_VOID_P EQUAL 4) OR CMAKE_SYSTEM_PROCESSOR STREQUAL "x86"))
  set(USE_FBGEMM OFF)
endif()


include(cmake/Dependencies.cmake)

if(USE_FBGEMM)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_FBGEMM")
endif()

# # ---[ Allowlist file if allowlist is specified
# include(cmake/Allowlist.cmake)

# ---[ Set link flag, handle additional deps for gcc 4.8 and above
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8.0 AND NOT ANDROID)
  message(STATUS "GCC ${CMAKE_CXX_COMPILER_VERSION}: Adding gcc and gcc_s libs to link line")
  list(APPEND Caffe2_DEPENDENCY_LIBS gcc_s gcc)
endif()

string(APPEND CMAKE_CXX_FLAGS " -O2 -fPIC")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-narrowing")
  # Eigen fails to build with some versions, so convert this to a warning
  # Details at http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1459
  string(APPEND CMAKE_CXX_FLAGS " -Wall")
  string(APPEND CMAKE_CXX_FLAGS " -Wextra")
  string(APPEND CMAKE_CXX_FLAGS " -Werror=return-type")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-missing-field-initializers")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-type-limits")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-array-bounds")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unknown-pragmas")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-sign-compare")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-parameter")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-function")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-result")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-local-typedefs")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-strict-overflow")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-strict-aliasing")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-error=deprecated-declarations")
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-range-loop-analysis")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-pass-failed")
    # sign-compare is not part of -Wall, see https://godbolt.org/z/s1YczM41T
    string(APPEND CMAKE_CXX_FLAGS " -Wsign-compare")
  endif()
  if(CMAKE_COMPILER_IS_GNUCXX AND NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.0))
    string(APPEND CMAKE_CXX_FLAGS " -Wno-stringop-overflow")
  endif()
  if(CMAKE_COMPILER_IS_GNUCXX)
    # Suppress "The ABI for passing parameters with 64-byte alignment has changed in GCC 4.6"
    string(APPEND CMAKE_CXX_FLAGS " -Wno-psabi")
  endif()

  # Use ld.gold if available, fall back to ld.bfd (the default ld) if not
  if(USE_GOLD_LINKER)
    if(USE_DISTRIBUTED AND USE_MPI)
      # Same issue as here with default MPI on Ubuntu
      # https://bugs.launchpad.net/ubuntu/+source/deal.ii/+bug/1841577
      message(WARNING "Refusing to use gold when USE_MPI=1")
    else()
      execute_process(
        COMMAND
        "${CMAKE_C_COMPILER}" -fuse-ld=gold -Wl,--version
         ERROR_QUIET
         OUTPUT_VARIABLE LD_VERSION)
      if(NOT "${LD_VERSION}" MATCHES "GNU gold")
        message(WARNING "USE_GOLD_LINKER was set but ld.gold isn't available, turning it off")
        set(USE_GOLD_LINKER OFF)
      else()
        message(STATUS "ld.gold is available, using it to link")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=gold")
      endif()
    endif()
  endif()

  string(APPEND CMAKE_CXX_FLAGS " -Wno-error=pedantic")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-error=redundant-decls")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-error=old-style-cast")
  # These flags are not available in GCC-4.8.5. Set only when using clang.
  # Compared against https://gcc.gnu.org/onlinedocs/gcc-4.8.5/gcc/Option-Summary.html
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-invalid-partial-specialization")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-typedef-redefinition")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unknown-warning-option")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-private-field")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-inconsistent-missing-override")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-aligned-allocation-unavailable")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-c++14-extensions")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-constexpr-not-const")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-missing-braces")
    string(APPEND CMAKE_CXX_FLAGS " -Qunused-arguments")
    if(${COLORIZE_OUTPUT})
      string(APPEND CMAKE_CXX_FLAGS " -fcolor-diagnostics")
    endif()
  endif()
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
    if(${COLORIZE_OUTPUT})
      string(APPEND CMAKE_CXX_FLAGS " -fdiagnostics-color=always")
    endif()
  endif()
  if((APPLE AND (NOT ("${CLANG_VERSION_STRING}" VERSION_LESS "9.0")))
    OR(CMAKE_COMPILER_IS_GNUCXX
    AND(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0 AND NOT APPLE)))
    string(APPEND CMAKE_CXX_FLAGS " -faligned-new")
  endif()
  if(WERROR)
    check_cxx_compiler_flag("-Werror" COMPILER_SUPPORT_WERROR)
    if(NOT COMPILER_SUPPORT_WERROR)
      set(WERROR FALSE)
    else()
      string(APPEND CMAKE_CXX_FLAGS " -Werror")
    endif()
  endif(WERROR)
  if(NOT APPLE)
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-but-set-variable")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-maybe-uninitialized")
  endif()
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fno-omit-frame-pointer -O0")
  string(APPEND CMAKE_LINKER_FLAGS_DEBUG " -fno-omit-frame-pointer -O0")
  string(APPEND CMAKE_CXX_FLAGS " -fno-math-errno")
  string(APPEND CMAKE_CXX_FLAGS " -fno-trapping-math")
  check_cxx_compiler_flag("-Werror=format" HAS_WERROR_FORMAT)
  if(HAS_WERROR_FORMAT)
    string(APPEND CMAKE_CXX_FLAGS " -Werror=format")
  endif()
  check_cxx_compiler_flag("-Werror=cast-function-type" HAS_WERROR_CAST_FUNCTION_TYPE)
  if(HAS_WERROR_CAST_FUNCTION_TYPE)
    string(APPEND CMAKE_CXX_FLAGS " -Werror=cast-function-type")
  endif()

if(USE_ASAN)
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fsanitize=address")
  string(APPEND CMAKE_LINKER_FLAGS_DEBUG " -fsanitize=address")
endif()

if(APPLE)
    string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-private-field")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-missing-braces")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-c++14-extensions")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-constexpr-not-const")
endif()


if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0.0)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-stringop-overflow")
endif()


if(NOT APPLE AND UNIX)
  list(APPEND Caffe2_DEPENDENCY_LIBS dl)
endif()


# Prefix path to Caffe2 headers.
# If a directory containing installed Caffe2 headers was inadvertently
# added to the list of include directories, prefixing
# PROJECT_SOURCE_DIR means this source tree always takes precedence.
include_directories(BEFORE ${PROJECT_SOURCE_DIR})

# Prefix path to generated Caffe2 headers.
# These need to take precedence over their empty counterparts located
# in PROJECT_SOURCE_DIR.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# ---[ Main build
add_subdirectory(c10)
add_subdirectory(caffe2)


# ---[ CMake related files
# Uninistall option.
if(NOT TARGET caffe2_uninstall)
  configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
      IMMEDIATE @ONLY)

  add_custom_target(caffe2_uninstall
      COMMAND ${CMAKE_COMMAND} -P
      ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

# ---[ Make configuration files for cmake to allow dependent libraries
# easier access to Caffe2.

if((NOT USE_GLOG) OR (NOT USE_GFLAGS) OR BUILD_CUSTOM_PROTOBUF)
  message(WARNING
      "Generated cmake files are only fully tested if one builds "
      "with system glog, gflags, and protobuf. Other settings may "
      "generate files that are not well tested.")
endif()


# Note(jiayq): when building static libraries, all PRIVATE dependencies
# will also become interface libraries, and as a result if there are any
# dependency libraries that are not exported, the following install export
# script will fail. As a result, we will only provide the targets cmake
# files for shared lib installation. For more info, read:
# https://cmake.org/pipermail/cmake/2016-May/063400.html
# if(BUILD_SHARED_LIBS)
#   configure_file(
#       ${PROJECT_SOURCE_DIR}/cmake/Caffe2ConfigVersion.cmake.in
#       ${PROJECT_BINARY_DIR}/Caffe2ConfigVersion.cmake
#       @ONLY)
#   configure_file(
#       ${PROJECT_SOURCE_DIR}/cmake/Caffe2Config.cmake.in
#       ${PROJECT_BINARY_DIR}/Caffe2Config.cmake
#       @ONLY)
#   install(FILES
#       ${PROJECT_BINARY_DIR}/Caffe2ConfigVersion.cmake
#       ${PROJECT_BINARY_DIR}/Caffe2Config.cmake
#       DESTINATION share/cmake/Caffe2
#       COMPONENT dev)
#   install(FILES
#       ${PROJECT_SOURCE_DIR}/cmake/public/cuda.cmake
#       ${PROJECT_SOURCE_DIR}/cmake/public/glog.cmake
#       ${PROJECT_SOURCE_DIR}/cmake/public/gflags.cmake
#       ${PROJECT_SOURCE_DIR}/cmake/public/mkl.cmake
#       ${PROJECT_SOURCE_DIR}/cmake/public/mkldnn.cmake
#       ${PROJECT_SOURCE_DIR}/cmake/public/protobuf.cmake
#       ${PROJECT_SOURCE_DIR}/cmake/public/threads.cmake
#       ${PROJECT_SOURCE_DIR}/cmake/public/utils.cmake
#       DESTINATION share/cmake/Caffe2/public
#       COMPONENT dev)
#   install(DIRECTORY
#       ${PROJECT_SOURCE_DIR}/cmake/Modules_CUDA_fix
#       DESTINATION share/cmake/Caffe2/
#       COMPONENT dev)

#   install(EXPORT Caffe2Targets DESTINATION share/cmake/Caffe2
#       FILE Caffe2Targets.cmake
#       COMPONENT dev)
# else()
#   message(WARNING
#       "Generated cmake files are only available when building "
#       "shared libs.")
# endif()

include(cmake/Summary.cmake)
caffe2_print_configuration_summary()

# # ---[ Torch Deploy
# if(USE_DEPLOY)
#   add_subdirectory(torch/csrc/deploy)
# endif()