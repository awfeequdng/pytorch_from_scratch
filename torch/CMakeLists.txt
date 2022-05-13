# cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
# project(torch CXX C)

set(TORCH_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TORCH_ROOT "${TORCH_SRC_DIR}/..")

message("TORCHSRC_DIR: ${TORCH_SRC_DIR}")

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()

# Generate files
set(TOOLS_PATH "${TORCH_ROOT}/tools")

message("GENERATED_CXX_PYTHON: ${GENRATED_CXX_PYTHON}")
message("CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")


set(TORCH_PYTHON_SRCS ${GENERATED_CXX_PYTHON})
append_filelist("libtorch_python_core_sources" TORCH_PYTHON_SRCS)


set(TORCH_PYTHON_INCLUDE_DIRECTORIES
    ${PYTHON_INCLUDE_DIR}

    ${TORCH_ROOT}

    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/third_party/gflags/include
    ${CMAKE_BINARY_DIR}/third_party/glog

    ${TORCH_ROOT}/third_party
    ${TORCH_ROOT}/third_party/glog/src

    ${TORCH_SRC_DIR}/csrc
)

# set(TORCH_PYTHON_LINK_LIBRARIES
#     python::python
#     pybind::pybind11
#     fmt::fmt-header-only)
set(TORCH_PYTHON_LINK_LIBRARIES
    python::python
    pybind::pybind11)

set(TORCH_PYTHON_COMPILE_OPTIONS)

set(TORCH_PYTHON_LINK_FLAGS "")

if (APPLE)
    # todo: 这行命令用来干嘛？？？？？
    string(APPEND TORCH_PYTHON_LINK_FLAGS " -undefined dynamic_lookup")
else()
    # todo: 一下这三个参数用来干嘛？？？
    list(APPEND TORCH_PYTHON_COMPILE_OPTIONS
        -fno-strict-aliasing
        -Wno-write-strings
        -Wno-strict-aliasing)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    # todo: 如下参数做什么用？？？？
    list(APPEND TORCH_PYTHON_COMPILE_OPTIONS
        -Wno-writable-strings)
endif()

add_library(torch_python SHARED ${TORCH_PYTHON_SRCS})

list(APPEND TORCH_PYTHON_COMPILE_DEFINITIONS BUILD_CAFFE2)
if(HAVE_SOVERSION)
  set_target_properties(torch_python PROPERTIES
      VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
endif()
# add_dependencies(torch_python torch_python_stubs)
# add_dependencies(torch_python flatbuffers)

target_compile_options(torch_python PRIVATE -Wno-unused-variable)

# Required workaround for generated sources
# See https://samthursfield.wordpress.com/2015/11/21/cmake-dependencies-between-targets-and-files-and-custom-commands/#custom-commands-in-different-directories
# add_dependencies(torch_python generate-torch-sources)


target_compile_definitions(torch_python PRIVATE "-DTHP_BUILD_MAIN_LIB")

# target_link_libraries(torch_python PRIVATE ${TORCH_PYTHON_LINK_LIBRARIES})
target_link_libraries(torch_python PRIVATE torch_library ${TORCH_PYTHON_LINK_LIBRARIES})

target_compile_definitions(torch_python PRIVATE ${TORCH_PYTHON_COMPILE_DEFINITIONS})

target_compile_definitions(torch_python PUBLIC ${TORCH_PYTHON_PUBLIC_COMPILE_DEFINITIONS})

target_compile_options(torch_python PRIVATE ${TORCH_PYTHON_COMPILE_OPTIONS})

target_include_directories(torch_python PUBLIC ${TORCH_PYTHON_INCLUDE_DIRECTORIES})

install(TARGETS torch_python DESTINATION "${TORCH_INSTALL_LIB_DIR}")
