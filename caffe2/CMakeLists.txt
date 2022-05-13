# ---[ Generate and install header and cpp files
include(../cmake/Codegen.cmake)

# caffe2_interface_library(torch torch_library)

add_subdirectory(../torch torch)