list(APPEND Caffe2_CPU_SRCS
  utils/math/elementwise.cc
  )
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS}
        utils/math/elementwise.cu
        )
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS}
        )

set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS}
        )


# TODO Once all source files are defined inside the local c10_utils_xxx targets,
# it should be the job of the parent CMakeLists.txt to decide what to do with the target (i.e. link it to caffe2)
# instead of us locally adding it to Caffe2_xxx variables.
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} PARENT_SCOPE)
