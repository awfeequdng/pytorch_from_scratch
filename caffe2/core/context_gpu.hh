#pragma once

#include "caffe2/core/context_base.hh"

namespace caffe2 {

class CUDAContext final : public BaseContext {

};

// The number of cuda threads to use. Since work is assigned to SMs at the
// granularity of a block, 128 is chosen to allow utilizing more SMs for
// smaller input sizes.
// 1D grid
constexpr int CAFFE_CUDA_NUM_THREADS = 128;
// 2D grid
constexpr int CAFFE_CUDA_NUM_THREADS_2D_DIMX = 16;
constexpr int CAFFE_CUDA_NUM_THREADS_2D_DIMY = 16;

} // namespace caffe2
