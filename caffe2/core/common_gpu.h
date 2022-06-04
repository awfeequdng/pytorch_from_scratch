#pragma once

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <curand.h>
#include <driver_types.h>


#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"

#include "c10/cuda/CUDAMacros.h"
#include "c10/cuda/CUDAMathCompat.h"
#include <c10/cuda/CUDAGuard.h>
// This is a macro defined for cuda fp16 support. In default, cuda fp16 is
// supported by NVCC 7.5, but it is also included in the Tegra X1 platform with
// a (custom?) NVCC 7.0. As a result, we would normally just check the cuda
// version here, but would also allow a use to pass in the flag
// CAFFE_HAS_CUDA_FP16 manually.

#ifndef CAFFE_HAS_CUDA_FP16
#define CAFFE_HAS_CUDA_FP16
#endif // CAFFE_HAS_CUDA_FP16

#ifdef CAFFE_HAS_CUDA_FP16
#include <cuda_fp16.h>
#endif

#define CUBLAS_ENFORCE(condition)                \
  do {                                           \
    cublasStatus_t status = condition;           \
    CAFFE_ENFORCE_EQ(                            \
        status,                                  \
        CUBLAS_STATUS_SUCCESS,                   \
        "Error at: ",                            \
        __FILE__,                                \
        ":",                                     \
        __LINE__,                                \
        ": ",                                    \
        ::caffe2::cublasGetErrorString(status)); \
  } while (0)
#define CUBLAS_CHECK(condition)                    \
  do {                                             \
    cublasStatus_t status = condition;             \
    CHECK(status == CUBLAS_STATUS_SUCCESS)         \
        << ::caffe2::cublasGetErrorString(status); \
  } while (0)