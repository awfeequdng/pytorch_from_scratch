#pragma once

#include <cstdint>
#include <string>
#include <type_traits>

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include <c10/util/typeid.h>
#include <c10/util/Half.h>

///////////////////////////////////////////////////////////////////////////////
// at::Half is defined in c10/util/Half.h. Currently half float operators are
// mainly on CUDA gpus.
// The reason we do not directly use the cuda __half data type is because that
// requires compilation with nvcc. The float16 data type should be compatible
// with the cuda __half data type, but will allow us to refer to the data type
// without the need of cuda.
static_assert(sizeof(unsigned short) == 2,
              "Short on this platform is not 16 bit.");
namespace caffe2 {
// Helpers to avoid using typeinfo with -rtti
template <typename T>
inline bool fp16_type();

template <>
inline bool fp16_type<at::Half>() {
  return true;
}

template <typename T>
inline bool fp16_type() {
  return false;
}

}  // namespace caffe2
