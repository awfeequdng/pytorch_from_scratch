#pragma once

#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>

namespace caffe2 {
class TypeMeta;
} // namespace caffe2

namespace c10 {
void set_default_dtype(caffe2::TypeMeta dtype);
const caffe2::TypeMeta get_default_dtype();
ScalarType get_default_dtype_as_scalartype();
const caffe2::TypeMeta get_default_complex_dtype();
} // namespace c10
