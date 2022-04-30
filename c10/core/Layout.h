#pragma once

#include <c10/core/Backend.h>
#include <c10/util/Exception.h>

#include <ostream>

namespace c10 {
enum class Layout : int8_t { Strided, Sparse, SparseCsr, Mkldnn, NumOptions };

constexpr auto kStrided = Layout::Strided;
constexpr auto kSparse = Layout::Sparse;
constexpr auto kSparseCsr = Layout::SparseCsr;
constexpr auto kMkldnn = Layout::Mkldnn;

inline Layout layout_from_backend(Backend backend) {
  switch (backend) {
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
      return Layout::Sparse;
    case Backend::MkldnnCPU:
      return Layout::Mkldnn;
    case Backend::SparseCsrCPU:
    case Backend::SparseCsrCUDA:
      return Layout::SparseCsr;
    default:
      return Layout::Strided;
  }
}

inline std::ostream& operator<<(std::ostream& stream, at::Layout layout) {
  switch (layout) {
    case at::kStrided:
      return stream << "Strided";
    case at::kSparse:
      return stream << "Sparse";
    case at::kSparseCsr:
      return stream << "SparseCsr";
    case at::kMkldnn:
      return stream << "Mkldnn";
    default:
      TORCH_CHECK(false, "Unknown layout");
  }
}

} // namespace c10
