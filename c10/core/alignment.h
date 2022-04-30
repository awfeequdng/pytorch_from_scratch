#pragma once

#include <cstddef>

namespace c10 {

// Use 64-byte alignment should be enough for computation up to AVX512.
constexpr size_t gAlignment = 64;

} // namespace c10
