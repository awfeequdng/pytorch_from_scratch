#pragma once

namespace c10
{
enum class Backend {
  CPU,
  CUDA,
  SparseCPU,
  SparseCUDA,
  SparseCsrCPU,
  SparseCsrCUDA,
  Undefined,
  NumOptions
};

} // namespace c10
