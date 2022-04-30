#pragma once

#include <c10/macros/Macros.h>

#include <cstddef>

namespace c10 {

void* alloc_cpu(size_t nbytes);
void free_cpu(void* data);

} // namespace c10
