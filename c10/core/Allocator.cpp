#include <c10/core/Allocator.h>

namespace c10 {

static void deleteInefficientStdFunctionContext(void* ptr) {
  delete static_cast<InefficientStdFunctionContext*>(ptr);
}

at::DataPtr InefficientStdFunctionContext::makeDataPtr(
    void* ptr,
    const std::function<void(void*)>& deleter,
    Device device) {
  return {
      ptr,
      new InefficientStdFunctionContext({ptr, deleter}),
      &deleteInefficientStdFunctionContext,
      device};
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
at::Allocator* allocator_array[at::COMPILE_TIME_MAX_DEVICE_TYPES];
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
uint8_t allocator_priority[at::COMPILE_TIME_MAX_DEVICE_TYPES] = {0};

void SetAllocator(at::DeviceType t, at::Allocator* alloc, uint8_t priority) {
  if (priority >= allocator_priority[static_cast<int>(t)]) {
    allocator_array[static_cast<int>(t)] = alloc;
    allocator_priority[static_cast<int>(t)] = priority;
  }
}

at::Allocator* GetAllocator(const at::DeviceType& t) {
  auto* alloc = allocator_array[static_cast<int>(t)];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alloc, "Allocator for ", t, " is not set.");
  return alloc;
}

} // namespace c10
