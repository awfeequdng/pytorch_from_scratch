#pragma once

#include <c10/cuda/CUDAMacros.h>
#include "caffe2/core/context_base.hh"

namespace caffe2 {
enum class CudaMemoryPoolType {
  NONE = 0,
  CUB = 1,
  THC = 2,
};

/**
 * Gets the current memory pool type used by Caffe2.
 *
 * The memory pool is set up during caffe2's global initialization time.
 */
CudaMemoryPoolType GetCudaMemoryPoolType();


/**
 * A struct to host thread-local cuda objects.
 *
 * In Caffe2, each thread has its own non-default cuda stream as well as
 * related objects such as cublas and curand handles. This is achieved by
 * having the ThreadLocalCUDAObjects wrapper that takes care of allocating
 * and deallocating these objects at the thread scope. This class is solely
 * used inside CUDAContext and should not be used externally.
 *
 * This class manages the mapping from logical stream ID (int stream_id
 * passed around in Caffe2) and CUDAStream objects.  We intend to eventually
 * deprecate the logical stream ID interface, but not for now.
 */
class ThreadLocalCUDAObjects {
  friend class CUDAContext;

 private:
  ThreadLocalCUDAObjects() {
    for (DeviceIndex i = 0; i < C10_COMPILE_TIME_MAX_GPUS; ++i) {
      cuda_streams_[i] = vector<c10::cuda::CUDAStream>();
    }
  }

  // Record current stream id for the current thread.
  // This is the new API we're trying to migrate use cases to and get rid of
  // explicit stream id passing. For now it's invoked in
  // CUDAContext::SwitchToDevice
  void SetCurrentStreamId(DeviceIndex gpu, StreamId stream_id) {
    // TODO: use current device id from thread local instead of passing gpu in
    if (stream_id != -1) {
      c10::cuda::setCurrentCUDAStream(GetCUDAStream(gpu, stream_id));
    }
  }

  // Retrieves the CUDAStream corresponding to a logical stream ID, ensuring
  // that it exists in cuda_streams_ if it has not been allocated yet.
  c10::cuda::CUDAStream GetCUDAStream(DeviceIndex gpu, StreamId stream_id) {
    vector<c10::cuda::CUDAStream>& gpu_streams = cuda_streams_[gpu];
    while (gpu_streams.size() <= static_cast<size_t>(stream_id)) {
      // NB: This streams are not guaranteed to be unique; we'll
      // wrap around once we run out of streams in the pool.
      gpu_streams.emplace_back(c10::cuda::getStreamFromPool(/* high priority */ false, gpu));
    }
    return gpu_streams[stream_id];
  }

  // Uses the logical stream id from the thread local to pick the stream
  // We're going to migrate all usages to this case API instead of passing the
  // stream id directly
  cudaStream_t GetStream(DeviceIndex gpu) {
    return c10::cuda::getCurrentCUDAStream(gpu).stream();
  }

  cudaStream_t GetStream(DeviceIndex gpu, StreamId stream_id) {
    return GetCUDAStream(gpu, stream_id).stream();
  }

  // Uses the logical stream id from the thread local to pick the stream
  // We're going to migrate all usages to this case API instead of passing the
  // stream id directly
  cublasHandle_t GetHandle(DeviceIndex gpu) {
    return GetHandle(c10::cuda::getCurrentCUDAStream(gpu));
  }

  cublasHandle_t GetHandle(c10::cuda::CUDAStream cuda_stream) {
    CUDAGuard guard(cuda_stream.device_index());
    // Default construct in the map if it doesn't exist, and return a mutable
    // reference to it.
    auto& r = cublas_handles_[cuda_stream];
    if (r == nullptr) {
      CUBLAS_ENFORCE(cublasCreate(&r));
      // The default is CUBLAS_POINTER_MODE_HOST. You can override
      // it after obtaining the cublas handle, but do that with
      // caution.
      CUBLAS_ENFORCE(cublasSetPointerMode(r, CUBLAS_POINTER_MODE_HOST));
      CUBLAS_ENFORCE(cublasSetStream(r, cuda_stream));
    }
    return r;
  }

  ~ThreadLocalCUDAObjects() noexcept {
    for (auto element : cublas_handles_) {
      if (element.second) {
        CUBLAS_CHECK(cublasDestroy(element.second));
      }
    }
  }
  // WARNING: mapping from logical stream ID to c10::cuda::CUDAStream
  // is NOT bijective; multiple logical stream IDs may map to the
  // same underlying stream ID.
  vector<c10::cuda::CUDAStream> cuda_streams_[C10_COMPILE_TIME_MAX_GPUS];
  std::unordered_map<c10::cuda::CUDAStream, cublasHandle_t> cublas_handles_;
};

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
