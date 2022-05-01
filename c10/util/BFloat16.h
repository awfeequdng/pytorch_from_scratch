#pragma once

// Defines the bfloat16 type (brain floating-point). This representation uses
// 1 bit for the sign, 8 bits for the exponent and 7 bits for the mantissa.

#include <c10/macros/Macros.h>
#include <climits>
#include <cstdint>
#include <cmath>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif

namespace c10
{
namespace detail {

inline C10_HOST_DEVICE float f32_from_bits(uint16_t src) {
    union {
        float f32;
        uint32_t u32;
    };
    u32 = src;
    u32 <<= 16;
    return f32;
}


inline C10_HOST_DEVICE uint16_t bits_from_f32(float src) {
    union {
        float f32;
        uint32_t u32;
    };
    f32 = src;

    return u32 >> 16;
}

inline C10_HOST_DEVICE uint16_t round_to_nearest_even(float src) {

    if (std::isnan(src)) {
        return UINT16_C(0x7FC0);
    } else {
        union {
            uint32_t U32;
            float F32;
        };

        F32 = src;
        uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
        return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
    }
}
} // namespace detail

struct alignas(2) BFloat16 {
  uint16_t x;

  BFloat16() = default;

  struct from_bits_t {};
  static constexpr C10_HOST_DEVICE from_bits_t from_bits() {
    return from_bits_t();
  }

  constexpr C10_HOST_DEVICE BFloat16(unsigned short bits, from_bits_t)
      : x(bits){};
  inline C10_HOST_DEVICE BFloat16(float value);
  inline C10_HOST_DEVICE operator float() const;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  inline C10_HOST_DEVICE BFloat16(const __nv_bfloat16& value);
  explicit inline C10_HOST_DEVICE operator __nv_bfloat16() const;
#endif
};

} // namespace c10

#include <c10/util/BFloat16-inl.h>