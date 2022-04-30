#pragma once


/* Main entry for c10/macros.
 *
 * In your code, include c10/macros/Macros.h directly, instead of individual
 * files in this folder.
 */

// For build systems that do not directly depend on CMake and directly build
// from the source directory (such as Buck), one may not have a cmake_macros.h
// file at all. In this case, the build system is responsible for providing
// correct macro definitions corresponding to the cmake_macros.h.in file.
//
// In such scenarios, one should define the macro
//     C10_USING_CUSTOM_GENERATED_MACROS
// to inform this header that it does not need to include the cmake_macros.h
// file.

#ifndef C10_USING_CUSTOM_GENERATED_MACROS
#include <c10/macros/cmake_macros.h>
#endif // C10_USING_CUSTOM_GENERATED_MACROS


// Disable the copy and assignment operator for a class. Note that this will
// disable the usage of the class in std containers.
#define C10_DISABLE_COPY_AND_ASSIGN(classname) \
  classname(const classname&) = delete;        \
  classname& operator=(const classname&) = delete

#define C10_CONCATENATE_IMPL(s1, s2) s1##s2
#define C10_CONCATENATE(s1, s2) C10_CONCATENATE_IMPL(s1, s2)

#define C10_STRINGIZE_IMPL(x) #x
#define C10_STRINGIZE(x) C10_STRINGIZE_IMPL(x)

/**
 * C10_ANONYMOUS_VARIABLE(str) introduces an identifier starting with
 * str and ending with a number that varies with the line.
 */
#ifdef __COUNTER__
#define C10_UID __COUNTER__
#define C10_ANONYMOUS_VARIABLE(str) C10_CONCATENATE(str, __COUNTER__)
#else
#define C10_UID __LINE__
#define C10_ANONYMOUS_VARIABLE(str) C10_CONCATENATE(str, __LINE__)
#endif

// C10_LIKELY/C10_UNLIKELY
//
// These macros provide parentheses, so you can use these macros as:
//
//    if C10_LIKELY(some_expr) {
//      ...
//    }
//
// NB: static_cast to boolean is mandatory in C++, because __builtin_expect
// takes a long argument, which means you may trigger the wrong conversion
// without it.
//
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define C10_LIKELY(expr) (expr)
#define C10_UNLIKELY(expr) (expr)
#endif

/// C10_NOINLINE - Functions whose declaration is annotated with this will not
/// be inlined.
#ifdef __GNUC__
#define C10_NOINLINE __attribute__((noinline))
#else
#define C10_NOINLINE
#endif


#if __has_attribute(always_inline) || defined(__GNUC__)
#define C10_ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define C10_ALWAYS_INLINE inline
#endif

#if defined(__GNUC__)
#define C10_ATTR_VISIBILITY_HIDDEN __attribute__((__visibility__("hidden")))
#else
#define C10_ATTR_VISIBILITY_HIDDEN
#endif

// C10_FALLTHROUGH - Annotate fallthrough to the next case in a switch.
#define C10_FALLTHROUGH [[fallthrough]]

// suppress an unused variable.
#define C10_UNUSED __attribute__((__unused__))

namespace c10 {} // namespace c10

namespace at {
using namespace c10;
}

