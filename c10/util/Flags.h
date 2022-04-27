#pragma once

#include <string>
#include <c10/macros/Macros.h>

namespace c10
{
void SetUsageMessage(const std::string& str);

const char* UsageMessage();

bool ParseCommandLineFlags(int* pargc, char*** pargv);

bool CommandLineFlagsHasBeenParsed();

} // namespace c10

#ifdef C10_USE_GFLAGS

#include <gflags/gflags.h>

// C10 uses hidden visibility by default. However, in gflags, it only uses
// export on Windows platform (with dllexport) but not on linux/mac (with
// default visibility). As a result, to ensure that we are always exporting
// global variables, we will redefine the GFLAGS_DLL_DEFINE_FLAG macro if we
// are building C10 as a shared libray.
// This has to be done after the inclusion of gflags, because some early
// versions of gflags.h (e.g. 2.0 on ubuntu 14.04) directly defines the
// macros, so we need to do definition after gflags is done.
#ifdef GFLAGS_DLL_DEFINE_FLAG
#undef GFLAGS_DLL_DEFINE_FLAG
#endif // GFLAGS_DLL_DEFINE_FLAG
#ifdef GFLAGS_DLL_DECLARE_FLAG
#undef GFLAGS_DLL_DECLARE_FLAG
#endif // GFLAGS_DLL_DECLARE_FLAG
#define GFLAGS_DLL_DEFINE_FLAG
#define GFLAGS_DLL_DECLARE_FLAG

// gflags before 2.0 uses namespace google and after 2.1 uses namespace gflags.
// Using GFLAGS_GFLAGS_H_ to capture this change.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif // GFLAGS_GFLAGS_H_

// Motivation about the gflags wrapper:
// (1) We would need to make sure that the gflags version and the non-gflags
// version of C10 are going to expose the same flags abstraction. One should
// explicitly use FLAGS_flag_name to access the flags.
// (2) For flag names, it is recommended to start with c10_ to distinguish it
// from regular gflags flags. For example, do
//    C10_DEFINE_BOOL(c10_my_flag, true, "An example");
// to allow one to use FLAGS_c10_my_flag.
// (3) Gflags has a design issue that does not properly expose the global flags,
// if one builds the library with -fvisibility=hidden. The current gflags (as of
// Aug 2018) only deals with the Windows case using dllexport, and not the Linux
// counterparts. As a result, we will explciitly use C10_EXPORT to export the
// flags defined in C10. This is done via a global reference, so the flag
// itself is not duplicated - under the hood it is the same global gflags flag.
#define C10_GFLAGS_DEF_WRAPPER(type, real_type, name, default_value, help_str) \
  DEFINE_##type(name, default_value, help_str);

#define C10_DEFINE_int(name, default_value, help_str) \
  C10_GFLAGS_DEF_WRAPPER(int32, gflags::int32, name, default_value, help_str)
#define C10_DEFINE_int32(name, default_value, help_str) \
  C10_DEFINE_int(name, default_value, help_str)
#define C10_DEFINE_int64(name, default_value, help_str) \
  C10_GFLAGS_DEF_WRAPPER(int64, gflags::int64, name, default_value, help_str)
#define C10_DEFINE_double(name, default_value, help_str) \
  C10_GFLAGS_DEF_WRAPPER(double, double, name, default_value, help_str)
#define C10_DEFINE_bool(name, default_value, help_str) \
  C10_GFLAGS_DEF_WRAPPER(bool, bool, name, default_value, help_str)
#define C10_DEFINE_string(name, default_value, help_str) \
  C10_GFLAGS_DEF_WRAPPER(string, ::fLS::clstring, name, default_value, help_str)

// DECLARE_typed_var should be used in header files and in the global namespace.
#define C10_GFLAGS_DECLARE_WRAPPER(type, real_type, name) DECLARE_##type(name);

#define C10_DECLARE_int(name) \
  C10_GFLAGS_DECLARE_WRAPPER(int32, gflags::int32, name)
#define C10_DECLARE_int32(name) C10_DECLARE_int(name)
#define C10_DECLARE_int64(name) \
  C10_GFLAGS_DECLARE_WRAPPER(int64, gflags::int64, name)
#define C10_DECLARE_double(name) \
  C10_GFLAGS_DECLARE_WRAPPER(double, double, name)
#define C10_DECLARE_bool(name) C10_GFLAGS_DECLARE_WRAPPER(bool, bool, name)
#define C10_DECLARE_string(name) \
  C10_GFLAGS_DECLARE_WRAPPER(string, ::fLS::clstring, name)

////////////////////////////////////////////////////////////////////////////////
// End gflags section.
////////////////////////////////////////////////////////////////////////////////

#else // C10_USE_GFLAGS

#error "not supported when C10_USE_GFLAGS is not defined"

#endif // C10_USE_GFLAGS

