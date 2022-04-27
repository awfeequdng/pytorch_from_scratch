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


#endif // C10_USE_GFLAGS