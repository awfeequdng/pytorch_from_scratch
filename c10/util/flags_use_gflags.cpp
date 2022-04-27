#include <c10/util/Flags.h>

#include <string>

#include <c10/macros/Macros.h>

#ifdef C10_USE_GFLAGS

namespace c10 {

using std::string;

void SetUsageMessage(const string& str) {
    if (UsageMessage() != nullptr) {
        // Usage message has already been set, so we will simply return.
        return;
    }
    gflags::SetUsageMessage(str);
}

const char* UsageMessage() {
    return gflags::ProgramUsage();
}

bool ParseCommandLineFlags(int* pargc, char*** pargv) {
    // In case there is no commandline flags to parse, simply return.
    if (*pargc == 0)
        return true;
    return gflags::ParseCommandLineFlags(pargc, pargv, true);
}

bool CommandLineFlagsHasBeenParsed() {
    // There is no way we query gflags right now, so we will simply return true.
    return true;
}

} // namespace c10
#endif // C10_USE_GFLAGS