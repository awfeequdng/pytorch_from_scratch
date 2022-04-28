#include <functional>
#include <string>

#include <c10/util/Logging.h>
#include <c10/util/Backtrace.h>

// Common code that we use regardless of whether we use glog or not.

C10_DEFINE_bool(
    caffe2_use_fatal_for_enforce,
    false,
    "If set true, when CAFFE_ENFORCE is not met, abort instead "
    "of throwing an exception.");

namespace c10
{

namespace {
std::function<std::string(void)> * GetFetchStackTrace() {
    static std::function<std::string(void)> func = []() {
        return get_backtrace(/*frames_to_skip=*/1);
    };
    return &func;
}
}

} // namespace c10
