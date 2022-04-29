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
} // namespace


void SetStackTraceFetcher(std::function<string(void)> fetcher) {
    *GetFetchStackTrace() = fetcher;
}

void ThrowEnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg,
    const void* caller) {
    c10::Error e(file, line, condition, msg, (*GetFetchStackTrace())(), caller);
    if (FLAGS_caffe2_use_fatal_for_enforce) {
        LOG(FATAL) << e.msg();
    }
    throw e;
}

void ThrowEnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const char* msg,
    const void* caller) {
    ThrowEnforceNotMet(file, line, condition, std::string(msg), caller);
}

void ThrowEnforceFiniteNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg,
    const void* caller) {
    throw c10::EnforceFiniteError(
        file, line, condition, msg, (*GetFetchStackTrace())(), caller);
}

void ThrowEnforceFiniteNotMet(
    const char* file,
    const int line,
    const char* condition,
    const char* msg,
    const void* caller) {
  ThrowEnforceFiniteNotMet(file, line, condition, std::string(msg), caller);
}

// PyTorch-style error message
// (This must be defined here for access to GetFetchStackTrace)
Error::Error(SourceLocation source_location, std::string msg)
    : Error(
          std::move(msg),
          str("Exception raised from ",
              source_location,
              " (most recent call first):\n",
              (*GetFetchStackTrace())())) {}

using APIUsageLoggerType = std::function<void(const std::string&)>;
using DDPUsageLoggerType = std::function<void(const DDPLoggingData&)>;


namespace {
bool IsAPIUsageDebugMode() {
  const char* val = getenv("PYTORCH_API_USAGE_STDERR");
  return val && *val; // any non-empty value
}

void APIUsageDebug(const string& event) {
  // use stderr to avoid messing with glog
  std::cerr << "PYTORCH_API_USAGE " << event << std::endl;
}

APIUsageLoggerType* GetAPIUsageLogger() {
  static APIUsageLoggerType func =
      IsAPIUsageDebugMode() ? &APIUsageDebug : [](const string&) {};
  return &func;
};

DDPUsageLoggerType* GetDDPUsageLogger() {
  static DDPUsageLoggerType func = [](const DDPLoggingData&) {};
  return &func;
};
} // namespace


void SetAPIUsageLogger(std::function<void(const std::string&)> logger) {
    TORCH_CHECK(logger);
    *GetAPIUsageLogger() = logger;
}

void SetPyTorchDDPUsageLogger(
    std::function<void(const DDPLoggingData&)> logger) {
    TORCH_CHECK(logger);
    *GetDDPUsageLogger() = logger;
}

void LogAPIUsage(const std::string& event) try {
  if (auto logger = GetAPIUsageLogger())
    (*logger)(event);
} catch (std::bad_function_call&) {
  // static destructor race
}

void LogPyTorchDDPUsage(const DDPLoggingData& ddpData) try {
  if (auto logger = GetDDPUsageLogger())
    (*logger)(ddpData);
} catch (std::bad_function_call&) {
  // static destructor race
}

namespace detail {
bool LogAPIUsageFakeReturn(const std::string& event) try {
  if (auto logger = GetAPIUsageLogger())
    (*logger)(event);
  return true;
} catch (std::bad_function_call&) {
  // static destructor race
  return true;
}

namespace {

void setLogLevelFlagFromEnv();

} // namespace
} // namespace detail
} // namespace c10

#if defined(C10_USE_GFLAGS) && defined(C10_USE_GLOG)
// When GLOG depends on GFLAGS, these variables are being defined in GLOG
// directly via the GFLAGS definition, so we will use DECLARE_* to declare
// them, and use them in Caffe2.
// GLOG's minloglevel
DECLARE_int32(minloglevel);
// GLOG's verbose log value.
DECLARE_int32(v);
// GLOG's logtostderr value
DECLARE_bool(logtostderr);
#endif // defined(C10_USE_GFLAGS) && defined(C10_USE_GLOG)

#if !defined(C10_USE_GLOG)
// This backward compatibility flags are in order to deal with cases where
// Caffe2 are not built with glog, but some init flags still pass in these
// flags. They may go away in the future.
C10_DEFINE_int32(minloglevel, 0, "Equivalent to glog minloglevel");
C10_DEFINE_int32(v, 0, "Equivalent to glog verbose");
C10_DEFINE_bool(logtostderr, false, "Equivalent to glog logtostderr");
#endif // !defined(c10_USE_GLOG)

#ifdef C10_USE_GLOG
// Provide easy access to the above variables, regardless whether GLOG is
// dependent on GFLAGS or not. Note that the namespace (fLI, fLB) is actually
// consistent between GLOG and GFLAGS, so we can do the below declaration
// consistently.
namespace c10 {
using fLB::FLAGS_logtostderr;
using fLI::FLAGS_minloglevel;
using fLI::FLAGS_v;
} // namespace c10

C10_DEFINE_int(
    caffe2_log_level,
    google::GLOG_WARNING,
    "The minimum log level that caffe2 will output.");

// Google glog's api does not have an external function that allows one to check
// if glog is initialized or not. It does have an internal function - so we are
// declaring it here. This is a hack but has been used by a bunch of others too
// (e.g. Torch).
namespace google {
namespace glog_internal_namespace_ {
bool IsGoogleLoggingInitialized();
} // namespace glog_internal_namespace_
} // namespace google

namespace c10 {
namespace {

void initGoogleLogging(char const* name) {
  // This trick can only be used on UNIX platforms
//   if (!::google::glog_internal_namespace_::IsGoogleLoggingInitialized())
  {
    ::google::InitGoogleLogging(name);
#if !defined(__XROS__)
    ::google::InstallFailureSignalHandler();
#endif
  }
}

} // namespace

void initLogging() {
  detail::setLogLevelFlagFromEnv();

  UpdateLoggingLevelsFromFlags();
}

bool InitCaffeLogging(int* argc, char** argv) {
  if (*argc == 0) {
    return true;
  }

  initGoogleLogging(argv[0]);

  UpdateLoggingLevelsFromFlags();

  return true;
}

void UpdateLoggingLevelsFromFlags() {
#ifdef FBCODE_CAFFE2
  // TODO(T82645998): Fix data race exposed by TSAN.
  folly::annotate_ignore_thread_sanitizer_guard g(__FILE__, __LINE__);
#endif
  // If caffe2_log_level is set and is lower than the min log level by glog,
  // we will transfer the caffe2_log_level setting to glog to override that.
  FLAGS_minloglevel = std::min(FLAGS_caffe2_log_level, FLAGS_minloglevel);
  // If caffe2_log_level is explicitly set, let's also turn on logtostderr.
  if (FLAGS_caffe2_log_level < google::GLOG_WARNING) {
    FLAGS_logtostderr = 1;
  }
  // Also, transfer the caffe2_log_level verbose setting to glog.
  if (FLAGS_caffe2_log_level < 0) {
    FLAGS_v = std::min(FLAGS_v, -FLAGS_caffe2_log_level);
  }
}

void ShowLogInfoToStderr() {
  FLAGS_logtostderr = 1;
  FLAGS_minloglevel = std::min(FLAGS_minloglevel, google::GLOG_INFO);
}
} // namespace c10

namespace c10 {
namespace detail {
namespace {

void setLogLevelFlagFromEnv() {
  const char* level_str = std::getenv("TORCH_CPP_LOG_LEVEL");

  // Not set, fallback to the default level (i.e. WARNING).
  std::string level{level_str != nullptr ? level_str : ""};
  if (level.empty()) {
    return;
  }

  std::transform(
      level.begin(), level.end(), level.begin(), [](unsigned char c) {
        return toupper(c);
      });

  if (level == "0" || level == "INFO") {
    FLAGS_caffe2_log_level = 0;

    return;
  }
  if (level == "1" || level == "WARNING") {
    FLAGS_caffe2_log_level = 1;

    return;
  }
  if (level == "2" || level == "ERROR") {
    FLAGS_caffe2_log_level = 2;

    return;
  }
  if (level == "3" || level == "FATAL") {
    FLAGS_caffe2_log_level = 3;

    return;
  }

  std::cerr
      << "`TORCH_CPP_LOG_LEVEL` environment variable cannot be parsed. Valid values are "
         "`INFO`, `WARNING`, `ERROR`, and `FATAL` or their numerical equivalents `0`, `1`, "
         "`2`, and `3`."
      << std::endl;
}

} // namespace
} // namespace detail
} // namespace c10

#else // C10_USE_GLOG
# error "only support glog"

#endif // C10_USE_GLOG