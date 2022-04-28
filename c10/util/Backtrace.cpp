#include <optional>

#include <c10/util/Backtrace.h>

#define SUPPORTS_BACKTRACE 1

#if SUPPORTS_BACKTRACE
#include <cxxabi.h>
#include <execinfo.h>
#endif // SUPPORTS_BACKTRACE

namespace c10
{

namespace
{
struct FrameInformation {
    /// If available, the demangled name of the function at this frame, else
    /// whatever (possibly mangled) name we got from `backtrace()`.
    std::string function_name;
    /// This is a number in hexadecimal form (e.g. "0xdead") representing the
    /// offset into the function's machine code at which the function's body
    /// starts, i.e. skipping the "prologue" that handles stack manipulation and
    /// other calling convention things.
    std::string offset_into_function;
    /// NOTE: In debugger parlance, the "object file" refers to the ELF file that
    /// the symbol originates from, i.e. either an executable or a library.
    std::string object_file;
};

bool is_python_frame(const FrameInformation& frame) {
    return frame.object_file == "python" || frame.object_file == "python3" ||
        (frame.object_file.find("libpython") != std::string::npos);
}

std::optional<FrameInformation> parse_frame_information(
    const std::string& frame_string) {
    FrameInformation frame;
    // This is the function name in the CXX ABI mangled format, e.g. something
    // like _Z1gv. Reference:
    // https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling
    std::string mangled_function_name;

    return std::nullopt;
}

} // namespace

std::string get_backtrace(
    size_t frames_to_skip,
    size_t maximum_number_of_frames,
    bool skip_python_frames) {

    return "";
}

} // namespace c10
