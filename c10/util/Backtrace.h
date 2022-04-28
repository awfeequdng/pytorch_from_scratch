#pragma once
#include <cstddef>
#include <string>

namespace c10
{
std::string get_backtrace(
    size_t frames_to_skip = 0,
    size_t max_num_of_frames = 64,
    bool skip_python_frames = true);

} // namespace c10
