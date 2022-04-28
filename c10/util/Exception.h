#pragma once

#include <exception>
#include <string>
#include <vector>

#include <c10/util/StringUtil.h>

namespace c10
{

class Error : public std::exception {
    std::string msg_;

    std::vector<std::string> context_;

    std::string backtrace_;

    std::string what_;
    std::string what_without_backtrace_;

  // This is a little debugging trick: you can stash a relevant pointer
  // in caller, and then when you catch the exception, you can compare
  // against pointers you have on hand to get more information about
  // where the exception came from.  In Caffe2, this is used to figure
  // out which operator raised an exception.
    const void*caller_;

public:
    Error(SourceLocation source_location, std::string msg);

      // Caffe2-style error message
    Error(
        const char* file,
        const uint32_t line,
        const char* condition,
        const std::string& msg,
        const std::string& backtrace,
        const void* caller = nullptr);

    // Base constructor
    Error(std::string msg, std::string backtrace, const void* caller = nullptr);


    // Add some new context to the message stack.  The last added context
    // will be formatted at the end of the context list upon printing.
    // WARNING: This method is O(n) in the size of the stack, so don't go
    // wild adding a ridiculous amount of context to error messages.
    void add_context(std::string msg);

    const std::string& msg() const {
        return msg_;
    }

    const std::vector<std::string>& context() const {
        return context_;
    }

    const std::string& backtrace() const {
        return backtrace_;
    }

    /// Returns the complete error message, including the source location.
    /// The returned pointer is invalidated if you call add_context() on
    /// this object.
    const char* what() const noexcept override {
        return what_.c_str();
    }

    const void* caller() const noexcept {
        return caller_;
    }

    /// Returns only the error message string, without source location.
    /// The returned pointer is invalidated if you call add_context() on
    /// this object.
    const char* what_without_backtrace() const noexcept {
        return what_without_backtrace_.c_str();
    }

    private:
    void refresh_what();
    std::string compute_what(bool include_backtrace) const;
};

} // namespace c10
