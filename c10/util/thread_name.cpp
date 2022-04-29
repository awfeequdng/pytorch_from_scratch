#include <c10/util/thread_name.h>

#include <pthread.h>

namespace c10 {

void setThreadName(std::string name) {
    constexpr size_t kMaxThreadName = 15;
    name.resize(std::min(name.size(), kMaxThreadName));
#if defined(__APPLE__)
    pthread_setname_np(name.c_str());
#else /* if defined(__GLIBC__) */
    pthread_setname_np(pthread_self(), name.c_str());
#endif //
}

} // namespace c10