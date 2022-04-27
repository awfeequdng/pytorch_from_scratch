#include <gtest/gtest.h>

#include <c10/util/Flags.h>

#include <iostream>

C10_DEFINE_bool(c10_flags_test_only_flag, true, "Only used in test.");

namespace c10_test
{

TEST(FlagsTest, TestGflagsCorrectness) {
    std::cout << c10::UsageMessage() << std::endl;
#ifdef C10_USE_GFLAGS
    EXPECT_EQ(FLAGS_c10_flags_test_only_flag, true);
    FLAGS_c10_flags_test_only_flag = false;
    FLAGS_c10_flags_test_only_flag = true;
    EXPECT_EQ(FLAGS_c10_flags_test_only_flag, true);
#else // C10_USE_GFLAGS
    std::cout << "Caffe2 is not built with gflags. Nothing to test here."
              << std::endl;
#endif
}

} // namespace c10_test
