#include <gtest/gtest.h>
#include <iostream>
#include <memory>

#include <c10/util/Registry.h>

namespace c10_test
{
class Foo {
public:
    explicit Foo(int x) {

    }
    virtual ~Foo() = default;
};

C10_DECLARE_REGISTRY(FooRegistry, Foo, int);
C10_DEFINE_REGISTRY(FooRegistry, Foo, int);
#define REGISTER_FOO(clsname) C10_REGISTER_CLASS(FooRegistry, clsname, clsname)

class Bar : public Foo {
 public:
  explicit Bar(int x) : Foo(x) {
    // LOG(INFO) << "Bar " << x;
  }
};
REGISTER_FOO(Bar);


} // namespace c10_test
