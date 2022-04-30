#pragma once
#include <atomic>
#include <cstddef>
#include <climits>

#include <c10/util/Exception.h>
// #include <c10/util/MaybeOwned.h>

namespace c10
{
namespace raw
{
// constructor tag used by intrusive_ptr constructors
struct DontIncreaseRefcount {};
} // namespace raw

class intrusive_ptr_target {
    mutable std::atomic<size_t> refcount_;
    mutable std::atomic<size_t> weakcount_;

protected:
    virtual ~intrusive_ptr_target() {
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            // Second condition is there to accommodate
            // unsafe_adapt_non_heap_allocated: since we are doing our own
            // deallocation in that case, it is correct for each
            // expected_decref to have happened (some user code tried to
            // decref and thus free the object, but it didn't happen right
            // away) or not (no user code tried to free the object, and
            // now it's getting destroyed through whatever mechanism the
            // caller of unsafe_adapt_non_heap_allocated wanted to
            // use). We choose our reference count such that the count
            // will not dip below INT_MAX regardless.
            refcount_.load() == 0 || refcount_.load() >= INT_MAX,
            "Tried to destruct an intrusive_ptr_target that still has intrusive_ptr to it; refcount was ",
            refcount_.load());
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            // See ~intrusive_ptr for optimization that will frequently result in 1
            // at destruction time.
            weakcount_.load() == 1 || weakcount_.load() == 0 ||
                weakcount_.load() == INT_MAX - 1 || weakcount_.load() == INT_MAX,
            "Tried to destruct an intrusive_ptr_target that still has weak_intrusive_ptr to it");
    }

    constexpr intrusive_ptr_target() noexcept : refcount_(0), weakcount_(0) {}

    // intrusive_ptr_target supports copy and move: but refcount and weakcount
    // don't participate (since they are intrinsic properties of the memory
    // location)
    intrusive_ptr_target(intrusive_ptr_target&& /*other*/) noexcept
        : intrusive_ptr_target() {}

    intrusive_ptr_target& operator=(intrusive_ptr_target&& /*other*/) noexcept {
        return *this;
    }

    intrusive_ptr_target(const intrusive_ptr_target& /*other*/) noexcept
        : intrusive_ptr_target() {}

    intrusive_ptr_target& operator=(
        const intrusive_ptr_target& /*other*/) noexcept {
        return *this;
    }

    private:
    /**
     * This is called when refcount reaches zero.
     * You can override this to release expensive resources.
     * There might still be weak references, so your object might not get
     * destructed yet, but you can assume the object isn't used anymore,
     * i.e. no more calls to methods or accesses to members (we just can't
     * destruct it yet because we need the weakcount accessible).
     *
     * Even if there are no weak references (i.e. your class is about to be
     * destructed), this function is guaranteed to be called first.
     * However, if you use your class for an object on the stack that is
     * destructed by the scope (i.e. without intrusive_ptr), this function will
     * not be called.
     */
    virtual void release_resources() {}
};

namespace detail
{
template <class TTarget>
struct intrusive_target_default_null_type final {
    static constexpr TTarget* singleton() noexcept {
        return nullptr;
    }
};

template <class TTarget, class ToNullType, class FromNullType>
TTarget* assign_ptr_(TTarget* rhs) {
    if (FromNullType::singleton() == rhs) {
        return ToNullType::singleton();
    } else {
        rhs;
    }
}

// Increment needs to be acquire-release to make use_count() and
// unique() reliable.
inline size_t atomic_refcount_increment(std::atomic<size_t>& refcount) {
  return refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
}

// weak_use_count() is only used for testing, so we don't need it to
// be reliable. Relaxed should be fine.
inline size_t atomic_weakcount_increment(std::atomic<size_t>& weakcount) {
  return weakcount.fetch_add(1, std::memory_order_relaxed) + 1;
}

// Both decrements need to be acquire-release for correctness. See
// e.g. std::shared_ptr implementation.
inline size_t atomic_refcount_decrement(std::atomic<size_t>& refcount) {
  return refcount.fetch_sub(1, std::memory_order_acq_rel) - 1;
}

inline size_t atomic_weakcount_decrement(std::atomic<size_t>& weakcount) {
  return weakcount.fetch_sub(1, std::memory_order_acq_rel) - 1;
}

} // namespace detail

template <
    class TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>>
class intrusive_ptr final {

    // error C2131: expression did not evaluate to a constant
    // 感觉多此一举
    static_assert(NullType::singleton() == NullType::singleton(),
          "NullType must have a constexpr singleton() method");

    static_assert(
        std::is_base_of<
            TTarget,
            typename std::remove_pointer<decltype(NullType::singleton())>::
            type>::value,
        "NullType::singleton() must return a element_type* pointer");

    TTarget* target_;

    // todo: 在什么情况下执行reatin_???
    void retain_() {
        if (target_ != NullType::singleton()) {
            size_t new_refcount =
                detail::atomic_refcount_increment(target_->refcount_);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          new_refcount != 1,
          "intrusive_ptr: Cannot increase refcount after it reached zero.");
        }
    }

    void reset_() noexcept {
        if (target_ != NullType::singleton() &&
            detail::atomic_refcount_decrement(target_->refcount_) == 0) {
            // justification for const_cast: release_resources is basically a
            // destructor and a destructor always mutates the object, even for const
            // objects. NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDelete)
            // todo: 为什么去掉了const属性后，又进行const_cast???
            const_cast<std::remove_const_t<TTarget>*>(target_)->release_resources();
            // todo: 这里这番释放操作是为了什么？？
            // 为什么不会有多次delete的问题？
            if (target_->weakcount_.load(std::memory_order_acquire) == 1 ||
                detail::atomic_weakcount_decrement(target_->weakcount_) == 0) {
                delete target_;
            }
        }
        target_ = NullType::singleton();
    }

    // raw pointer constructors are not public because we shouldn't make
    // intrusive_ptr out of raw pointers except from inside the make_intrusive(),
    // reclaim() and weak_intrusive_ptr::lock() implementations.

    // This constructor will increase the ref counter for you.
    // This constructor will be used by the make_intrusive(), and also pybind11,
    // which wrap the intrusive_ptr holder around the raw pointer and incref
    // correspondingly (pybind11 requires raw pointer constructor to incref by
    // default).
    explicit intrusive_ptr(TTarget* target)
        : intrusive_ptr(target, raw::DontIncreaseRefcount{}) {
        if (target_ != NullType::singleton()) {
            // We just created result.target_, so we know no other thread has
            // access to it, so we know we needn't care about memory ordering.
            // (On x86_64, a store with memory_order_relaxed generates a plain old
            // `mov`, whereas an atomic increment does a lock-prefixed `add`, which is
            // much more expensive: https://godbolt.org/z/eKPzj8.)
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
                target_->refcount_ == 0 && target_->weakcount_ == 0,
                "intrusive_ptr: Newly-created target had non-zero refcounts. Does its "
                "constructor do something strange like incref or create an "
                "intrusive_ptr from `this`?");
            target_->refcount_.store(1, std::memory_order_relaxed);
            target_->weakcount_.store(1, std::memory_order_relaxed);
        }
    }

public:
    using element_type = TTarget;

    intrusive_ptr() noexcept
        : intrusive_ptr(NullType::singleton(), raw::DontIncreaseRefcount{}) {}

    // This constructor will not increase the ref counter for you.
    // We use the tagged dispatch mechanism to explicitly mark this constructor
    // to not increase the refcount
    explicit intrusive_ptr(TTarget* target, raw::DontIncreaseRefcount) noexcept
        : target_(target) {}

    // TTarget指针被intrusive_ptr接管，引用计数加1
    // 之前由unique_ptr管理TTarget，引用计数应该为0
    explicit intrusive_ptr(std::unique_ptr<TTarget> rhs) noexcept
        : intrusive_ptr(rhs.release()) {}

    intrusive_ptr(intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
        rhs.target_ = NullType::singleton();
    }

    template <class From, class FromNullType>
    /* implicit */ intrusive_ptr(intrusive_ptr<From, FromNullType>&& rhs) noexcept
        : target_(
                detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
        static_assert(
            std::is_convertible<From*, TTarget*>::value,
            "Type mismatch. intrusive_ptr move constructor got pointer of wrong type.");
        rhs.target_ = FromNullType::singleton();
    }

    intrusive_ptr(const intrusive_ptr& rhs) : target_(rhs.target_) {
        retain_();
    }

    template <class From, class FromNullType>
    /* implicit */ intrusive_ptr(const intrusive_ptr<From, FromNullType>& rhs)
        : target_(
                detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
        static_assert(
            std::is_convertible<From*, TTarget*>::value,
            "Type mismatch. intrusive_ptr copy constructor got pointer of wrong type.");
        retain_();
    }

    ~intrusive_ptr() noexcept {
        reset_();
    }

    intrusive_ptr& operator=(intrusive_ptr&& rhs) & noexcept {
        return operator=<TTarget, NullType>(std::move(rhs));
    }

    template <class From, class FromNullType>
    intrusive_ptr& operator=(intrusive_ptr<From, FromNullType>&& rhs) & noexcept {
        static_assert(
            std::is_convertible<From*, TTarget*>::value,
            "Type mismatch. intrusive_ptr move assignment got pointer of wrong type.");
        // todo: 此处rhs的值没有被move掉啊？？？
        // 难道std::move操作执行为引用操作，然后swap将this的值和rhs的值交换了，并且this的值时空的？？？
        intrusive_ptr tmp = std::move(rhs);
        swap(tmp);
        return *this;
    }

    intrusive_ptr& operator=(const intrusive_ptr& rhs) & noexcept {
        return operator=<TTarget, NullType>(rhs);
    }

    template <class From, class FromNullType>
    intrusive_ptr& operator=(const intrusive_ptr<From, NullType>& rhs) & {
        static_assert(
            std::is_convertible<From*, TTarget*>::value,
            "Type mismatch. intrusive_ptr copy assignment got pointer of wrong type.");
        intrusive_ptr tmp = rhs;
        swap(tmp);
        return *this;
    }

    TTarget* get() const noexcept {
        return target_;
    }

    TTarget& operator*() const noexcept {
        return *target_;
    }

    TTarget* operator->() const noexcept {
        // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDelete)
        return target_;
    }

    operator bool() const noexcept {
        return target_ != NullType::singleton();
    }

    void reset() noexcept {
        reset_();
    }

    void swap(intrusive_ptr& rhs) noexcept {
        TTarget* tmp = target_;
        target_ = rhs.target_;
        rhs.target_ = tmp;
    }

    // We do a lot of null-pointer checks in our code, good to have this be cheap.
    bool defined() const noexcept {
        return target_ != NullType::singleton();
    }

    size_t use_count() const noexcept {
        if (target_ == NullType::singleton()) {
        return 0;
        }
        return target_->refcount_.load(std::memory_order_acquire);
    }

    size_t weak_use_count() const noexcept {
        if (target_ == NullType::singleton()) {
        return 0;
        }
        return target_->weakcount_.load(std::memory_order_acquire);
    }

    bool unique() const noexcept {
        return use_count() == 1;
    }

    /**
     * Returns an owning (!) pointer to the underlying object and makes the
     * intrusive_ptr instance invalid. That means the refcount is not decreased.
     * You *must* put the returned pointer back into a intrusive_ptr using
     * intrusive_ptr::reclaim(ptr) to properly destruct it.
     * This is helpful for C APIs.
     */
    TTarget* release() noexcept {
        // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
        TTarget* result = target_;
        target_ = NullType::singleton();
        return result;
    }

    /**
     * Takes an owning pointer to TTarget* and creates an intrusive_ptr that takes
     * over ownership. That means the refcount is not increased.
     * This is the counter-part to intrusive_ptr::release() and the pointer
     * passed in *must* have been created using intrusive_ptr::release().
     */
    static intrusive_ptr reclaim(TTarget* owning_ptr) {
        return intrusive_ptr(owning_ptr, raw::DontIncreaseRefcount{});
    }

    /**
     * Takes an owning pointer to TTarget* and creates an intrusive_ptr
     * representing a new reference, i.e. the raw pointer retains
     * ownership.
     */
    static intrusive_ptr reclaim_copy(TTarget* owning_ptr) {
        auto ret = reclaim(owning_ptr);
        // 此时如果refcount引用计数为0的话，下面的retain_会报错
        // todo:
        ret.retain_();
        return ret;
    }

    template <class...Args>
    static intrusive_ptr make(Args&... args) {
        return intrusive_ptr(new TTarget(std::forward<Args>(args)...));
    }

    /**
     * Turn a new instance of TTarget (e.g., literally allocated
     * using new TTarget(...) into an intrusive_ptr.  If possible,
     * use intrusive_ptr::make instead which statically guarantees
     * that the allocation was done properly.
     *
     * At the moment, the only reason this method exists is because
     * pybind11 holder types expect to be able to allocate in
     * this way (because pybind11 handles the new allocation itself).
     */
    static intrusive_ptr unsafe_steal_from_new(TTarget* raw_ptr) {
        return intrusive_ptr(raw_ptr);
    }
};


template <
    class TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>,
    class... Args>
inline intrusive_ptr<TTarget, NullType> make_intrusive(Args&&... args) {
  return intrusive_ptr<TTarget, NullType>::make(std::forward<Args>(args)...);
}

template <class TTarget, class NullType>
inline void swap(
    intrusive_ptr<TTarget, NullType>& lhs,
    intrusive_ptr<TTarget, NullType>& rhs) noexcept {
  lhs.swap(rhs);
}


// To allow intrusive_ptr inside std::map or std::set, we need operator<
template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator<(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.get() < rhs.get();
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator==(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.get() == rhs.get();
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator!=(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return !operator==(lhs, rhs);
}

// template <typename T>
// struct MaybeOwnedTraits<c10::intrusive_ptr<T>> {
//     using owned_type = c10::intrusive_ptr<T>;
//     using borrow_type = c10::intrusive_ptr<T>;

//     static borrow_type createBorrow(const owned_type& from) {
//         return borrow_type::reclaim(from.get());
//     }

//     static void assignBorrow(borrow_type& lhs, const borrow_type& rhs) {
//         lhs.release();
//         lhs = borrow_type::reclaim(rhs.get());
//     }

//     static void destroyBorrow(borrow_type& toDestroy) {
//         toDestroy.release();
//     }

//     static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
//         return borrow;
//     }

//     static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
//         return &borrow;
//     }

//     static bool debugBorrowIsValid(const borrow_type& /*borrow*/) {
//         return true;
//     }
// };

} // namespace c10
