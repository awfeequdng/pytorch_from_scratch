#pragma once

#include <memory>

/// MaybeOwnedTraits<T> describes how to borrow from T.  Here is how we
/// can implement borrowing from an arbitrary type T using a raw
/// pointer to const:
template <typename T>
struct MaybeOwnedTraitsGenericImpl {
    using owned_type = T;
    using borrow_type = const T*;

    static borrow_type createBorrow(const owned_type& from) {
        return &from;
    }

    static void assignBorrow(borrow_type& lhs, borrow_type rhs) {
        lhs = rhs;
    }

    static void destroyBorrow(borrow_type& /*toDestroy*/) {}

    static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
        return *borrow;
    }

    static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
        return borrow;
    }

    static bool debugBorrowIsValid(const borrow_type& borrow) {
        return borrow != nullptr;
    }
};

template <typename T>
struct MaybeOwnedTraits;

// Explicitly enable MaybeOwned<shared_ptr<T>>, rather than allowing
// MaybeOwned to be used for any type right away.
template <typename T>
struct MaybeOwnedTraits<std::shared_ptr<T>>
    : public MaybeOwnedTraitsGenericImpl<std::shared_ptr<T>> {};
