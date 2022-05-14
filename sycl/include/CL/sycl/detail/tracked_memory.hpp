//==------------ tracked_memory.hpp - Tracked memory allocations -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>

#include <memory>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

#ifndef SYCL_ENABLE_TRACKED_MEMORY_ALLOCATIONS
template <typename T> class TrackedSharedPtr : public std::shared_ptr<T> {
public:
  constexpr TrackedSharedPtr() noexcept : std::shared_ptr<T>() {}
  constexpr TrackedSharedPtr(std::nullptr_t) noexcept
      : std::shared_ptr<T>(nullptr) {}
  template <class Y>
  explicit TrackedSharedPtr(Y *ptr) : std::shared_ptr<T>(ptr) {}
  template <class Y, class Deleter>
  TrackedSharedPtr(Y *ptr, Deleter d) : std::shared_ptr<T>(ptr, d) {}
  template <class Deleter>
  TrackedSharedPtr(std::nullptr_t ptr, Deleter d)
      : std::shared_ptr<T>(ptr, d) {}
  template <class Y, class Deleter, class Alloc>
  TrackedSharedPtr(Y *ptr, Deleter d, Alloc alloc)
      : std::shared_ptr<T>(ptr, d, alloc) {}
  template <class Deleter, class Alloc>
  TrackedSharedPtr(std::nullptr_t ptr, Deleter d, Alloc alloc)
      : std::shared_ptr<T>(ptr, d, alloc) {}

  TrackedSharedPtr(const TrackedSharedPtr &r) noexcept
      : std::shared_ptr<T>(r) {}
  template <class Y>
  TrackedSharedPtr(const TrackedSharedPtr<Y> &r) noexcept
      : std::shared_ptr<T>(r) {}
  TrackedSharedPtr(TrackedSharedPtr &&r) noexcept : std::shared_ptr<T>(r) {}
  template <class Y>
  TrackedSharedPtr(TrackedSharedPtr<Y> &&r) noexcept : std::shared_ptr<T>(r) {}

  TrackedSharedPtr(const std::shared_ptr<T> &r) noexcept
      : std::shared_ptr<T>(r) {}
  template <class Y>
  TrackedSharedPtr(const std::shared_ptr<Y> &r) noexcept
      : std::shared_ptr<T>(r) {}
  TrackedSharedPtr(std::shared_ptr<T> &&r) noexcept : std::shared_ptr<T>(r) {}
  template <class Y>
  TrackedSharedPtr(std::shared_ptr<Y> &&r) noexcept : std::shared_ptr<T>(r) {}

  TrackedSharedPtr& operator=(const TrackedSharedPtr& other) {
      std::shared_ptr<T>::operator=(other);
      return *this;
  }
};

template <typename T> using shared_ptr = TrackedSharedPtr<T>;
#else
template <typename T> using shared_ptr = std::shared_ptr<T>;
#endif

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)


#ifndef SYCL_ENABLE_TRACKED_MEMORY_ALLOCATIONS
namespace std {
template <typename T> struct hash<cl::sycl::detail::TrackedSharedPtr<T>> {
  size_t operator()(const cl::sycl::detail::TrackedSharedPtr<T> &ptr) const {
    return hash<std::shared_ptr<T>>()(
        static_cast<const std::shared_ptr<T> &>(ptr));
  }
};
} // namespace std
#endif
