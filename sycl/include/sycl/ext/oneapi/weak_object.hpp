//==-------------- weak_object.hpp --- SYCL weak objects -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {

template <typename SYCLObjT> class weak_object {
public:
  using object_type = SYCLObjT;

  constexpr weak_object() noexcept : MObjWeakPtr() {}
  weak_object(const SYCLObjT &SYCLObj) noexcept
      : MObjWeakPtr(sycl::detail::getSyclObjImpl(SYCLObj)) {}
  weak_object(const weak_object &Other) noexcept : MObjWeakPtr(Other) {}
  weak_object(weak_object &&Other) noexcept : MObjWeakPtr(Other) {}

  weak_object &operator=(const SYCLObjT &SYCLObj) noexcept {
    MObjWeakPtr = sycl::detail::getSyclObjImpl(SYCLObj);
    return *this;
  }
  weak_object &operator=(const weak_object &Other) noexcept {
    MObjWeakPtr = Other.MObjWeakPtr;
    return *this;
  }
  weak_object &operator=(weak_object &&Other) noexcept {
    MObjWeakPtr = std::move(Other.MObjWeakPtr);
    return *this;
  }

  void reset() noexcept { MObjWeakPtr.reset(); }
  void swap(weak_object &Other) noexcept {
    MObjWeakPtr.swap(Other.MObjWeakPtr);
  }

  bool expired() const noexcept { return MObjWeakPtr.expired(); }

  std::optional<SYCLObjT> try_lock() const noexcept {
    auto MObjImplPtr = MObjWeakPtr.lock();
    if (!MObjImplPtr)
      return std::nullopt;
    return sycl::detail::createSyclObjFromImpl(MObjWeakPtr);
  }
  SYCLObjT lock() const {
     std::optional<SYCLObjT> OptionalObj = try_lock();
     if (!OptionalObj)
       throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                             "Referenced object has expired.");
    return *OptionalObj;
  }

  bool owner_before(const SYCLObjT &Other) const noexcept {
    return MObjWeakPtr.owner_before(sycl::detail::getSyclObjImpl(Other));
  }
  bool owner_before(const weak_object &Other) const noexcept {
    return MObjWeakPtr.owner_before(Other.MObjWeakPtr);
  }

private:
  typename decltype(SYCLObjT::impl)::weak_type MObjWeakPtr;
};

} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
