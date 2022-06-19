//==---------------- annotated_ptr.hpp - SYCL annotated_ptr ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/annotated_ptr/annotated_ref.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {
template <typename T, typename PropertyListT = empty_properties_t>
class annotated_ptr_base {
public:
  annotated_ptr_base() {}
  annotated_ptr_base(T *Ptr) : MPtr(Ptr) {}
  // Dummy pointer to prevent large template errors when invalid property lists
  // are passed.
  T *MPtr = nullptr;
};

// TODO: Add kernel argument attributes.
template <typename T, typename... Props>
class annotated_ptr_base<T, properties_t<Props...>> {
public:
  annotated_ptr_base() {}
  annotated_ptr_base(T *Ptr) : MPtr(Ptr) {}
  T *MPtr
#ifdef __SYCL_DEVICE_ONLY__
      [[__sycl_detail__::add_ir_annotations_member(
          PropertyMetaName<Props>::value...,
          PropertyMetaValue<Props>::value...)]]
#endif
      = nullptr;
};
} // namespace detail

template <typename T, typename PropertyListT = detail::empty_properties_t>
class annotated_ptr : private detail::annotated_ptr_base<T, PropertyListT> {
public:
  static_assert(is_property_list<PropertyListT>::value,
                "Property list is invalid.");

  using reference = annotated_ref<T, PropertyListT>;

  annotated_ptr() noexcept {}

  explicit annotated_ptr(T *Ptr, const PropertyListT &) noexcept
      : detail::annotated_ptr_base(Ptr) {}

  annotated_ptr(annotated_ptr const &Other) noexcept = default;

  template <typename P>
  explicit annotated_ptr(annotated_ptr<T, P> const &ConvertFrom) noexcept;

  template <typename PropertyListU, typename PropertyListV>
  explicit annotated_ptr(annotated_ptr<T, PropertyListU> const &,
                         properties<PropertyListV>) noexcept;

  reference operator*() const noexcept { return reference(MPtr); }

  reference operator[](std::ptrdiff_t Idx) const noexcept {
    return reference(MPtr + Idx);
  }

  annotated_ptr operator+(size_t Offset) const noexcept {
    return annotated_ptr(MPtr + Offset);
  }

  std::ptrdiff_t operator-(annotated_ptr FromPtr) const noexcept {
    return MPtr - FromPtr.MPtr;
  }

  operator bool() const noexcept { return MPtr; }

  T *get() noexcept { return MPtr; }

  const T *get() const noexcept { return MPtr; }

  annotated_ptr &operator=(const T *Ptr) noexcept {
    MPtr = Ptr;
    return *this;
  }

  annotated_ptr &operator=(annotated_ptr const &Other) noexcept {
    MPtr = Other.MPtr;
    return *this;
  }

  annotated_ptr &operator++() noexcept {
    MPtr++;
    return *this;
  }

  annotated_ptr operator++(int) noexcept {
    annotated_ptr NewAnnotPtr(*this);
    operator++();
    return NewAnnotPtr;
  }

  annotated_ptr &operator--() noexcept {
    MPtr--;
    return *this;
  }

  annotated_ptr operator--(int) noexcept {
    annotated_ptr NewAnnotPtr(*this);
    operator--();
    return NewAnnotPtr;
  }

  template <typename PropertyT> static constexpr bool has_property() {
    return PropertyListT::template has_property<PropertyT>();
  }

  template <typename PropertyT> static constexpr auto get_property() {
    return PropertyListT::template get_property<PropertyT>();
  }
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
