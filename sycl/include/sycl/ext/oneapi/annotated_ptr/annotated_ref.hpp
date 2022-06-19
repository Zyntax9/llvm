//==---------------- annotated_ref.hpp - SYCL annotated_ref ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/properties.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {
template <typename T, typename PropertyListT = empty_properties_t>
class annotated_ref_base {
public:
  annotated_ref_base() {}
  annotated_ref_base(T *Ptr) : MPtr(Ptr) {}
  // Dummy pointer to prevent large template errors when invalid property lists
  // are passed.
  T *MPtr = nullptr;
};

// TODO: Add kernel argument attributes.
template <typename T, typename... Props>
class annotated_ref_base<T, properties_t<Props...>> {
public:
  annotated_ref_base() {}
  annotated_ref_base(T *Ptr) : MPtr(Ptr) {}
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
class annotated_ref : private detail::annotated_ref_base<T, PropertyListT> {
public:
  static_assert(is_property_list<PropertyListT>::value,
                "Property list is invalid.");

  annotated_ref(T *Ptr) : detail::annotated_ref_base(Ptr) {} 
  operator T() noexcept { return *MPtr; }
  operator const T() const noexcept { return *MPtr; }
  void operator=(const T &NewValue) { *MPtr = NewValue; }
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
