//==----- properties.hpp - SYCL properties associated with annotated_ptr ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {

struct alignment_key {
  template<int K>
  using value_t = property_value<alignment_key, std::integral_constant<int, K>>;
};

struct restrict_key {
  using value_t = property_value<restrict_key>;
};

struct runtime_aligned_key {
  using value_t = property_value<runtime_aligned_key>;
};

template<int K>
inline constexpr alignment_key::value_t<K> alignment;
inline constexpr restrict_key::value_t restrict;
inline constexpr runtime_aligned_key::value_t runtime_aligned;

template<>
struct is_property_key<alignment_key> : std::true_type {};
template<>
struct is_property_key<restrict_key> : std::true_type {};
template<>
struct is_property_key<runtime_aligned_key> : std::true_type {};

template<typename T, typename PropertyListT>
struct is_property_key_of<
  alignment_key, annotated_ptr<T, PropertyListT>> : std::true_type {};
template<typename T, typename PropertyListT>
struct is_property_key_of<
  alignment_key, annotated_ref<T, PropertyListT>> : std::true_type {};
template<typename T, typename PropertyListT>
struct is_property_key_of<
  restrict_key, annotated_ptr<T, PropertyListT>> : std::true_type {};
template<typename T, typename PropertyListT>
struct is_property_key_of<
  restrict_key, annotated_ref<T, PropertyListT>> : std::true_type {};
template<typename T, typename PropertyListT>
struct is_property_key_of<
  runtime_aligned_key, annotated_ptr<T, PropertyListT>> : std::true_type {};
template<typename T, typename PropertyListT>
struct is_property_key_of<
  runtime_aligned_key, annotated_ref<T, PropertyListT>> : std::true_type {};

namespace detail {
template <> struct PropertyToKind<alignment_key> {
  static constexpr PropKind Kind = PropKind::Alignment;
};
template <> struct PropertyToKind<restrict_key> {
  static constexpr PropKind Kind = PropKind::Restrict;
};
template <> struct PropertyToKind<runtime_aligned_key> {
  static constexpr PropKind Kind = PropKind::RuntimeAligned;
};

template <>
struct IsCompileTimeProperty<alignment_key> : std::true_type {};
template <>
struct IsCompileTimeProperty<restrict_key> : std::true_type {};
template <>
struct IsCompileTimeProperty<runtime_aligned_key> : std::true_type {};

template <int K> struct PropertyMetaName<alignment_key::value_t<K>> {
  static constexpr const char *value = "sycl-alignment";
};
template <>
struct PropertyMetaName<restrict_key::value_t> {
  static constexpr const char *value = "sycl-restrict";
};
template <>
struct PropertyMetaName<runtime_aligned_key::value_t> {
  static constexpr const char *value = "sycl-runtime-aligned";
};

template <int K>
struct PropertyMetaValue<alignment_key::value_t<K>> {
  static constexpr int value = K;
};

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
