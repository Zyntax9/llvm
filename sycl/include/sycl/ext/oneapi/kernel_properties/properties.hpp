//==------- properties.hpp - SYCL properties associated with kernels -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

#include <array>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {
// Trait for checking that all size_t values are non-zero.
template <size_t... Xs> struct AllNonZero {
  static inline constexpr bool value = true;
};
template <size_t X, size_t... Xs> struct AllNonZero<X, Xs...> {
  static inline constexpr bool value = X > 0 && AllNonZero<Xs...>::value;
};

// Simple helpers for containing primitive types as template arguments.
template <size_t... Sizes> struct SizeList {};
template <char... Sizes> struct CharList {};

// Helper for converting characters to a constexpr string.
template <char... Chars> struct CharsToStr {
  static inline constexpr const char value[] = {Chars..., '\0'};
};

// Helper for converting a list of size_t values to a comma-separated string
// representation. This is done by extracting the digit one-by-one and when
// finishing a value, the parsed result is added to a separate list of
// "parsed" characters with the delimiter.
template <typename List, typename ParsedList, char... Chars>
struct SizeListToStrHelper;
template <size_t Value, size_t... Values, char... ParsedChars, char... Chars>
struct SizeListToStrHelper<SizeList<Value, Values...>, CharList<ParsedChars...>,
                           Chars...>
    : SizeListToStrHelper<SizeList<Value / 10, Values...>,
                          CharList<ParsedChars...>, '0' + (Value % 10),
                          Chars...> {};
template <size_t... Values, char... ParsedChars, char... Chars>
struct SizeListToStrHelper<SizeList<0, Values...>, CharList<ParsedChars...>,
                           Chars...>
    : SizeListToStrHelper<SizeList<Values...>,
                          CharList<ParsedChars..., Chars..., ','>> {};
template <char... ParsedChars, char... Chars>
struct SizeListToStrHelper<SizeList<0>, CharList<ParsedChars...>, Chars...>
    : CharsToStr<ParsedChars..., Chars...> {};

// Converts size_t values to a comma-separated string representation.
template <size_t... Sizes>
struct SizeListToStr : SizeListToStrHelper<SizeList<Sizes...>, CharList<>> {};
} // namespace detail

struct properties_tag {};

struct work_group_size_key {
  template <size_t... Dims>
  using value_t = property_value<work_group_size_key,
                                 std::integral_constant<size_t, Dims>...>;
};

struct work_group_size_hint_key {
  template <size_t... Dims>
  using value_t = property_value<work_group_size_hint_key,
                                 std::integral_constant<size_t, Dims>...>;
};

struct sub_group_size_key {
  template <uint32_t Size>
  using value_t = property_value<sub_group_size_key,
                                 std::integral_constant<uint32_t, Size>>;
};

template <size_t... Dims>
struct property_value<work_group_size_key,
                      std::integral_constant<size_t, Dims>...> {
  static_assert(sizeof...(Dims) > 0,
                "work_group_size property must have at least one value.");
  static_assert(detail::AllNonZero<Dims...>::value,
                "work_group_size property must only contain non-zero values.");

  constexpr size_t operator[](int Dim) {
    return std::array<size_t, sizeof...(Dims)>{Dims...}[Dim];
  }
};

template <size_t... Dims>
struct property_value<work_group_size_hint_key,
                      std::integral_constant<size_t, Dims>...> {
  static_assert(sizeof...(Dims) > 0,
                "work_group_size_hint property must have at least one value.");
  static_assert(
      detail::AllNonZero<Dims...>::value,
      "work_group_size_hint property must only contain non-zero values.");

  constexpr size_t operator[](int Dim) {
    return std::array<size_t, sizeof...(Dims)>{Dims...}[Dim];
  }
};

template <size_t... Dims>
inline constexpr work_group_size_key::value_t<Dims...> work_group_size;

template <size_t... Dims>
inline constexpr work_group_size_hint_key::value_t<Dims...>
    work_group_size_hint;

template <uint32_t Size>
inline constexpr sub_group_size_key::value_t<Size> sub_group_size;

template <> struct is_property_key<work_group_size_key> : std::true_type {};
template <>
struct is_property_key<work_group_size_hint_key> : std::true_type {};
template <> struct is_property_key<sub_group_size_key> : std::true_type {};

namespace detail {
template <> struct PropertyToKind<work_group_size_key> {
  static constexpr PropKind Kind = PropKind::WorkGroupSize;
};
template <> struct PropertyToKind<work_group_size_hint_key> {
  static constexpr PropKind Kind = PropKind::WorkGroupSizeHint;
};
template <> struct PropertyToKind<sub_group_size_key> {
  static constexpr PropKind Kind = PropKind::SubGroupSize;
};

template <>
struct IsCompileTimeProperty<work_group_size_key> : std::true_type {};
template <>
struct IsCompileTimeProperty<work_group_size_hint_key> : std::true_type {};
template <>
struct IsCompileTimeProperty<sub_group_size_key> : std::true_type {};

template <size_t... Dims>
struct PropertyMetaInfo<work_group_size_key::value_t<Dims...>> {
  static constexpr const char *name = "sycl-work-group-size";
  static constexpr const char *value = SizeListToStr<Dims...>::value;
};
template <size_t... Dims>
struct PropertyMetaInfo<work_group_size_hint_key::value_t<Dims...>> {
  static constexpr const char *name = "sycl-work-group-size-hint";
  static constexpr const char *value = SizeListToStr<Dims...>::value;
};
template <uint32_t Size>
struct PropertyMetaInfo<sub_group_size_key::value_t<Size>> {
  static constexpr const char *name = "sycl-sub-group-size";
  static constexpr uint32_t value = Size;
};

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
