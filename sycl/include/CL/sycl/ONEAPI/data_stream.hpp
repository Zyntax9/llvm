//==-------------- data_stream.hpp - SYCL data streams ----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/buffer.hpp>
#include <CL/sycl/handler.hpp>

#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ONEAPI {
namespace detail {

/// Base non-template class which is a base class for all data streams
/// classes. It is needed to detect the data streams.
class data_stream_base {};

/// Predicate returning true if all template type parameter is a data stream.
template <typename T> struct IsDataStream {
  static constexpr bool value =
    std::is_base_of<data_stream_base, T>::value;
};

} // namespace detail


template <access::mode AccessMode, typename DataT, int Dimensions>
class data_stream : private detail::data_stream_base {
public:
  using value_type = DataT;
  using reference
    = std::conditional<AccessMode == access::mode::read, const DataT &,
                       const DataT &>;
  static constexpr access::mode accessor_mode = AccessMode;

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT>
  data_stream(buffer<T, Dims, AllocatorT> &bufferRef,
              handler &commandGroupHandlerRef,
              const property_list &propList = {})
    : access{bufferRef, commandGroupHandlerRef} {}

  /// Checks if this context has a property of type propertyT.
  ///
  /// \return true if this context has a property of type propertyT.
  template <typename propertyT> bool has_property() const;

  /// Gets the specified property of this context.
  ///
  /// Throws invalid_object_error if this context does not have a property
  /// of type propertyT.
  ///
  /// \return a copy of the property of type propertyT.
  template <typename propertyT> propertyT get_property() const;

//private:
  using accessor_type
    = accessor<DataT, Dimensions, AccessMode, target::global_buffer>;
  accessor_type access;

  friend class handler;
};

} // namespace ONEAPI
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
