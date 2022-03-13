//==------------------ usm_impl.hpp - USM API Utils -------------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/usm.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
namespace usm {

void *alignedAlloc(size_t Alignment, size_t Size, const context_impl *CtxImpl,
                   const device_impl *DevImpl, cl::sycl::usm::alloc Kind,
                   const property_list &PropList = {});

void free(void *Ptr, const context_impl *CtxImpl);

} // namespace usm
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
