//==------------------ device_global_map_entry.cpp -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/device_global_map_entry.hpp>
#include <detail/usm/usm_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

void *
DeviceGlobalMapEntry::getOrAllocateDeviceGlobalUSM(const device_impl *DevImpl,
                                                   context_impl *CtxImpl) {
  assert(!MIsDeviceImageScopeDecorated &&
         "USM allocations should not be acquired for device_global with "
         "device_image_scope property.");
  std::lock_guard<std::mutex> Lock(MDeviceToUSMPtrMapMutex);

  auto DGUSMPtr = MDeviceToUSMPtrMap.find({DevImpl, CtxImpl});
  if (DGUSMPtr != MDeviceToUSMPtrMap.end())
    return DGUSMPtr->second;

  void *NewDGUSMPtr = detail::usm::alignedAlloc(
      0, MDeviceGlobalTSize, CtxImpl, DevImpl, cl::sycl::usm::alloc::device);

  MDeviceToUSMPtrMap.insert({{DevImpl, CtxImpl}, NewDGUSMPtr});
  CtxImpl->addAssociatedDeviceGlobal(MDeviceGlobalPtr);
  return NewDGUSMPtr;
}

void DeviceGlobalMapEntry::removeAssociatedResources(
    const context_impl *CtxImpl) {
  std::lock_guard<std::mutex> Lock{MDeviceToUSMPtrMapMutex};
  for (device Device : CtxImpl->getDevices()) {
    auto USMPtrIt =
        MDeviceToUSMPtrMap.find({getSyclObjImpl(Device).get(), CtxImpl});
    if (USMPtrIt != MDeviceToUSMPtrMap.end()) {
      detail::usm::free(USMPtrIt->second, CtxImpl);
      MDeviceToUSMPtrMap.erase(USMPtrIt);
    }
  }
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
