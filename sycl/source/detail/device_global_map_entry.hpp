//==----------------- device_global_map_entry.hpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/pi.hpp>

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// Forward declaration
class context_impl;
class device_impl;
class platform_impl;

struct DeviceGlobalUSMMem {
  DeviceGlobalUSMMem(void *Ptr) : MPtr(Ptr) {}
  ~DeviceGlobalUSMMem();

  void *getPtr() const noexcept { return MPtr; }
  std::optional<RT::PiEvent> getZeroInitEvent(const plugin &Plugin);

private:
  void *MPtr;
  std::mutex MZeroInitEventMutex;
  std::optional<RT::PiEvent> MZeroInitEvent;

  friend struct DeviceGlobalMapEntry;
};

struct DeviceGlobalMapEntry {
  // The unique identifier of the device_global.
  std::string MUniqueId;
  // Pointer to the device_global on host.
  const void *MDeviceGlobalPtr;
  // The kernel set identifier for the kernel set using the device_global.
  KernelSetId MKernelSetId;
  // Size of the underlying type in the device_global.
  std::uint32_t MDeviceGlobalTSize;
  // True if the device_global has been decorated with device_image_scope.
  bool MIsDeviceImageScopeDecorated;

  // Constructor for only initializing ID and pointer. The other members will
  // be initialized later.
  DeviceGlobalMapEntry(std::string UniqueId, const void *DeviceGlobalPtr)
      : MUniqueId(UniqueId), MDeviceGlobalPtr(DeviceGlobalPtr), MKernelSetId(0),
        MDeviceGlobalTSize(0), MIsDeviceImageScopeDecorated(false) {}

  // Constructor for only initializing ID, type size, and device image scope
  // flag. The pointer to the device global will be initialized later.
  DeviceGlobalMapEntry(std::string UniqueId, KernelSetId KSId,
                       std::uint32_t DeviceGlobalTSize,
                       bool IsDeviceImageScopeDecorated)
      : MUniqueId(UniqueId), MDeviceGlobalPtr(nullptr), MKernelSetId(KSId),
        MDeviceGlobalTSize(DeviceGlobalTSize),
        MIsDeviceImageScopeDecorated(IsDeviceImageScopeDecorated) {}

  // Initialize the pointer to the associated device_global.
  void initialize(const void *DeviceGlobalPtr) {
    assert(DeviceGlobalPtr && "Device global pointer cannot be null");
    assert(!MDeviceGlobalPtr &&
           "Device global pointer has already been initialized.");
    MDeviceGlobalPtr = DeviceGlobalPtr;
  }

  // Initialize the device_global's element type size and the flag signalling
  // if the device_global has the device_image_scope property.
  void initialize(KernelSetId KSId, std::uint32_t DeviceGlobalTSize,
                  bool IsDeviceImageScopeDecorated) {
    assert(DeviceGlobalTSize != 0 && "Device global initialized with 0 size.");
    assert(MDeviceGlobalTSize == 0 &&
           "Device global has already been initialized.");
    MKernelSetId = KSId;
    MDeviceGlobalTSize = DeviceGlobalTSize;
    MIsDeviceImageScopeDecorated = IsDeviceImageScopeDecorated;
  }

  // Gets or allocates USM memory for a device_global.
  DeviceGlobalUSMMem &
  getOrAllocateDeviceGlobalUSM(const std::shared_ptr<queue_impl> &QueueImpl,
                               bool ZeroInit = false);

  // Removes resources for device_globals associated with the context.
  void removeAssociatedResources(const context_impl *CtxImpl);

private:
  // Map from a device and a context to the associated USM allocation for the
  // device_global. This should always be empty if MIsDeviceImageScopeDecorated
  // is true.
  std::map<std::pair<const device_impl *, const context_impl *>,
           DeviceGlobalUSMMem>
      MDeviceToUSMPtrMap;
  std::mutex MDeviceToUSMPtrMapMutex;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
