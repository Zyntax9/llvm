//==------ tracked_memory_manager.hpp --- SYCL program manager -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>

#include <string>
#include <mutex>
#include <unordered_map>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
class context;
namespace detail {

class TrackedMemoryManager {
  struct TrackedMemoryEntry {
    TrackedMemoryEntry() {
#ifdef _WIN32
#else
#endif
    }

  private:
    std::string MTrace;
  };

public:
  void registerTrackedMemory(uintptr_t Ptr) {
    TrackedMemoryEntry Entry;
    MTrackedMemoryEntries.insert({Ptr, Entry});
  }

private:
  std::unordered_map<uintptr_t, TrackedMemoryEntry> MTrackedMemoryEntries;
  std::mutex MTrackedMemoryEntriesMutex;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
