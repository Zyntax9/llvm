// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s
// expected-no-diagnostics

#include <CL/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

static device_global<int> DeviceGlobal1;
static device_global<int, decltype(properties(device_image_scope))>
    DeviceGlobal2;
static device_global<int, decltype(properties(host_access_none))> DeviceGlobal3;
static device_global<int, decltype(properties(init_mode_reset))> DeviceGlobal4;
static device_global<int, decltype(properties(implement_in_csr_on))>
    DeviceGlobal5;
static device_global<int, decltype(properties(
                              implement_in_csr_off, host_access_write,
                              device_image_scope, init_mode_reprogram))>
    DeviceGlobal6;

int main() {
  static_assert(!DeviceGlobal1.has_property<device_image_scope_key>());
  static_assert(!DeviceGlobal1.has_property<host_access_key>());
  static_assert(!DeviceGlobal1.has_property<init_mode_key>());
  static_assert(!DeviceGlobal1.has_property<implement_in_csr_key>());

  static_assert(DeviceGlobal2.has_property<device_image_scope_key>());
  static_assert(!DeviceGlobal2.has_property<host_access_key>());
  static_assert(!DeviceGlobal2.has_property<init_mode_key>());
  static_assert(!DeviceGlobal2.has_property<implement_in_csr_key>());
  static_assert(DeviceGlobal2.get_property<device_image_scope_key>() ==
                device_image_scope);

  static_assert(!DeviceGlobal3.has_property<device_image_scope_key>());
  static_assert(DeviceGlobal3.has_property<host_access_key>());
  static_assert(!DeviceGlobal3.has_property<init_mode_key>());
  static_assert(!DeviceGlobal3.has_property<implement_in_csr_key>());
  static_assert(DeviceGlobal3.get_property<host_access_key>().value ==
                host_access_enum::none);

  static_assert(!DeviceGlobal4.has_property<device_image_scope_key>());
  static_assert(!DeviceGlobal4.has_property<host_access_key>());
  static_assert(DeviceGlobal4.has_property<init_mode_key>());
  static_assert(!DeviceGlobal4.has_property<implement_in_csr_key>());
  static_assert(DeviceGlobal4.get_property<init_mode_key>().value ==
                init_mode_enum::reset);

  static_assert(!DeviceGlobal5.has_property<device_image_scope_key>());
  static_assert(!DeviceGlobal5.has_property<host_access_key>());
  static_assert(!DeviceGlobal5.has_property<init_mode_key>());
  static_assert(DeviceGlobal5.has_property<implement_in_csr_key>());
  static_assert(DeviceGlobal5.get_property<implement_in_csr_key>().value);

  static_assert(DeviceGlobal6.has_property<device_image_scope_key>());
  static_assert(DeviceGlobal6.has_property<host_access_key>());
  static_assert(DeviceGlobal6.has_property<init_mode_key>());
  static_assert(DeviceGlobal6.has_property<implement_in_csr_key>());
  static_assert(DeviceGlobal6.get_property<device_image_scope_key>() ==
                device_image_scope);
  static_assert(DeviceGlobal6.get_property<host_access_key>().value ==
                host_access_enum::write);
  static_assert(DeviceGlobal6.get_property<init_mode_key>().value ==
                init_mode_enum::reprogram);
  static_assert(!DeviceGlobal6.get_property<implement_in_csr_key>().value);

  // TODO: This is currently needed as otherwise the header will not declare
  // the registration.
  sycl::queue q;
  q.single_task([]() {});
  return 0;
}
