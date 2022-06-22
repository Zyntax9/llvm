//==---------- pi_primary_context.cpp - PI unit tests ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <cuda.h>

#include "TestGetPlatforms.hpp"
#include <pi_cuda.hpp>

#define SYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL 1
#include <sycl/ext/oneapi/experimental/backend/cuda.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;

struct CudaPrimaryContextTests : public ::testing::TestWithParam<platform> {

protected:
  device deviceA_;
  device deviceB_;

  void SetUp() override {
    std::vector<device> CudaDevices = GetParam().get_devices();

    deviceA_ = CudaDevices[0];
    deviceB_ = CudaDevices.size() > 1 ? CudaDevices[1] : deviceA_;
  }

  void TearDown() override {}
};

TEST_P(CudaPrimaryContextTests, piSingleContext) {
  std::cout << "create single context" << std::endl;
  context Context(
      deviceA_, async_handler{},
      {sycl::ext::oneapi::cuda::property::context::use_primary_context{}});

  CUdevice CudaDevice = get_native<backend::ext_oneapi_cuda>(deviceA_);
  std::vector<CUcontext> CudaContexts =
      get_native<backend::ext_oneapi_cuda>(Context);

  CUcontext PrimaryCudaContext;
  cuDevicePrimaryCtxRetain(&PrimaryCudaContext, CudaDevice);

  ASSERT_EQ(CudaContexts.size(), 1);
  ASSERT_EQ(CudaContexts[0], PrimaryCudaContext);

  cuDevicePrimaryCtxRelease(CudaDevice);
}

TEST_P(CudaPrimaryContextTests, piMultiContextSingleDevice) {
  std::cout << "create multiple contexts for one device" << std::endl;
  context ContextA(
      deviceA_, async_handler{},
      {sycl::ext::oneapi::cuda::property::context::use_primary_context{}});
  context ContextB(
      deviceA_, async_handler{},
      {sycl::ext::oneapi::cuda::property::context::use_primary_context{}});

  std::vector<CUcontext> CudaContextsA =
      get_native<backend::ext_oneapi_cuda>(ContextA);
  std::vector<CUcontext> CudaContextsB =
      get_native<backend::ext_oneapi_cuda>(ContextB);

  ASSERT_EQ(CudaContextsA.size(), 1);
  ASSERT_EQ(CudaContextsB.size(), 1);
  ASSERT_EQ(CudaContextsA[0], CudaContextsB[0]);
}

TEST_P(CudaPrimaryContextTests, piMultiContextMultiDevice) {
  if (deviceA_ == deviceB_)
    return;

  CUdevice CudaDeviceA = get_native<backend::ext_oneapi_cuda>(deviceA_);
  CUdevice CudaDeviceB = get_native<backend::ext_oneapi_cuda>(deviceB_);

  ASSERT_NE(CudaDeviceA, CudaDeviceB);

  std::cout << "create multiple contexts for multiple devices" << std::endl;
  context ContextA(
      deviceA_, async_handler{},
      {sycl::ext::oneapi::cuda::property::context::use_primary_context{}});
  context ContextB(
      deviceB_, async_handler{},
      {sycl::ext::oneapi::cuda::property::context::use_primary_context{}});

  std::vector<CUcontext> CudaContextsA =
      get_native<backend::ext_oneapi_cuda>(ContextA);
  std::vector<CUcontext> CudaContextsB =
      get_native<backend::ext_oneapi_cuda>(ContextB);

  ASSERT_EQ(CudaContextsA.size(), 1);
  ASSERT_EQ(CudaContextsB.size(), 1);
  ASSERT_EQ(CudaContextsA[0], CudaContextsB[0]);
}

INSTANTIATE_TEST_SUITE_P(
    OnCudaPlatform, CudaPrimaryContextTests,
    ::testing::ValuesIn(pi::getPlatformsWithName("CUDA BACKEND")));
