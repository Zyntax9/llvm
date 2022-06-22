//==------- test_interop_get_native.cpp - SYCL CUDA get_native tests -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <cuda.h>

#include "TestGetPlatforms.hpp"

#define SYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL 1
#include <sycl/ext/oneapi/experimental/backend/cuda.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;

struct CudaInteropGetNativeTests : public ::testing::TestWithParam<platform> {

protected:
  std::unique_ptr<queue> syclQueue_;
  device syclDevice_;

  void SetUp() override {
    syclDevice_ = GetParam().get_devices()[0];
    syclQueue_ = std::unique_ptr<queue>{new queue{syclDevice_}};
  }

  void TearDown() override { syclQueue_.reset(); }
};

TEST_P(CudaInteropGetNativeTests, getNativeDevice) {
  CUdevice cudaDevice = get_native<backend::ext_oneapi_cuda>(syclDevice_);
  char cudaDeviceName[2] = {0, 0};
  CUresult result = cuDeviceGetName(cudaDeviceName, 2, cudaDevice);
  ASSERT_EQ(result, CUDA_SUCCESS);
  ASSERT_NE(cudaDeviceName[0], 0);
}

TEST_P(CudaInteropGetNativeTests, getNativeContext) {
  std::vector<CUcontext> cudaContexts =
      get_native<backend::ext_oneapi_cuda>(syclQueue_->get_context());
  ASSERT_FALSE(cudaContexts.empty());
}

TEST_P(CudaInteropGetNativeTests, getNativeQueue) {
  CUstream cudaStream = get_native<backend::ext_oneapi_cuda>(*syclQueue_);
  ASSERT_NE(cudaStream, nullptr);

  CUcontext streamContext = nullptr;
  CUresult result = cuStreamGetCtx(cudaStream, &streamContext);
  ASSERT_EQ(result, CUDA_SUCCESS);

  std::vector<CUcontext> cudaContexts =
      get_native<backend::ext_oneapi_cuda>(syclQueue_->get_context());
  ASSERT_EQ(streamContext, cudaContexts[0]);
}

TEST_P(CudaInteropGetNativeTests, interopTaskGetMem) {
  buffer<int, 1> syclBuffer(range<1>{1});
  syclQueue_->submit([&](handler &cgh) {
    auto syclAccessor = syclBuffer.get_access<access::mode::read>(cgh);
    cgh.host_task([=](interop_handle ih) {
      int *cudaPtr =
          ih.get_native_mem<backend::ext_oneapi_cuda>(syclAccessor);
      CUdeviceptr cudaPtrBase;
      size_t cudaPtrSize = 0;
      std::vector<CUcontext> cudaContexts =
          get_native<backend::ext_oneapi_cuda>(syclQueue_->get_context());
      ASSERT_EQ(CUDA_SUCCESS, cuCtxPushCurrent(cudaContexts[0]));
      ASSERT_EQ(CUDA_SUCCESS,
                cuMemGetAddressRange(&cudaPtrBase, &cudaPtrSize,
                                     reinterpret_cast<CUdeviceptr>(cudaPtr)));
      ASSERT_EQ(CUDA_SUCCESS, cuCtxPopCurrent(nullptr));
      ASSERT_EQ(sizeof(int), cudaPtrSize);
    });
  });
}

TEST_P(CudaInteropGetNativeTests, hostTaskGetNativeMem) {
  buffer<int, 1> syclBuffer(range<1>{1});
  syclQueue_->submit([&](handler &cgh) {
    auto syclAccessor = syclBuffer.get_access<access::mode::read>(cgh);
    cgh.host_task([=](interop_handle ih) {
      int *cudaPtr =
          ih.get_native_mem<backend::ext_oneapi_cuda>(syclAccessor);
      CUdeviceptr cudaPtrBase;
      size_t cudaPtrSize = 0;
      std::vector<CUcontext> cudaContexts =
          get_native<backend::ext_oneapi_cuda>(syclQueue_->get_context());
      ASSERT_EQ(CUDA_SUCCESS, cuCtxPushCurrent(cudaContexts[0]));
      ASSERT_EQ(CUDA_SUCCESS,
                cuMemGetAddressRange(&cudaPtrBase, &cudaPtrSize,
                                     reinterpret_cast<CUdeviceptr>(cudaPtr)));
      ASSERT_EQ(CUDA_SUCCESS, cuCtxPopCurrent(nullptr));
      ASSERT_EQ(sizeof(int), cudaPtrSize);
    });
  });
}

TEST_P(CudaInteropGetNativeTests, hostTaskGetNativeContext) {
  std::vector<CUcontext> cudaContexts =
      get_native<backend::ext_oneapi_cuda>(syclQueue_->get_context());
  syclQueue_->submit([&](handler &cgh) {
    cgh.host_task([=](interop_handle ih) {
      std::vector<CUcontext> cudaInteropContexts =
          ih.get_native_context<backend::ext_oneapi_cuda>();
      ASSERT_EQ(cudaInteropContexts[0], cudaContexts[0]);
    });
  });
}

INSTANTIATE_TEST_SUITE_P(
    OnCudaPlatform, CudaInteropGetNativeTests,
    ::testing::ValuesIn(pi::getPlatformsWithName("CUDA BACKEND")));
