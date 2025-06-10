// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PADDLE_WITH_HIP
// HIP not support cusolver

#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/backends/gpu/cuda/cuda_helper.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/impl/lu_kernel_impl.h"
#include "paddle/phi/kernels/lu_kernel.h"

namespace phi {

template <typename T>
void cusolver_bufferSize(const cusolverDnHandle_t& cusolverH,
                         int m,
                         int n,
                         T* d_A,
                         int lda,
                         int* lwork);
// template <typename T>
// void cusolver_getrf(const cusolverDnHandle_t& cusolverH,
//                     int m,
//                     int n,
//                     T* d_A,
//                     int lda,
//                     T* d_work,
//                     int* d_Ipiv,
//                     int* d_info);

template <>
void cusolver_bufferSize<float>(const cusolverDnHandle_t& cusolverH,
                                int m,
                                int n,
                                float* d_A,
                                int lda,
                                int* lwork) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnSgetrf_bufferSize(cusolverH, m, n, d_A, lda, lwork));
}

template <>
void cusolver_bufferSize<double>(const cusolverDnHandle_t& cusolverH,
                                 int m,
                                 int n,
                                 double* d_A,
                                 int lda,
                                 int* lwork) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnDgetrf_bufferSize(cusolverH, m, n, d_A, lda, lwork));
}

// template <>
// void cusolver_getrf<float>(const cusolverDnHandle_t& cusolverH,
//                            int m,
//                            int n,
//                            float* d_A,
//                            int lda,
//                            float* d_work,
//                            int* d_Ipiv,
//                            int* d_info) {
//   PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSgetrf(
//       cusolverH, m, n, d_A, lda, d_work, d_Ipiv, d_info));
// }

// template <>
// void cusolver_getrf<double>(const cusolverDnHandle_t& cusolverH,
//                             int m,
//                             int n,
//                             double* d_A,
//                             int lda,
//                             double* d_work,
//                             int* d_Ipiv,
//                             int* d_info) {
//   PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDgetrf(
//       cusolverH, m, n, d_A, lda, d_work, d_Ipiv, d_info));
// }

void cusolver_getrf(const cusolverDnHandle_t& cusolverH,
                    const cusolverDnParams_t& params,
                    int64_t m,
                    int64_t n,
                    const cudaDataType dataTypeA,
                    void* d_A,
                    int64_t lda,
                    int64_t* d_Ipiv,
                    cudaDataType computeType,
                    void* d_work,
                    size_t lwork_d,
                    void* h_work,
                    size_t lwork_h,
                    int* d_info) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnXgetrf(cusolverH,
                                                       params,
                                                       m,
                                                       n,
                                                       dataTypeA,
                                                       d_A,
                                                       lda,
                                                       d_Ipiv,
                                                       computeType,
                                                       d_work,
                                                       lwork_d,
                                                       h_work,
                                                       lwork_h,
                                                       d_info));
}

void test_cuda(const std::string& str) {
  std::cout << str << " begin" << std::endl;
  // 1. wait all kernel finish
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());

  // 2. get error state
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());

  // 3. check if cuda 700
  size_t bytes = 256;
  char* cuda_mem;
  char* cpu_mem = new char[bytes + 1];

  cudaMalloc(&cuda_mem, bytes + 1);
  cudaMemset(cuda_mem, 0, bytes + 1);
  cudaMemcpyAsync(cpu_mem, cuda_mem, bytes, cudaMemcpyDeviceToHost);

  cudaFree(cuda_mem);
  delete[] cpu_mem;
  std::cout << str << " end" << std::endl;
}

template <typename T, typename Context>
void lu_decomposed_kernel(const Context& dev_ctx,
                          int64_t m,
                          int64_t n,
                          T* d_A,
                          int64_t lda,
                          int64_t* d_Ipiv,
                          int* d_info,
                          const int algo = 0) {
  /* step 1: get cusolver handle and create advanced parameters*/
  auto cusolverH = dev_ctx.cusolver_dn_handle();
  std::cout << "m" << m << "n" << n << "lda" << lda << std::endl;
  test_cuda("lu 111");
  cusolverDnParams_t params;
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnCreateParams(&params));
  test_cuda("lu 222");
  if (algo == 0) {
    test_cuda("lu 333");
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSetAdvOptions(
        params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0));
    test_cuda("lu 444");
  } else {
    test_cuda("lu 555");
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSetAdvOptions(
        params, CUSOLVERDN_GETRF, CUSOLVER_ALG_1));
    test_cuda("lu 666");
  }

  /* step 2: query working space of getrf */
  size_t lwork_d, lwork_h;
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnXgetrf_bufferSize(
      cusolverH,
      params,
      m,
      n,
      phi::backends::gpu::ToCudaDataType<T>(),
      d_A,
      lda,
      phi::backends::gpu::ToCudaDataType<T>(),
      &lwork_d,
      &lwork_h));
  test_cuda("lu 777");
  auto d_work_buff = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      lwork_d * sizeof(T),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  test_cuda("lu 888");
  auto h_work_buff = phi::memory_utils::Alloc(
      phi::CPUPlace(),
      lwork_h * sizeof(T),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  test_cuda("lu 999");
  /* step 3: LU factorization */
  if (d_Ipiv) {
    test_cuda("lu 1111");
    cusolver_getrf(cusolverH,
                   params,
                   m,
                   n,
                   phi::backends::gpu::ToCudaDataType<T>(),
                   d_A,
                   lda,
                   d_Ipiv,
                   phi::backends::gpu::ToCudaDataType<T>(),
                   d_work_buff->ptr(),
                   lwork_d,
                   h_work_buff->ptr(),
                   lwork_h,
                   d_info);
    test_cuda("lu 2222");
  } else {
    test_cuda("lu 3333");
    cusolver_getrf(cusolverH,
                   params,
                   m,
                   n,
                   phi::backends::gpu::ToCudaDataType<T>(),
                   d_A,
                   lda,
                   NULL,
                   phi::backends::gpu::ToCudaDataType<T>(),
                   d_work_buff->ptr(),
                   lwork_d,
                   h_work_buff->ptr(),
                   lwork_h,
                   d_info);
    test_cuda("lu 4444");
  }
  test_cuda("lu 777");
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
  test_cuda("lu 5555");
}

template <typename T, typename Context>
void LUKernel(const Context& dev_ctx,
              const DenseTensor& x,
              bool pivot,
              DenseTensor* out,
              DenseTensor* pivots,
              DenseTensor* infos) {
  const int64_t kMaxBlockDim = 512;

  *out = Transpose2DTo6D<Context, T>(dev_ctx, x);

  auto outdims = out->dims();
  auto outrank = outdims.size();

  int m = static_cast<int>(outdims[outrank - 1]);
  int n = static_cast<int>(outdims[outrank - 2]);
  int lda = std::max(1, m);
  if (pivot) {
    auto ipiv_dims = common::slice_ddim(outdims, 0, outrank - 1);
    ipiv_dims[outrank - 2] = std::min(m, n);
    pivots->Resize(ipiv_dims);
  }
  dev_ctx.template Alloc<int64_t>(pivots);
  auto ipiv_data = pivots->data<int64_t>();

  auto info_dims = common::slice_ddim(outdims, 0, outrank - 2);
  infos->Resize(info_dims);
  dev_ctx.template Alloc<int>(infos);
  auto info_data = infos->data<int>();

  auto batchsize = product(info_dims);
  batchsize = std::max(static_cast<int>(batchsize), 1);
  dev_ctx.template Alloc<T>(out);
  auto out_data = out->data<T>();
  for (int b = 0; b < batchsize; b++) {
    auto out_data_item = &out_data[b * m * n];
    int* info_data_item = &info_data[b];
    if (pivot) {
      auto ipiv_data_item = &ipiv_data[b * std::min(m, n)];
      lu_decomposed_kernel(
          dev_ctx, m, n, out_data_item, lda, ipiv_data_item, info_data_item);
    } else {
      lu_decomposed_kernel(
          dev_ctx, m, n, out_data_item, lda, NULL, info_data_item);
    }
  }
  *out = Transpose2DTo6D<Context, T>(dev_ctx, *out);
}

}  // namespace phi

PD_REGISTER_KERNEL(lu,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::LUKernel,
                   float,
                   double) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}

#endif  // not PADDLE_WITH_HIP
