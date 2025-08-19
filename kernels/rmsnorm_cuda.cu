#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

// Utility: type traits

template<typename T>
struct Vec { using scalar = T; };

template<>
struct Vec<half> { using scalar = half; };

// Load/store helpers
__device__ inline float to_float(float x) { return x; }
__device__ inline float to_float(half x) { return __half2float(x); }
__device__ inline half  to_half(float x) { return __float2half(x); }

// Block reduction (sum)
template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
  __shared__ T shared[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;

  // warp reduce
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
  if (wid == 0) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Forward kernel: per-row RMS + scale + weight

template<typename scalar_t>
__global__ void rmsnorm_fwd_kernel(const scalar_t* __restrict__ x,
                                   const scalar_t* __restrict__ w,
                                   scalar_t* __restrict__ y,
                                   float* __restrict__ inv_rms_out,
                                   int hidden, float eps) {
  int row = blockIdx.x; // one block per row
  int tid = threadIdx.x;
  int stride = blockDim.x;

  const scalar_t* x_row = x + (size_t)row * hidden;
  scalar_t* y_row = y + (size_t)row * hidden;

  float sumsq = 0.f;
  for (int i = tid; i < hidden; i += stride) {
    float xi = to_float(x_row[i]);
    sumsq += xi * xi;
  }
  sumsq = blockReduceSum<float>(sumsq);

  float inv_rms = rsqrtf(sumsq / hidden + eps);
  if (tid == 0) inv_rms_out[row] = inv_rms;

  for (int i = tid; i < hidden; i += stride) {
    float xi = to_float(x_row[i]);
    float wi = to_float(w[i]);
    float yi = (xi * inv_rms) * wi;
    y_row[i] = (sizeof(scalar_t) == sizeof(half)) ? to_half(yi) : (scalar_t)yi;
  }
}

// Backward kernel: compute dx, accumulate dweight

template<typename scalar_t>
__global__ void rmsnorm_bwd_kernel(const scalar_t* __restrict__ dy,
                                   const scalar_t* __restrict__ x,
                                   const scalar_t* __restrict__ w,
                                   const float* __restrict__ inv_rms_in,
                                   scalar_t* __restrict__ dx,
                                   scalar_t* __restrict__ dw,
                                   int hidden) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int stride = blockDim.x;

  const scalar_t* dy_row = dy + (size_t)row * hidden;
  const scalar_t* x_row  = x  + (size_t)row * hidden;
  scalar_t* dx_row       = dx + (size_t)row * hidden;

  float inv_rms = inv_rms_in[row];
  float r3_over_N = (inv_rms * inv_rms * inv_rms) / hidden;

  float dot = 0.f; // sum_j x_j * du_j
  // First pass: compute dot and also partial dw
  for (int i = tid; i < hidden; i += stride) {
    float dyi = to_float(dy_row[i]);
    float wi  = to_float(w[i]);
    float xi  = to_float(x_row[i]);
    float du  = dyi * wi;
    dot += xi * du;
    // dweight += dy * (x * inv_rms)
    float contrib = dyi * (xi * inv_rms);
    atomicAdd((float*)dw + i, contrib);
  }
  dot = blockReduceSum<float>(dot);
  float a = -r3_over_N * dot;

  // Second pass: dx = inv_rms * du + x * a
  for (int i = tid; i < hidden; i += stride) {
    float dyi = to_float(dy_row[i]);
    float wi  = to_float(w[i]);
    float xi  = to_float(x_row[i]);
    float du  = dyi * wi;
    float dxi = inv_rms * du + xi * a;
    dx_row[i] = (sizeof(scalar_t) == sizeof(half)) ? to_half(dxi) : (scalar_t)dxi;
  }
}

std::vector<torch::Tensor> rmsnorm_forward_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  auto B = x.size(0) * x.size(1) / x.size(-1);
  int rows = x.numel() / x.size(-1);
  int hidden = x.size(-1);

  auto y = torch::empty_like(x);
  auto inv_rms = torch::empty({rows}, x.options().dtype(torch::kFloat));

  int threads = std::min(1024, 1 << (int)std::ceil(std::log2((double)hidden)));
  dim3 block(threads);
  dim3 grid(rows);

  auto stream = at::cuda::getCurrentCUDAStream();

  if (x.scalar_type() == torch::kFloat16) {
    rmsnorm_fwd_kernel<half><<<grid, block, 0, stream>>>(
      (half*)x.data_ptr<at::Half>(), (half*)weight.data_ptr<at::Half>(),
      (half*)y.data_ptr<at::Half>(), inv_rms.data_ptr<float>(), hidden, (float)eps);
  } else if (x.scalar_type() == torch::kFloat32) {
    rmsnorm_fwd_kernel<float><<<grid, block, 0, stream>>>(
      x.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(),
      inv_rms.data_ptr<float>(), hidden, (float)eps);
  } else {
    TORCH_CHECK(false, "Unsupported dtype");
  }
  return {y, inv_rms};
}

std::vector<torch::Tensor> rmsnorm_backward_cuda(torch::Tensor dy, torch::Tensor x, torch::Tensor weight, torch::Tensor inv_rms, double eps) {
  int rows = x.numel() / x.size(-1);
  int hidden = x.size(-1);

  auto dx = torch::empty_like(x);
  auto dw = torch::zeros_like(weight);

  int threads = std::min(1024, 1 << (int)std::ceil(std::log2((double)hidden)));
  dim3 block(threads);
  dim3 grid(rows);

  auto stream = at::cuda::getCurrentCUDAStream();

  if (x.scalar_type() == torch::kFloat16) {
    rmsnorm_bwd_kernel<half><<<grid, block, 0, stream>>>(
      (half*)dy.data_ptr<at::Half>(), (half*)x.data_ptr<at::Half>(), (half*)weight.data_ptr<at::Half>(),
      inv_rms.data_ptr<float>(), (half*)dx.data_ptr<at::Half>(), (half*)dw.data_ptr<at::Half>(), hidden);
  } else if (x.scalar_type() == torch::kFloat32) {
    rmsnorm_bwd_kernel<float><<<grid, block, 0, stream>>>(
      dy.data_ptr<float>(), x.data_ptr<float>(), weight.data_ptr<float>(),
      inv_rms.data_ptr<float>(), dx.data_ptr<float>(), dw.data_ptr<float>(), hidden);
  } else {
    TORCH_CHECK(false, "Unsupported dtype");
  }
  return {dx, dw};
}