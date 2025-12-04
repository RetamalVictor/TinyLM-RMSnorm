#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <type_traits>
#include <cmath>

// Conversions
__device__ inline float to_float(float x) { return x; }
__device__ inline float to_float(half x)  { return __half2float(x); }

template<typename T>
__device__ inline T from_float(float x);

template<>
__device__ inline float from_float<float>(float x) { return x; }

template<>
__device__ inline half from_float<half>(float x) { return __float2half(x); }

// Block reduction (sum) that returns the reduced value in lane 0 of warp 0
// (Other threads will have undefined 'val'; we'll broadcast via shared mem.)
template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
  __shared__ T shared[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;
  // warp reduce
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  // write per-warp sum
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  // first warp loads per-warp sums
  val = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0;
  if (wid == 0) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val; // valid in warp 0, lane 0
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
  float reduced = blockReduceSum<float>(sumsq);

  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = rsqrtf(reduced / hidden + eps);
    inv_rms_out[row] = s_inv_rms;
  }
  __syncthreads();
  float inv_rms = s_inv_rms; // broadcast

  for (int i = tid; i < hidden; i += stride) {
    float xi = to_float(x_row[i]);
    float wi = to_float(w[i]);
    float yi = (xi * inv_rms) * wi;
    y_row[i] = from_float<scalar_t>(yi);
  }
}

// Backward kernel: compute dx, accumulate dweight (in FP32)

template<typename scalar_t>
__global__ void rmsnorm_bwd_kernel(const scalar_t* __restrict__ dy,
                                   const scalar_t* __restrict__ x,
                                   const scalar_t* __restrict__ w,
                                   const float* __restrict__ inv_rms_in,
                                   scalar_t* __restrict__ dx,
                                   float* __restrict__ dw_fp32,
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
    atomicAdd(dw_fp32 + i, contrib);
  }
  float reduced_dot = blockReduceSum<float>(dot);

  __shared__ float s_dot;
  if (tid == 0) s_dot = reduced_dot;
  __syncthreads();
  float a = -r3_over_N * s_dot;

  // Second pass: dx = inv_rms * du + x * a
  for (int i = tid; i < hidden; i += stride) {
    float dyi = to_float(dy_row[i]);
    float wi  = to_float(w[i]);
    float xi  = to_float(x_row[i]);
    float du  = dyi * wi;
    float dxi = inv_rms * du + xi * a;
    dx_row[i] = from_float<scalar_t>(dxi);
  }
}

std::vector<torch::Tensor> rmsnorm_forward_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  int rows = x.numel() / x.size(-1);
  int hidden = x.size(-1);

  auto y = torch::empty_like(x);
  auto inv_rms = torch::empty({rows}, x.options().dtype(torch::kFloat));

  int threads = std::min(1024, 1 << (int)std::ceil(std::log2((double)hidden)));
  threads = max(32, threads); // at least one warp
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
  // Accumulate dweight in fp32 for stability (and to support half weights)
  auto dw32 = torch::zeros({weight.size(0)}, x.options().dtype(torch::kFloat));

  int threads = std::min(1024, 1 << (int)std::ceil(std::log2((double)hidden)));
  threads = max(32, threads);
  dim3 block(threads);
  dim3 grid(rows);

  auto stream = at::cuda::getCurrentCUDAStream();

  if (x.scalar_type() == torch::kFloat16) {
    rmsnorm_bwd_kernel<half><<<grid, block, 0, stream>>>(
      (half*)dy.data_ptr<at::Half>(), (half*)x.data_ptr<at::Half>(), (half*)weight.data_ptr<at::Half>(),
      inv_rms.data_ptr<float>(), (half*)dx.data_ptr<at::Half>(), dw32.data_ptr<float>(), hidden);
  } else if (x.scalar_type() == torch::kFloat32) {
    rmsnorm_bwd_kernel<float><<<grid, block, 0, stream>>>(
      dy.data_ptr<float>(), x.data_ptr<float>(), weight.data_ptr<float>(),
      inv_rms.data_ptr<float>(), dx.data_ptr<float>(), dw32.data_ptr<float>(), hidden);
  } else {
    TORCH_CHECK(false, "Unsupported dtype");
  }

  // Cast dweight to parameter dtype
  auto dw = (weight.scalar_type() == torch::kFloat16)
              ? dw32.to(torch::kHalf)
              : dw32;
  return {dx, dw};
}