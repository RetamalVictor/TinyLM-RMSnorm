#include <torch/extension.h>
#include <vector>

// forward declarations
std::vector<torch::Tensor> rmsnorm_forward_cuda(torch::Tensor x, torch::Tensor weight, double eps);
std::vector<torch::Tensor> rmsnorm_backward_cuda(torch::Tensor dy, torch::Tensor x, torch::Tensor weight, torch::Tensor inv_rms, double eps);

// Python bindings
std::vector<torch::Tensor> rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  return rmsnorm_forward_cuda(x, weight, eps);
}

std::vector<torch::Tensor> rmsnorm_backward(torch::Tensor dy, torch::Tensor x, torch::Tensor weight, torch::Tensor inv_rms, double eps) {
  TORCH_CHECK(dy.is_cuda() && x.is_cuda() && weight.is_cuda() && inv_rms.is_cuda(), "all must be CUDA");
  return rmsnorm_backward_cuda(dy, x, weight, inv_rms, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
  m.def("backward", &rmsnorm_backward, "RMSNorm backward (CUDA)");
}