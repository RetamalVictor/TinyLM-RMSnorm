from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rmsnorm_cuda',
    ext_modules=[
        CUDAExtension(
            name='rmsnorm_cuda',
            sources=['kernels/rmsnorm_binding.cpp', 'kernels/rmsnorm_cuda.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math']},
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)