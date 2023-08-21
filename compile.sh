# cudaroot=/home/workplace/cuda-11.6/
# ${cudaroot}/bin/nvcc -gencode arch=compute_80,code=sm_80 -I${cudaroot}/include -L${cudaroot}/lib64 \
#                  -I`pwd`/cutlass/include -I`pwd`/ -I${cudaroot}/ \
# fpAintB_test.cu -o fpAintB_test

cudaroot=/root/paddlejob/workspace/env_run/wufeisheng/cuda-11.6
cutlassroot=/root/paddlejob/workspace/env_run/wufeisheng/cutlass

${cudaroot}/bin/nvcc -lineinfo -O3 -gencode arch=compute_80,code=sm_80 -I${cudaroot}/include -L${cudaroot}/lib64 \
                -shared \
                -std=c++17 \
                -I ${cutlassroot}/include \
                -I ${cutlassroot}/tools/util/include -I`pwd`/ -I${cudaroot}/ \
                -I ${cutlassroot}/examples/common \
                -Xcompiler \
                -fPIC $(python3 -m pybind11 --includes) \
                gemm_dequant.cu -o cutlass$(python3-config --extension-suffix)
