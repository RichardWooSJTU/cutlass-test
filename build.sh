cudaroot=/root/paddlejob/workspace/env_run/wufeisheng/cuda-11.6
cutlassroot=/root/paddlejob/workspace/env_run/wufeisheng/cutlass

# ${cudaroot}/bin/nvcc -lineinfo -O3 -gencode arch=compute_80,code=sm_80 -I${cudaroot}/include -L${cudaroot}/lib64 \
${cudaroot}/bin/nvcc -gencode arch=compute_80,code=sm_80 -I${cudaroot}/include -L${cudaroot}/lib64 \
                -std=c++17 \
                -I ${cutlassroot}/include \
                -I ${cutlassroot}/tools/util/include -I`pwd`/ -I${cudaroot}/ \
                -I ${cutlassroot}/examples/common \
                main.cu -o gemm_dequant