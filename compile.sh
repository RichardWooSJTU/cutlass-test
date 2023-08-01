# cudaroot=/home/workplace/cuda-11.6/
# ${cudaroot}/bin/nvcc -gencode arch=compute_80,code=sm_80 -I${cudaroot}/include -L${cudaroot}/lib64 \
#                  -I`pwd`/cutlass/include -I`pwd`/ -I${cudaroot}/ \
# fpAintB_test.cu -o fpAintB_test

cudaroot=/usr/local/cuda-11.6
${cudaroot}/bin/nvcc -gencode arch=compute_80,code=sm_80 -I${cudaroot}/include -L${cudaroot}/lib64 \
                  -std=c++17 \
                 -I /root/paddlejob/workspace/env_run/wufeisheng/cutlass/include \
                -I /root/paddlejob/workspace/env_run/wufeisheng/cutlass/tools/util/include -I`pwd`/ -I${cudaroot}/ \
                -I /root/paddlejob/workspace/env_run/wufeisheng/cutlass/examples/common \
gemm_dequant.cu -o gemm_dequant