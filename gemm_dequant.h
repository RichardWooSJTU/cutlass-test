#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "gemm_with_dequant.h"
#include "gemm_ref.h"


void RunGemmDequant(const int8_t* a, const int8_t* b, const float* dequant_scale, float* c, int m, int k, int n) {
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = float;
  using ElementCompute = int32_t;
  using ElementD = ElementC;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;

  // using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
  // using WarpShape        = cutlass::gemm::GemmShape<64, 64, 64>;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape        = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;


  static int const kStages = 5;

  /// Linear scaling operator
  using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementCompute,
    ElementCompute
  >;

  using GemmDequant = cutlass::GemmDequant<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueFunctorOp,
    kStages
  >;

  using LayoutC = typename GemmDequant::LayoutC;

  // Initialize data

  // int m = 40, k = 1024, n = 8192;

  int8_t* A_dev, *B_dev;
  float* C_dev = nullptr;
  float* D_dev;
  float* dequant_scale_dev;


  
  // Assume origin matrix is col-major

  cudaMalloc(&A_dev, k * m); // k * m in row-major
  cudaMalloc(&B_dev, n * k); // k * n in col-major
  cudaMalloc(&D_dev, n * m * sizeof(float)); // m * n in row-major
  cudaMalloc(&dequant_scale_dev, n * sizeof(float)); 

  cudaMemcpy(A_dev, a, k * m, cudaMemcpyHostToDevice);
  cudaMemcpy(B_dev, b, n * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dequant_scale_dev, dequant_scale, n * sizeof(float), cudaMemcpyHostToDevice);



  int64_t lda = LayoutA::packed({m, k}).stride(0);
  int64_t ldb = LayoutB::packed({k, n}).stride(0);
  int64_t ldc = LayoutC::packed({m, n}).stride(0);

  std::cout << "lda " << lda << "\n";
  std::cout << "ldb " << ldb << "\n";
  std::cout << "ldc " << ldc << "\n";

  cutlass::gemm::GemmCoord problem_size(m, n, k);

  GemmDequant::Arguments args(
      problem_size,
      {A_dev, lda},
      {B_dev, ldb},
      {C_dev, ldc},
      {D_dev, ldc},
      {dequant_scale_dev, 0},
      {
        ElementCompute(1.0f),
        ElementCompute(0.0f)
      }
    );

    //
    // Launch
    //

    GemmDequant gemm;

    // Initialize
    auto status = gemm.initialize(args);
    if (status != cutlass::Status::kSuccess) {
      std::cout  << "status " <<  int(status) << "\n";
      return;
    }

    // Run
    status = gemm();

    std::cout  << "status " <<  int(status) << "\n";
    auto cuda_status = cudaDeviceSynchronize();
    std::cout  << "cuda_status " <<  int(cuda_status) << "\n";

    cudaMemcpy(c, D_dev, n * m * sizeof(float), cudaMemcpyDeviceToHost);
}

