/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**

*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
// #include "dequant_epilogue_visitor_per_row.h"
#include "dequant_epilogue_visitor.h"
#include "gemm_with_dequant_epilogue_visitor.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "dequant_epilogue_with_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {


/////////////////////////////////////////////////////////////////////////////////////////////////

///
template <
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename LayoutB_,
  typename ElementC_,
  typename ElementCompute_,
  typename OperatorClass_,
  typename ArchTag_,
  typename ThreadblockShape_,
  typename WarpShape_,
  typename InstructionShape_,
  typename EpilogueFunctorOp_,
  int kStages_,
  int AlignmentA_ = 128 / cutlass::sizeof_bits<ElementA_>::value,
  int AlignmentB_ = 128 / cutlass::sizeof_bits<ElementB_>::value
>
class GemmDequant {
public:

  ///////////////////////////////////////////////////////////////////////////////////////////////

  //
  // Type definitions
  //

  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;
  using ElementCompute = ElementCompute_;

  using LayoutA = LayoutA_;
  using LayoutB = LayoutB_;

  using EpilogueFunctorOp = EpilogueFunctorOp_;

  // These are mandatory layouts.
  using LayoutC = cutlass::layout::RowMajor;

  using TensorRefA = TensorRef<ElementA, LayoutA>;
  using TensorRefB = TensorRef<ElementB, LayoutB>;
  using TensorRefC = TensorRef<ElementC, LayoutC>;

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape        = WarpShape_;
  using InstructionShape = InstructionShape_;

  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;

  static int const kStages  = kStages_;
  static int const AlignmentA = AlignmentA_;
  static int const AlignmentB = AlignmentB_;

  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // basic GEMM kernel
  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemmUniversal<
  // using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
    ElementA,
    LayoutA,
    cutlass::ComplexTransform::kNone,
    AlignmentA,
    ElementB,
    LayoutB,
    cutlass::ComplexTransform::kNone,
    AlignmentB,
    ElementC,
    LayoutC,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueFunctorOp,
    ThreadblockSwizzle,
    kStages,
    cutlass::arch::OpMultiplyAddSaturate
    // typename cutlass::gemm::device::DefaultGemmConfiguration<
    //     OperatorClass, ArchTag, ElementA, ElementB, ElementC, ElementCompute>::Operator,
    // cutlass::gemm::SharedMemoryClearOption::kNone
  >::GemmKernel;

  ///////////////////////////////////////////////////////////////////////////////////////////////
  using ElementEpilogueCompute = float;
  using ElementEpilogueAcc     = int32_t;

  using DequantScaleIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
        cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
            typename DefaultGemmKernel::Epilogue::OutputTileIterator::ThreadMap::Shape,
            typename DefaultGemmKernel::Epilogue::OutputTileIterator::ThreadMap::Count,
            DefaultGemmKernel::Epilogue::OutputTileIterator::ThreadMap::kThreads,
            DefaultGemmKernel::Epilogue::OutputTileIterator::kElementsPerAccess,
            cutlass::sizeof_bits<ElementEpilogueCompute>::value>,
        ElementEpilogueCompute>;

  // Epilogue visitor
  using EpilogueVisitor = typename cutlass::epilogue::threadblock::DequantEpilogueVisitor<
    ThreadblockShape,
    DefaultGemmKernel::kThreadCount,
    DequantScaleIterator,
    typename DefaultGemmKernel::Epilogue::OutputTileIterator,
    ElementEpilogueAcc,
    ElementEpilogueCompute,
    EpilogueFunctorOp
  >;

  using ElementScale = typename EpilogueVisitor::AlphaScaleElementType;
  using LayoutScale = cutlass::layout::RowMajor;
  using TensorRefScale = TensorRef<ElementScale, LayoutScale>;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DequantEpilogueWithVisitorFromExistingEpilogue<
    EpilogueVisitor,
    typename DefaultGemmKernel::Epilogue
  >::Epilogue;

  // GEMM
  using GemmKernel = gemm::kernel::GemmWithEpilogueVisitor<
    typename DefaultGemmKernel::Mma,
    Epilogue,
    ThreadblockSwizzle
  >;


public:

  /// Arguments class
  struct Arguments {

    typename GemmKernel::Arguments         gemm;
    cutlass::gemm::GemmCoord extend;

    //
    // Methods
    //
    Arguments() { }

    Arguments(
      cutlass::gemm::GemmCoord problem_size,
      TensorRefA ref_A_,
      TensorRefB ref_B_,
      TensorRefC ref_C_,
      TensorRefC ref_D_,
      TensorRefScale ref_scale_,
      typename EpilogueFunctorOp::Params linear_scaling
    ):
      gemm(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        ref_A_,
        ref_B_,
        ref_C_,
        ref_D_,
        ref_scale_,
        typename EpilogueVisitor::Arguments(
          linear_scaling
        )
      ),
      extend(problem_size)
    {

    }
  };

  struct Params {

    typename GemmKernel::Params         gemm;
    MatrixCoord extend;
    //
    // Methods
    //
    Params() { }

    Params(Arguments const &args):
      gemm(args.gemm),
      extend(MatrixCoord(args.extend.m(), args.extend.n()))
    {

    }
  };

public:

  // Gemm


  //
  // Methods
  //

private:

  Params params_;

public:

  /// Ctor
  GemmDequant() {

  }

  /// Initialize
  Status initialize(Arguments const &args) {

    params_ = Params(args);

    return cutlass::Status::kSuccess;
  }

  /// Run
  Status run(cudaStream_t stream) {

    //
    // Launch the GEMM + max kernel
    //

    dim3 gemm_grid = ThreadblockSwizzle().get_grid_shape(params_.gemm.grid_tiled_shape);
    dim3 gemm_block(GemmKernel::kThreadCount, 1, 1);

    int gemm_smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    cudaError_t result;

    if (gemm_smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    gemm_smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::Kernel<GemmKernel><<<gemm_grid, gemm_block, gemm_smem_size, stream>>>(params_.gemm);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    return cutlass::Status::kSuccess;
  }

  /// Function call operator
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
