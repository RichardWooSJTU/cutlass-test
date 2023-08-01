#include <iostream>
#include <sys/time.h>
#include <vector>

#pragma once


template<typename T>
void Transpose(const T* a, T* b, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			b[j * rows + i] = a[i * cols + j];
		}
	}
	return;
}

template<typename InT, typename OutT>
OutT Elementwise(const InT* src1, const InT* src2, int l) {
    OutT dst = 0;
    for (int i = 0; i < l; ++i) {
        // std::cout << static_cast<OutT>(src2[i]) << std::endl;
        dst += static_cast<OutT>(src1[i]) * static_cast<OutT>(src2[i]);
    }
    return dst;
}

template <typename InT, typename OutT, bool TransposedB = false>
typename std::enable_if<!TransposedB, void>::type
GEMM(const std::vector<InT>& A, 
          const std::vector<InT>& B, 
          std::vector<OutT>& C, 
          int m, 
          int k, 
          int n) {
    // std::vector<InT> Bt(k * n);
    // Transpose(B.data(), Bt.data(), k, n);;
    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         C[i*n + j] = Elementwise<InT, OutT>(&A[i * k], &Bt[j * k], k);
    //     }
    // }
}

template <typename InT, typename OutT, bool TransposedB = false>
typename std::enable_if<TransposedB, void>::type
GEMM(const std::vector<InT>& A, 
          const std::vector<InT>& Bt, 
          std::vector<OutT>& C, 
          int m, 
          int k, 
          int n) {
    // std::vector<InT> Bt(k * n);
    // Transpose(B.data(), Bt.data(), k, n);;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i*n + j] = Elementwise<InT, OutT>(&A[i * k], &Bt[j * k], k);
        }
    }
}