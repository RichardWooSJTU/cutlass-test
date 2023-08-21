
# include "gemm_dequant.h"
# include <vector>

int main() {
  int m = 1, k = 16, n = 2;
  std::vector<int8_t> a{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
  std::vector<int8_t> bt{ 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  std::vector<float> scale(n, 0.1);

  std::vector<float> c(m, n);

  RunGemmDequant(a.data(), bt.data(), scale.data(), c.data(), m, k, n);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << c[i * n + j] << " ";
    }
    std::cout << "\n";
  }

}