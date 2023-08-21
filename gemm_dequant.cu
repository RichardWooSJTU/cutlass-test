
# include "gemm_dequant.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

py::array_t<float> RunGemmDequantPybind(py::array_t<int8_t>& a, py::array_t<int8_t>& b, py::array_t<float>& dequant_scale) {
  py::buffer_info buf1 = a.request();
  py::buffer_info buf2 = b.request();
  py::buffer_info scale_buf = dequant_scale.request();

  int m = buf1.shape[0];
  int k = buf1.shape[1];
  int n = buf2.shape[0];

  py::print(m, k, n);

  auto result = py::array_t<float>(m * n);
  result.resize({ m, n});
  py::buffer_info buf_result = result.request();

  auto* a_ptr = (int8_t*)buf1.ptr;
  auto* b_ptr = (int8_t*)buf2.ptr;
  auto* scale_ptr = (float*)scale_buf.ptr;
  auto* res_ptr = (float*)buf_result.ptr;

  RunGemmDequant(a_ptr, b_ptr, scale_ptr, res_ptr, m, k, n);
  return result;
}


PYBIND11_MODULE(cutlass, m) {
    m.doc() = "pybind11 cutlass gemm"; // optional module docstring
    m.def("gemm_dequant", &RunGemmDequantPybind, "A function exec gemm dequant");
}