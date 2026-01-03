#include "kutergin_a_fox_algorithm/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "kutergin_a_fox_algorithm/common/include/common.hpp"

namespace kutergin_a_fox_algorithm {

FoxAlgorithmSEQ::FoxAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool FoxAlgorithmSEQ::ValidationImpl() {
  const auto &matrix_a = GetInput().first;
  const auto &matrix_b = GetInput().second;

  if (matrix_a.empty() || matrix_b.empty()) {
    return false;
  }

  const auto n = static_cast<int>(matrix_a.size());
  for (int i = 0; i < n; ++i) {
    if (matrix_a[i].size() != static_cast<std::size_t>(n)) {
      return false;
    }
  }

  if (matrix_b.size() != static_cast<std::size_t>(n)) {
    return false;
  }

  for (int i = 0; i < n; ++i) {
    if (matrix_b[i].size() != static_cast<std::size_t>(n)) {
      return false;
    }
  }

  return true;
}

bool FoxAlgorithmSEQ::PreProcessingImpl() {
  return true;
}

void FoxAlgorithmSEQ::MultiplyBlock(int ars, int are, int acs, int ace, int bcs, int bce,
                                    const std::vector<std::vector<double>> &ma,
                                    const std::vector<std::vector<double>> &mb, std::vector<std::vector<double>> &mc) {
  for (int i = ars; i < are; ++i) {
    for (int k = acs; k < ace; ++k) {
      const double a_ik = ma[i][k];
      for (int j = bcs; j < bce; ++j) {
        mc[i][j] += a_ik * mb[k][j];
      }
    }
  }
}

bool FoxAlgorithmSEQ::RunImpl() {
  const auto &matrix_a = GetInput().first;
  const auto &matrix_b = GetInput().second;
  const auto n = static_cast<int>(matrix_a.size());

  std::vector<std::vector<double>> matrix_c(n, std::vector<double>(n, 0.0));

  int block_size = 64;
  block_size = std::min(n, block_size);
  const int grid_size = (n + block_size - 1) / block_size;

  for (int iter = 0; iter < grid_size; ++iter) {
    for (int block_i = 0; block_i < grid_size; ++block_i) {
      for (int block_j = 0; block_j < grid_size; ++block_j) {
        const int a_block_k = (block_i + iter) % grid_size;

        const int ars = block_i * block_size;
        const int are = std::min(ars + block_size, n);
        const int acs = a_block_k * block_size;
        const int ace = std::min(acs + block_size, n);
        const int bcs = block_j * block_size;
        const int bce = std::min(bcs + block_size, n);

        MultiplyBlock(ars, are, acs, ace, bcs, bce, matrix_a, matrix_b, matrix_c);
      }
    }
  }

  GetOutput() = std::move(matrix_c);
  return true;
}

bool FoxAlgorithmSEQ::PostProcessingImpl() {
  const auto &matrix_c = GetOutput();
  return !matrix_c.empty();
}

}  // namespace kutergin_a_fox_algorithm
