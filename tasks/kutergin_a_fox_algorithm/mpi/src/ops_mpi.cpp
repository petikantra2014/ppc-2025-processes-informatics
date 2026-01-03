#include "kutergin_a_fox_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "kutergin_a_fox_algorithm/common/include/common.hpp"

namespace {

int RowsForRank(int rank, int rpp, int rem) {
  return (rank < rem) ? (rpp + 1) : rpp;
}

void CopyRows(const std::vector<std::vector<double>> &src, int start_row, int rows, int n, std::vector<double> &dst) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < n; ++j) {
      dst[(static_cast<std::size_t>(i) * n) + j] = src[start_row + i][j];
    }
  }
}

void CopyBlockToMatrix(const std::vector<double> &src, int rows, int n, int start_row,
                       std::vector<std::vector<double>> &dst) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < n; ++j) {
      dst[start_row + i][j] = src[(static_cast<std::size_t>(i) * n) + j];
    }
  }
}

}  // namespace

namespace kutergin_a_fox_algorithm {

FoxAlgorithmMPI::FoxAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool FoxAlgorithmMPI::CheckMatrices(const std::vector<std::vector<double>> &ma,
                                    const std::vector<std::vector<double>> &mb) {
  if (ma.empty() || mb.empty()) {
    return false;
  }

  const auto n = static_cast<int>(ma.size());

  if (!std::ranges::all_of(ma, [n](const auto &row) { return row.size() == static_cast<std::size_t>(n); })) {
    return false;
  }

  if (mb.size() != static_cast<std::size_t>(n)) {
    return false;
  }

  if (!std::ranges::all_of(mb, [n](const auto &row) { return row.size() == static_cast<std::size_t>(n); })) {
    return false;
  }

  return true;
}

void FoxAlgorithmMPI::SpreadB(int n, std::vector<double> &lb) {
  const auto &matrix_b = GetInput().second;

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      lb[(static_cast<std::size_t>(i) * n) + j] = matrix_b[i][j];
    }
  }

  MPI_Bcast(lb.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void FoxAlgorithmMPI::SpreadA(int rk, int sz, int n, int lr, std::vector<double> &la) {
  const auto &matrix_a = GetInput().first;

  const int rpp = n / sz;
  const int rem = n % sz;

  if (rk != 0) {
    MPI_Recv(la.data(), lr * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return;
  }

  int curr_row = 0;

  CopyRows(matrix_a, curr_row, lr, n, la);
  curr_row += lr;

  for (int dist = 1; dist < sz; ++dist) {
    const int rows = RowsForRank(dist, rpp, rem);
    std::vector<double> buf(static_cast<std::size_t>(rows) * n);

    CopyRows(matrix_a, curr_row, rows, n, buf);
    curr_row += rows;

    MPI_Send(buf.data(), rows * n, MPI_DOUBLE, dist, 0, MPI_COMM_WORLD);
  }
}

void FoxAlgorithmMPI::MultiplyBlocks(int n, int lr, const std::vector<double> &la, const std::vector<double> &lb,
                                     std::vector<double> &lc) {
  for (int i = 0; i < lr; ++i) {
    for (int k = 0; k < n; ++k) {
      const double tmp = la[(static_cast<std::size_t>(i) * n) + k];
      for (int j = 0; j < n; ++j) {
        lc[(static_cast<std::size_t>(i) * n) + j] += tmp * lb[(static_cast<std::size_t>(k) * n) + j];
      }
    }
  }
}

void FoxAlgorithmMPI::CollectResults(int rk, int sz, int n, int rpp, int rem, int lr, const std::vector<double> &lc) {
  if (rk != 0) {
    MPI_Send(lc.data(), lr * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    return;
  }

  CopyBlockToMatrix(lc, lr, n, 0, GetOutput());

  int curr_row = lr;

  for (int proc = 1; proc < sz; ++proc) {
    const int rows = RowsForRank(proc, rpp, rem);
    std::vector<double> buf(static_cast<std::size_t>(rows) * n);

    MPI_Recv(buf.data(), rows * n, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    CopyBlockToMatrix(buf, rows, n, curr_row, GetOutput());

    curr_row += rows;
  }
}

bool FoxAlgorithmMPI::ValidationImpl() {
  int rk = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rk);

  if (rk == 0) {
    return CheckMatrices(GetInput().first, GetInput().second);
  }

  return true;
}

bool FoxAlgorithmMPI::PreProcessingImpl() {
  return true;
}

bool FoxAlgorithmMPI::RunImpl() {
  int rk = 0;
  int sz = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rk);
  MPI_Comm_size(MPI_COMM_WORLD, &sz);

  int n = 0;
  if (rk == 0) {
    n = static_cast<int>(GetInput().first.size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n == 0) {
    return false;
  }

  const int rpp = n / sz;
  const int rem = n % sz;
  const int lr = RowsForRank(rk, rpp, rem);

  std::vector<double> lb(static_cast<std::size_t>(n) * n);
  SpreadB(n, lb);

  std::vector<double> la(static_cast<std::size_t>(lr) * n);
  SpreadA(rk, sz, n, lr, la);

  std::vector<double> lc(static_cast<std::size_t>(lr) * n, 0.0);
  MultiplyBlocks(n, lr, la, lb, lc);

  if (rk == 0) {
    GetOutput().assign(static_cast<std::size_t>(n), std::vector<double>(n));
  }

  CollectResults(rk, sz, n, rpp, rem, lr, lc);
  return true;
}

bool FoxAlgorithmMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kutergin_a_fox_algorithm
