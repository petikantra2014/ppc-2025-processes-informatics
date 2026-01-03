#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "kutergin_a_fox_algorithm/common/include/common.hpp"
#include "kutergin_a_fox_algorithm/mpi/include/ops_mpi.hpp"
#include "kutergin_a_fox_algorithm/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace kutergin_a_fox_algorithm {

class KuterginFoxAlgorithmFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto params = GetParam();
    TestType test_params = std::get<2>(params);

    test_id_ = std::get<0>(test_params);

    switch (test_id_) {
      case 1:
        GenerateMatrices(1);
        break;
      case 2:
        GenerateMatrices(2);
        break;
      case 3:
        GenerateMatrices(4);
        break;
      case 4:
        GenerateMatrices(8);
        break;
      case 5:
        GenerateMatrices(16);
        break;
      case 6:
        GenerateZero(10);
        break;
      case 7:
        GenerateIdentity(8);
        break;
      case 8:
        GenerateDiagonal(6);
        break;
      case 9:
        GenerateTriangular(7, true);
        break;
      case 10:
        GenerateTriangular(7, false);
        break;
      case 11:
        GenerateConstant(5, 2.0, 3.0);
        break;
      case 12:
        GenerateRandom(25, 123);
        break;
      case 13:
        GenerateSpecial(4);
        break;
      default:
        GenerateMatrices(2);
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i].size() != expected_[i].size()) {
        return false;
      }
      for (size_t j = 0; j < output_data[i].size(); ++j) {
        double exp = expected_[i][j];
        double act = output_data[i][j];
        if (std::isnan(exp) && std::isnan(act)) {
          continue;
        }
        if (std::abs(exp - act) > 1e-7) {
          return false;
        }
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return {ma_, mb_};
  }

 private:
  int test_id_ = 0;
  std::vector<std::vector<double>> ma_, mb_, expected_;

  void Compute() {
    int n = static_cast<int>(ma_.size());
    expected_.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
          expected_[i][j] += ma_[i][k] * mb_[k][j];
        }
      }
    }
  }

  void GenerateMatrices(int n) {
    ma_.assign(n, std::vector<double>(n, 1.0));
    mb_.assign(n, std::vector<double>(n, 2.0));
    Compute();
  }

  void GenerateZero(int n) {
    ma_.assign(n, std::vector<double>(n, 0.0));
    mb_.assign(n, std::vector<double>(n, 0.0));
    expected_.assign(n, std::vector<double>(n, 0.0));
  }

  void GenerateIdentity(int n) {
    ma_.assign(n, std::vector<double>(n, 0.0));
    mb_.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
      ma_[i][i] = mb_[i][i] = 1.0;
    }
    expected_ = ma_;
  }

  void GenerateDiagonal(int n) {
    ma_.assign(n, std::vector<double>(n, 0.0));
    mb_.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
      ma_[i][i] = static_cast<double>(i + 1);
      mb_[i][i] = 2.0;
    }
    Compute();
  }

  void GenerateTriangular(int n, bool upper) {
    ma_.assign(n, std::vector<double>(n, 0.0));
    mb_.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (upper && j >= i) {
          ma_[i][j] = mb_[i][j] = 1.0;
        }
        if (!upper && i >= j) {
          ma_[i][j] = mb_[i][j] = 1.0;
        }
      }
    }
    Compute();
  }

  void GenerateConstant(int n, double va, double vb) {
    ma_.assign(n, std::vector<double>(n, va));
    mb_.assign(n, std::vector<double>(n, vb));
    expected_.assign(n, std::vector<double>(n, va * vb * n));
  }

  void GenerateRandom(int n, int seed) {
    ma_.assign(n, std::vector<double>(n));
    mb_.assign(n, std::vector<double>(n));
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        ma_[i][j] = dist(gen);
        mb_[i][j] = dist(gen);
      }
    }
    Compute();
  }

  void GenerateSpecial(int n) {
    ma_.assign(n, std::vector<double>(n, 1.0));
    mb_.assign(n, std::vector<double>(n, 1.0));
    ma_[0][0] = std::numeric_limits<double>::quiet_NaN();
    Compute();
  }
};

TEST_P(KuterginFoxAlgorithmFuncTests, MatrixMultiplicationTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 13> kParams = {
    std::make_tuple(1, "1x1"),         std::make_tuple(2, "2x2"),       std::make_tuple(3, "4x4"),
    std::make_tuple(4, "8x8"),         std::make_tuple(5, "16x16"),     std::make_tuple(6, "zero"),
    std::make_tuple(7, "identity"),    std::make_tuple(8, "diagonal"),  std::make_tuple(9, "upper_tri"),
    std::make_tuple(10, "lower_tri"),  std::make_tuple(11, "constant"), std::make_tuple(12, "random"),
    std::make_tuple(13, "special_nan")};

const auto kTasks =
    std::tuple_cat(ppc::util::AddFuncTask<FoxAlgorithmMPI, InType>(kParams, PPC_SETTINGS_kutergin_a_fox_algorithm),
                   ppc::util::AddFuncTask<FoxAlgorithmSEQ, InType>(kParams, PPC_SETTINGS_kutergin_a_fox_algorithm));

INSTANTIATE_TEST_SUITE_P(KuterginFoxTests, KuterginFoxAlgorithmFuncTests, ppc::util::ExpandToValues(kTasks),
                         KuterginFoxAlgorithmFuncTests::PrintFuncTestName<KuterginFoxAlgorithmFuncTests>);

}  // namespace kutergin_a_fox_algorithm
