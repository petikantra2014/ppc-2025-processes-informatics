#include <gtest/gtest.h>

#include <vector>

#include "kutergin_a_fox_algorithm/common/include/common.hpp"
#include "kutergin_a_fox_algorithm/mpi/include/ops_mpi.hpp"
#include "kutergin_a_fox_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kutergin_a_fox_algorithm {

class KuterginFoxAlgorithmPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    this->input_data_ = InType{};
    const int sz = 512;

    std::vector<std::vector<double>> ma(sz, std::vector<double>(sz, 1.0));
    std::vector<std::vector<double>> mb(sz, std::vector<double>(sz, 2.0));

    input_data_ = {ma, mb};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Упрощенная проверка для perf: только размер и наличие данных
    if (output_data.empty()) {
      return false;
    }
    if (output_data.size() != 512) {
      return false;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(KuterginFoxAlgorithmPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, FoxAlgorithmMPI, FoxAlgorithmSEQ>(PPC_SETTINGS_kutergin_a_fox_algorithm);

INSTANTIATE_TEST_SUITE_P(KuterginFoxPerfTests, KuterginFoxAlgorithmPerfTests,
                         ppc::util::TupleToGTestValues(kAllPerfTasks),
                         KuterginFoxAlgorithmPerfTests::CustomPerfTestName);

}  // namespace kutergin_a_fox_algorithm
