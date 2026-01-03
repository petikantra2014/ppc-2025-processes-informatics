#pragma once

#include <utility>
#include <vector>

#include "kutergin_a_fox_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kutergin_a_fox_algorithm {

class FoxAlgorithmMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit FoxAlgorithmMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static bool CheckMatrices(const std::vector<std::vector<double>> &ma, const std::vector<std::vector<double>> &mb);
  void SpreadB(int n, std::vector<double> &lb);
  void SendRowsA(int target, int rpp, int rem, int n, int &curr);
  void SpreadA(int rk, int sz, int n, int lr, std::vector<double> &la);
  static void MultiplyBlocks(int n, int lr, const std::vector<double> &la, const std::vector<double> &lb,
                             std::vector<double> &lc);
  void CollectResults(int rk, int sz, int n, int rpp, int rem, int lr, const std::vector<double> &lc);
  void SyncFinalResult(int rk, int n);
};

}  // namespace kutergin_a_fox_algorithm
