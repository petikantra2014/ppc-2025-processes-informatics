#pragma once

#include <vector>

#include "kutergin_a_fox_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kutergin_a_fox_algorithm {

class FoxAlgorithmSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit FoxAlgorithmSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void MultiplyBlock(int ars, int are, int acs, int ace, int bcs, int bce,
                            const std::vector<std::vector<double>> &ma, const std::vector<std::vector<double>> &mb,
                            std::vector<std::vector<double>> &mc);
};

}  // namespace kutergin_a_fox_algorithm
